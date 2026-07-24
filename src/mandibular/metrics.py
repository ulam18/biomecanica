"""
Calculo das metricas biomecanicas do movimento mandibular.

Todas as medidas de distancia sao normalizadas por uma referencia facial
estavel (distancia entre os cantos externos dos olhos), o que torna as
metricas aproximadamente invariantes a distancia entre o rosto e a camera.

Convencoes (assumindo cabeca estavel, conforme limitacoes do projeto):
    - Constroi-se um referencial da face a partir do eixo inter-ocular:
        x_face -> direcao horizontal (canto esq. -> canto dir. do olho);
        y_face -> perpendicular, apontando para baixo.
    - A abertura bucal e a componente vertical (y_face) da distancia entre os
      labios interno superior e inferior.
    - O desvio lateral e a componente horizontal (x_face) da posicao do queixo
      em relacao a raiz do nariz (nasion), ponto proximo da linha media.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import CycleConfig, Landmark
from .landmarks import FaceLandmarks


class MovementState(str, Enum):
    """Estado instantaneo do movimento mandibular."""
    FECHADO = "fechado"
    ABRINDO = "abrindo"
    ABERTO = "aberto"
    FECHANDO = "fechando"


@dataclass
class FrameMetrics:
    """Metricas calculadas para um unico frame."""
    opening_px: float        # abertura bucal bruta, em pixels
    opening_rel: float       # abertura bucal (unidades de referencia facial)
    lateral_px: float        # desvio lateral do queixo, em pixels (com sinal)
    lateral_rel: float       # desvio lateral do queixo (com sinal; + = eixo x_face positivo)
    face_width_px: float     # referencia facial em pixels (para escala/mm)
    opening_mm: float | None # abertura em mm (se houver calibracao)
    lateral_mm: float | None # desvio lateral em mm (se houver calibracao)


def lateral_direction(lateral_rel: float, mirrored: bool, deadzone: float = 0.02) -> str:
    """
    Traduz o sinal do desvio lateral em "direita"/"esquerda"/"centro", do
    ponto de vista do proprio paciente.

    Quando a imagem e espelhada (`mirrored=True`, o padrao, usado para dar
    uma experiencia de "espelho" mais intuitiva ao usuario), o lado positivo
    do eixo x_face corresponde a direita do paciente. Sem espelhamento
    (camera "de frente", como uma foto tirada por outra pessoa), essa relacao
    se inverte. Sem essa correcao, o rotulo mostrado na tela ficaria trocado
    sempre que `--no-flip` fosse usado.
    """
    if abs(lateral_rel) < deadzone:
        return "centro"
    positive_is_right = mirrored
    if lateral_rel > 0:
        return "direita" if positive_is_right else "esquerda"
    return "esquerda" if positive_is_right else "direita"


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros_like(v)
    return v / n


def compute_frame_metrics(
    face: FaceLandmarks,
    reference_distance_mm: float | None = None,
) -> FrameMetrics:
    """
    Calcula as metricas biomecanicas de um frame a partir dos landmarks.

    A referencia facial e a distancia entre os cantos externos dos olhos,
    que nao e afetada pela abertura da boca.
    """
    eye_l = face.point(Landmark.EYE_OUTER_LEFT)
    eye_r = face.point(Landmark.EYE_OUTER_RIGHT)

    face_width = float(np.linalg.norm(eye_r - eye_l))
    if face_width < 1e-6:
        face_width = 1e-6

    # Referencial da face (robusto a rotacao no plano da imagem / roll).
    x_face = _unit(eye_r - eye_l)          # horizontal
    y_face = np.array([-x_face[1], x_face[0]])  # perpendicular (para baixo na imagem)

    # --- Abertura bucal: componente vertical entre labios internos ---
    upper = face.point(Landmark.UPPER_LIP_INNER)
    lower = face.point(Landmark.LOWER_LIP_INNER)
    opening_px = abs(float(np.dot(lower - upper, y_face)))
    opening_rel = opening_px / face_width

    # --- Desvio lateral: componente horizontal do queixo vs. nasion ---
    chin = face.point(Landmark.CHIN)
    nasion = face.point(Landmark.NASION)
    lateral_px = float(np.dot(chin - nasion, x_face))
    lateral_rel = lateral_px / face_width

    # --- Conversao opcional para milimetros ---
    opening_mm = lateral_mm = None
    if reference_distance_mm is not None and reference_distance_mm > 0:
        mm_per_px = reference_distance_mm / face_width
        opening_mm = opening_px * mm_per_px
        lateral_mm = lateral_px * mm_per_px

    return FrameMetrics(
        opening_px=opening_px,
        opening_rel=opening_rel,
        lateral_px=lateral_px,
        lateral_rel=lateral_rel,
        face_width_px=face_width,
        opening_mm=opening_mm,
        lateral_mm=lateral_mm,
    )


@dataclass
class Cycle:
    """Um ciclo completo de abertura/fechamento."""
    start_time: float
    end_time: float
    peak_opening: float      # abertura maxima relativa no ciclo

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class CycleDetector:
    """
    Detecta ciclos de abertura/fechamento a partir do sinal de abertura.

    Funciona com uma maquina de estados e histerese. Os limiares sao definidos
    em relacao a faixa (baseline..pico) capturada na calibracao. Se nao houver
    calibracao, uma faixa dinamica (min/max observados) e usada.
    """

    def __init__(self, config: CycleConfig | None = None):
        self.config = config or CycleConfig()
        self.state = MovementState.FECHADO
        self.cycles: list[Cycle] = []

        # Faixa de referencia para os limiares.
        self._baseline: float | None = None   # abertura com boca fechada
        self._span: float | None = None        # baseline -> pico calibrado

        # Faixa dinamica (fallback sem calibracao).
        self._dyn_min = float("inf")
        self._dyn_max = float("-inf")

        # Estado do ciclo em andamento.
        self._cycle_start_t: float | None = None
        self._cycle_peak = 0.0

    # -- Calibracao -------------------------------------------------------
    def calibrate(self, closed_value: float, open_value: float) -> None:
        """Define a faixa com base em amostras de boca fechada e aberta."""
        self._baseline = closed_value
        self._span = max(open_value - closed_value, 1e-6)

    @property
    def is_calibrated(self) -> bool:
        return self._baseline is not None and self._span is not None

    @property
    def baseline(self) -> float | None:
        """Abertura de referencia (boca fechada) usada nos limiares."""
        return self._baseline

    @property
    def span(self) -> float | None:
        """Faixa (fechado -> aberto) usada para escalar os limiares."""
        return self._span

    def _thresholds(self, opening: float) -> tuple[float, float]:
        """Retorna (limiar_abrir, limiar_fechar) em unidades de abertura."""
        if self.is_calibrated:
            base, span = self._baseline, self._span
        else:
            # Faixa dinamica adaptativa.
            self._dyn_min = min(self._dyn_min, opening)
            self._dyn_max = max(self._dyn_max, opening)
            base = self._dyn_min
            span = max(self._dyn_max - self._dyn_min, 1e-6)
        open_th = base + self.config.open_fraction * span
        close_th = base + self.config.close_fraction * span
        return open_th, close_th

    # -- Atualizacao por frame -------------------------------------------
    def update(self, opening: float, t: float) -> bool:
        """
        Atualiza a maquina de estados com a abertura atual no instante t.
        Retorna True se um ciclo foi concluido neste frame.
        """
        open_th, close_th = self._thresholds(opening)
        completed = False

        if self.state in (MovementState.FECHADO, MovementState.FECHANDO):
            if opening >= open_th:
                # Inicio de uma nova abertura.
                self.state = MovementState.ABRINDO
                self._cycle_start_t = t
                self._cycle_peak = opening
            else:
                self.state = MovementState.FECHADO
        elif self.state in (MovementState.ABRINDO, MovementState.ABERTO):
            self._cycle_peak = max(self._cycle_peak, opening)
            if opening >= open_th:
                self.state = MovementState.ABERTO
            if opening <= close_th:
                # Fechou: fecha o ciclo.
                self.state = MovementState.FECHANDO
                if self._cycle_start_t is not None:
                    duration = t - self._cycle_start_t
                    if duration >= self.config.min_cycle_seconds:
                        self.cycles.append(
                            Cycle(
                                start_time=self._cycle_start_t,
                                end_time=t,
                                peak_opening=self._cycle_peak,
                            )
                        )
                        completed = True
                self._cycle_start_t = None
                self._cycle_peak = 0.0

        return completed

    # -- Estatisticas -----------------------------------------------------
    @property
    def repetitions(self) -> int:
        return len(self.cycles)

    def repeatability(self) -> dict[str, float]:
        """
        Metricas de repetibilidade entre os ciclos detectados.

        Retorna medias, desvios-padrao e coeficiente de variacao (CV) da
        amplitude e da duracao. CV baixo indica movimento mais repetivel.
        """
        if not self.cycles:
            return {}

        peaks = np.array([c.peak_opening for c in self.cycles], dtype=float)
        durations = np.array([c.duration for c in self.cycles], dtype=float)

        def cv(arr: np.ndarray) -> float:
            m = float(np.mean(arr))
            return float(np.std(arr) / m) if m > 1e-9 else 0.0

        return {
            "n_ciclos": float(len(self.cycles)),
            "amplitude_media": float(np.mean(peaks)),
            "amplitude_dp": float(np.std(peaks)),
            "amplitude_cv": cv(peaks),
            "duracao_media_s": float(np.mean(durations)),
            "duracao_dp_s": float(np.std(durations)),
            "duracao_cv": cv(durations),
        }
