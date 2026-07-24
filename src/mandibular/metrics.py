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

from .config import CycleConfig, Landmark, SYMMETRIC_PAIRS
from .landmarks import FaceLandmarks

# Escala do indice de simetria: um desvio medio (relativo a largura facial)
# igual a este valor leva o indice a 0. 0.15 (~15%) e generoso: faces normais
# ficam bem acima de 0.9.
SYMMETRY_SCALE = 0.15


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


# ===========================================================================
# Analise facial frontal (simetria/proporcoes) e de perfil (angulos)
# Modulos adicionados sobre a arquitetura de pipeline: funcoes puras que
# recebem FaceLandmarks e nao dependem do recorder/app.
# ===========================================================================
def _face_frame(face: FaceLandmarks) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Constroi o referencial da face a partir do eixo inter-ocular.

    Retorna (largura_facial_px, x_face, y_face), onde x_face aponta do canto
    externo do olho esquerdo para o direito e y_face e perpendicular, para
    baixo na imagem.
    """
    eye_l = face.point(Landmark.EYE_OUTER_LEFT)
    eye_r = face.point(Landmark.EYE_OUTER_RIGHT)
    face_width = float(np.linalg.norm(eye_r - eye_l))
    if face_width < 1e-6:
        face_width = 1e-6
    x_face = _unit(eye_r - eye_l)
    y_face = np.array([-x_face[1], x_face[0]], dtype=np.float32)
    return face_width, x_face, y_face


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angulo (graus) no vertice b, formado pelos segmentos b->a e b->c."""
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _cant_deg(face: FaceLandmarks) -> float:
    """
    Angulo (graus) entre a linha dos olhos (bipupilar) e a linha dos cantos da
    boca (intercomissural). ~0 quando paralelas (face simetrica frontal).
    """
    eye_axis = face.point(Landmark.EYE_OUTER_RIGHT) - face.point(Landmark.EYE_OUTER_LEFT)
    mouth_axis = face.point(Landmark.MOUTH_RIGHT) - face.point(Landmark.MOUTH_LEFT)
    a_eye = np.arctan2(eye_axis[1], eye_axis[0])
    a_mouth = np.arctan2(mouth_axis[1], mouth_axis[0])
    cant = np.degrees(a_mouth - a_eye)
    return float((cant + 180.0) % 360.0 - 180.0)  # normaliza para [-180, 180]


@dataclass
class SymmetryMetrics:
    """
    Metricas de simetria facial na vista frontal.

    - `index` (0..1): 1 = perfeitamente simetrico.
    - `by_region`: indice por regiao (olhos, nariz, boca, face).
    - `midline_offset_rel`: deslocamento medio (com sinal) dos pares em relacao
      a linha media (confunde assimetria real com rotacao; por isso o yaw e
      reportado a parte).
    - `yaw_proxy`: estimativa da rotacao horizontal da cabeca (~0 = frontal).
    - `is_frontal`: True se |yaw_proxy| estiver dentro da tolerancia.
    - `cant_deg`: angulo entre a linha bipupilar e a intercomissural.
    """
    index: float
    by_region: dict[str, float]
    midline_offset_rel: float
    yaw_proxy: float
    is_frontal: bool
    cant_deg: float


def compute_symmetry(
    face: FaceLandmarks, frontal_yaw_tol: float = 0.12
) -> SymmetryMetrics:
    """
    Calcula a simetria facial a partir de pares de landmarks espelhados.

    IMPORTANTE: a simetria so e interpretavel na vista FRONTAL. Quando a cabeca
    esta girada (`is_frontal` = False), a projecao 2D distorce os pares.
    """
    face_width, x_face, y_face = _face_frame(face)
    nasion = face.point(Landmark.NASION)

    def uv(idx: int) -> tuple[float, float]:
        p = face.point(idx) - nasion
        return float(np.dot(p, x_face)), float(np.dot(p, y_face))

    region_devs: dict[str, list[float]] = {}
    offsets: list[float] = []
    for region, left_idx, right_idx in SYMMETRIC_PAIRS:
        u_l, v_l = uv(left_idx)
        u_r, v_r = uv(right_idx)
        horiz = (u_l + u_r)            # ~0 se espelhados
        vert = (v_l - v_r)             # ~0 se na mesma altura
        dev = float(np.hypot(horiz, vert)) / face_width
        region_devs.setdefault(region, []).append(dev)
        offsets.append((u_l + u_r) / 2.0 / face_width)

    def dev_to_index(dev: float) -> float:
        return float(np.clip(1.0 - dev / SYMMETRY_SCALE, 0.0, 1.0))

    by_region = {
        r: dev_to_index(float(np.mean(devs))) for r, devs in region_devs.items()
    }
    all_devs = [d for devs in region_devs.values() for d in devs]
    index = dev_to_index(float(np.mean(all_devs)))
    midline_offset_rel = float(np.mean(offsets))

    if face.has_depth:
        z_l = face.z(Landmark.EYE_OUTER_LEFT)
        z_r = face.z(Landmark.EYE_OUTER_RIGHT)
        yaw_proxy = (z_l - z_r) / face_width
    else:
        u_nose, _ = uv(Landmark.NOSE_TIP)
        yaw_proxy = u_nose / (face_width / 2.0)

    is_frontal = abs(yaw_proxy) <= frontal_yaw_tol

    return SymmetryMetrics(
        index=index,
        by_region=by_region,
        midline_offset_rel=midline_offset_rel,
        yaw_proxy=float(yaw_proxy),
        is_frontal=bool(is_frontal),
        cant_deg=_cant_deg(face),
    )


@dataclass
class FrontalAngles:
    """
    Angulos da vista frontal (medidas relativas/intrasujeito -- SEM corte
    clinico; a literatura trata toda face como assimetrica).

    - `mand_deviation_deg`: inclinacao (com sinal) da linha media mandibular
      (nasio -> mento) vs vertical da linha media facial. 0 = mento alinhado;
      + = desvio para a direita da imagem. Base do biofeedback de evolucao.
    - `cant_deg`: angulo entre a linha dos olhos e a linha da boca.
    """
    mand_deviation_deg: float
    cant_deg: float


def compute_frontal_angles(face: FaceLandmarks) -> FrontalAngles:
    """Calcula o angulo de desvio mandibular e o cant na vista frontal."""
    _, x_face, y_face = _face_frame(face)
    v = face.point(Landmark.CHIN) - face.point(Landmark.NASION)
    u = float(np.dot(v, x_face))
    w = float(np.dot(v, y_face))
    mand = float(np.degrees(np.arctan2(u, w)))
    return FrontalAngles(mand_deviation_deg=mand, cant_deg=_cant_deg(face))


@dataclass
class ProportionMetrics:
    """
    Proporcoes faciais classicas (vista frontal, estimativas 2D). Os valores
    "ideais" sao canones esteticos, NAO limiares clinicos.

    - `thirds_ratio`: (Glabela->Subnasal)/(Subnasal->Mento). Canon = 1.0.
    - `fifths_ratio`: largura facial / largura de um olho. Canon = 5.0.
    - `intercanthal_alar_ratio`: intercantal medial / interalar. Canon ~1.0.
    - `mouth_within_canon`: interalar < intercomissural < interpupilar.
    """
    thirds_ratio: float
    fifths_ratio: float
    intercanthal_alar_ratio: float
    mouth_within_canon: bool


def compute_proportions(face: FaceLandmarks) -> ProportionMetrics:
    """Calcula proporcoes faciais frontais (tercos, quintos, canones)."""
    def dist(i: int, j: int) -> float:
        return float(np.linalg.norm(face.point(i) - face.point(j)))

    mid_third = dist(Landmark.GLABELA, Landmark.SUBNASALE)
    low_third = dist(Landmark.SUBNASALE, Landmark.CHIN)
    thirds_ratio = mid_third / low_third if low_third > 1e-6 else 0.0

    face_width = dist(Landmark.CHEEK_LEFT, Landmark.CHEEK_RIGHT)
    eye_width = dist(Landmark.EYE_OUTER_LEFT, Landmark.EYE_INNER_LEFT)
    fifths_ratio = face_width / eye_width if eye_width > 1e-6 else 0.0

    intercanthal = dist(Landmark.EYE_INNER_LEFT, Landmark.EYE_INNER_RIGHT)
    interalar = dist(Landmark.NOSE_ALA_LEFT, Landmark.NOSE_ALA_RIGHT)
    intercanthal_alar_ratio = intercanthal / interalar if interalar > 1e-6 else 0.0

    intercommissural = dist(Landmark.MOUTH_LEFT, Landmark.MOUTH_RIGHT)
    interpupillary = dist(Landmark.EYE_OUTER_LEFT, Landmark.EYE_OUTER_RIGHT)
    mouth_within_canon = interalar < intercommissural < interpupillary

    return ProportionMetrics(
        thirds_ratio=thirds_ratio,
        fifths_ratio=fifths_ratio,
        intercanthal_alar_ratio=intercanthal_alar_ratio,
        mouth_within_canon=bool(mouth_within_canon),
    )


@dataclass
class ProfileMetrics:
    """
    Metricas do rosto em vista de perfil (plano sagital) -- ESTIMATIVAS. A
    simetria E->D NAO e mensuravel de perfil.
    """
    facing: str
    chin_projection_rel: float
    nose_projection_rel: float
    convexity_deg: float
    opening_rel: float
    yaw_proxy: float


def compute_profile_metrics(face: FaceLandmarks) -> ProfileMetrics:
    """Calcula metricas sagitais do rosto de perfil."""
    face_width, x_face, y_face = _face_frame(face)
    nasion = face.point(Landmark.NASION)
    chin = face.point(Landmark.CHIN)

    face_height = float(np.linalg.norm(chin - nasion))
    if face_height < 1e-6:
        face_height = 1e-6

    def horiz(idx: int) -> float:
        return float(np.dot(face.point(idx) - nasion, x_face))

    nose_h = horiz(Landmark.NOSE_TIP)
    chin_h = horiz(Landmark.CHIN)
    facing = "direito" if nose_h >= 0 else "esquerdo"

    def uv_pt(idx: int) -> np.ndarray:
        p = face.point(idx) - nasion
        return np.array([np.dot(p, x_face), np.dot(p, y_face)], dtype=np.float32)

    convexity = _angle_deg(
        uv_pt(Landmark.FOREHEAD), uv_pt(Landmark.SUBNASALE), uv_pt(Landmark.CHIN)
    )

    upper = face.point(Landmark.UPPER_LIP_INNER)
    lower = face.point(Landmark.LOWER_LIP_INNER)
    opening_rel = abs(float(np.dot(lower - upper, y_face))) / face_width

    if face.has_depth:
        z_l = face.z(Landmark.EYE_OUTER_LEFT)
        z_r = face.z(Landmark.EYE_OUTER_RIGHT)
        yaw_proxy = (z_l - z_r) / face_width
    else:
        yaw_proxy = nose_h / (face_width / 2.0)

    return ProfileMetrics(
        facing=facing,
        chin_projection_rel=chin_h / face_height,
        nose_projection_rel=nose_h / face_height,
        convexity_deg=convexity,
        opening_rel=opening_rel,
        yaw_proxy=float(yaw_proxy),
    )


@dataclass
class ProfileAngles:
    """
    Angulos de tecido mole do perfil (graus) -- ESTIMATIVAS 2D.

    - `convexidade_facial`: G-Sn-Pg.
    - `convexidade_total`: G-Prn-Pg (inclui o nariz).
    - `nasofrontal`: G-N-Prn.
    - `nasolabial`: ~Prn-Sn-Ls.
    - `labiomental`: Li-Sm-Pg.
    """
    convexidade_facial: float
    convexidade_total: float
    nasofrontal: float
    nasolabial: float
    labiomental: float


def compute_profile_angles(face: FaceLandmarks) -> ProfileAngles:
    """Calcula os angulos de perfil de tecido mole."""
    g = face.point(Landmark.GLABELA)
    n = face.point(Landmark.NASION)
    prn = face.point(Landmark.NOSE_TIP)
    sn = face.point(Landmark.SUBNASALE)
    pg = face.point(Landmark.CHIN)
    ls = face.point(Landmark.LABIALE_SUP)
    li = face.point(Landmark.LABIALE_INF)
    sm = face.point(Landmark.SUBLABIALE)
    return ProfileAngles(
        convexidade_facial=_angle_deg(g, sn, pg),
        convexidade_total=_angle_deg(g, prn, pg),
        nasofrontal=_angle_deg(g, n, prn),
        nasolabial=_angle_deg(prn, sn, ls),
        labiomental=_angle_deg(li, sm, pg),
    )
