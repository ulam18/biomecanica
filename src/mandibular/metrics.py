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

# Rotulo/descricao da unidade "relativa" usada em abertura e lateralidade
# (absoluta e dinamica): o denominador e SEMPRE a distancia interocular
# (norm(EYE_OUTER_RIGHT - EYE_OUTER_LEFT), cantos externos dos olhos) -- NAO
# a largura total do rosto (ex.: bochecha a bochecha) nem qualquer outra
# distancia. Ver `compute_frame_metrics` / `_face_frame`. Usado em HUD,
# graficos e metadados para nao chamar de "largura facial" o que na verdade
# e a distancia interocular.
REL_UNIT_LABEL = "adimensional (norm. pela dist. interocular)"
REL_UNIT_DESCRIPTION = (
    "adimensional - normalizado pela distancia interocular "
    "(cantos externos dos olhos, landmarks 33-263)"
)

# Escala do indice de simetria: um desvio medio (relativo a distancia
# interocular, cantos externos dos olhos) igual a este valor leva o indice a
# 0. 0.15 (~15%) e generoso: faces normais ficam bem acima de 0.9.
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


@dataclass
class HeadPose:
    """
    Estimativa da orientacao da cabeca (roll/yaw/pitch), em graus.

    - `roll_deg`: inclinacao no plano da imagem (eixo inter-ocular vs
      horizontal). Calculo 2D direto, exato, sempre disponivel.
    - `yaw_deg`/`pitch_deg`: proxies a partir da profundidade relativa (z) por
      landmark que o MediaPipe Face Landmarker fornece (uma unica camera, sem
      estereo). Servem apenas para o controle de qualidade (gate de "virado/
      inclinado demais"), NAO sao medidas clinicas de angulo. `None` quando o
      frame nao tem profundidade disponivel (ex.: landmarks sinteticos de
      teste) -- nesse caso yaw/pitch nao entram no controle de qualidade.
    - `yaw_proxy`/`pitch_proxy`: valores adimensionais antes da conversao
      para graus (uteis para depuracao/testes); tambem `None` se indisponivel.
    """
    roll_deg: float
    yaw_deg: float | None
    pitch_deg: float | None
    yaw_proxy: float | None
    pitch_proxy: float | None


def estimate_head_pose(face: FaceLandmarks) -> HeadPose:
    """
    Estima roll/yaw/pitch da cabeca a partir dos landmarks de um unico frame.

    Sem uma segunda camera (visao frontal apenas), yaw e pitch nao podem ser
    medidos com precisao geometrica; usa-se a profundidade relativa (z) que o
    proprio MediaPipe fornece por landmark como aproximacao (mesma fonte ja
    usada por `compute_symmetry` para o yaw). Quando o frame nao tem
    profundidade (`face.has_depth` False), yaw/pitch ficam `None` -- nao ha
    base confiavel para estima-los so em 2D a partir de um unico ponto (ex.:
    a ponta do nariz pode estar ausente/nao mapeada em landmarks sinteticos).
    """
    face_width, _x_face, _y_face = _face_frame(face)
    eye_l = face.point(Landmark.EYE_OUTER_LEFT)
    eye_r = face.point(Landmark.EYE_OUTER_RIGHT)
    dx, dy = eye_r - eye_l
    roll_deg = float(np.degrees(np.arctan2(dy, dx)))

    if not face.has_depth:
        return HeadPose(
            roll_deg=roll_deg, yaw_deg=None, pitch_deg=None,
            yaw_proxy=None, pitch_proxy=None,
        )

    nasion = face.point(Landmark.NASION)
    chin = face.point(Landmark.CHIN)
    face_height = float(np.linalg.norm(chin - nasion))
    if face_height < 1e-6:
        face_height = 1e-6

    yaw_proxy = (
        face.z(Landmark.EYE_OUTER_LEFT) - face.z(Landmark.EYE_OUTER_RIGHT)
    ) / face_width
    pitch_proxy = (
        face.z(Landmark.FOREHEAD) - face.z(Landmark.CHIN)
    ) / face_height

    yaw_deg = float(np.degrees(np.arcsin(np.clip(yaw_proxy, -1.0, 1.0))))
    pitch_deg = float(np.degrees(np.arcsin(np.clip(pitch_proxy, -1.0, 1.0))))

    return HeadPose(
        roll_deg=roll_deg,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        yaw_proxy=float(yaw_proxy),
        pitch_proxy=float(pitch_proxy),
    )


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
    """
    Um ciclo completo de abertura/fechamento, com as metricas laterais
    associadas (secao 5 do escopo de refinamento frontal).

    - `baseline_opening`: referencia calibrada de "boca fechada" (usada para
      `amplitude` = pico - referencia).
    - `start_opening`/`end_opening`: abertura na amostra usada como inicio e
      fim EXPORTADOS do ciclo (ver `closed_baseline_limit` abaixo -- nao sao
      necessariamente os frames de cruzamento dos limiares de histerese da
      maquina de estados).
    - `closed_baseline_limit`: banda de "boca realmente fechada"
      (baseline + boundary_closed_fraction*span) usada para validar
      inicio/fim, DISTINTA dos limiares de abertura/fechamento (60%/25%) da
      maquina de estados (esses continuam existindo so para a histerese).
    - `start_within_baseline`/`end_within_baseline`: True se a amostra de
      inicio/fim de fato estiver dentro dessa banda (False so no caso-limite
      em que a sessao comecou/terminou sem material suficiente no pre-buffer
      -- nesse caso NAO se deve descrever o ciclo como "fechado" na ponta
      correspondente).
    - `lateral_*_at_peak`: valor absoluto/dinamico no instante do PICO de
      abertura (nao no inicio/fim do ciclo).
    - `lateral_dynamic_max/min/abs_max/mean`: estatisticas de `lateral_dynamic`
      ao longo de TODO o ciclo (abrindo->fechando).
    """
    cycle_id: int
    start_time: float
    end_time: float
    peak_time: float
    peak_opening: float          # abertura maxima relativa no ciclo (valor bruto de pico)
    baseline_opening: float      # referencia "fechado" calibrada (para amplitude)
    start_opening: float         # abertura na amostra de inicio exportada
    end_opening: float           # abertura na amostra de fim exportada
    closed_baseline_limit: float
    start_within_baseline: bool
    end_within_baseline: bool
    lateral_absolute_at_peak: float | None = None
    lateral_dynamic_at_peak: float | None = None
    lateral_dynamic_max: float | None = None
    lateral_dynamic_min: float | None = None
    lateral_dynamic_abs_max: float | None = None
    lateral_dynamic_mean: float | None = None
    # -- Origem do inicio exportado (secao 1/2 do refinamento de precisao) --
    start_origin: str = "last_stable_closed_sample"  # ou "fallback"
    start_fallback_reason: str | None = None  # motivo, quando start_origin=="fallback"
    anchor_confirm_gap_s: float = 0.0  # tempo entre a ancora fechada e a confirmacao (abertura lenta)
    # -- Direcao pelo plato de abertura maxima (secao 3) ---------------------
    lateral_dynamic_median_plateau: float | None = None
    plateau_sample_count: int = 0
    plateau_used_fallback: bool = False

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def amplitude(self) -> float:
        """Abertura real do ciclo (pico - referencia de boca fechada)."""
        return self.peak_opening - self.baseline_opening

    @property
    def opening_velocity(self) -> float:
        """Velocidade media de abertura (amplitude / tempo ate o pico)."""
        dt = self.peak_time - self.start_time
        return self.amplitude / dt if dt > 1e-6 else 0.0

    @property
    def closing_velocity(self) -> float:
        """Velocidade media de fechamento (queda desde o pico / tempo apos o pico)."""
        dt = self.end_time - self.peak_time
        return (self.peak_opening - self.end_opening) / dt if dt > 1e-6 else 0.0


def cycle_predominant_direction(
    cycle: Cycle, mirrored: bool, deadzone: float = 0.02
) -> tuple[str, float]:
    """
    Direcao anatomica predominante do ciclo.

    CRITERIO (secao 3 do refinamento de precisao): usa a MEDIANA de
    `lateral_dynamic_filtered` no PLATO de abertura maxima (amostras com
    abertura >= 95% do pico do ciclo), nao o valor de um unico frame no
    pico exato -- um so frame ruidoso podia decidir a classificacao por
    margens minimas (caso real: ciclo classificado "direita" por 0.0006
    acima da deadzone so por causa do frame do pico; a mediana do plato
    ficava claramente dentro da zona "centro"). Mediana (nao media) para
    resistir a outliers isolados dentro do proprio plato.

    `cycle.lateral_dynamic_median_plateau` ja vem calculado por
    `CycleDetector._finish_cycle` (com fallback para uma janela temporal ao
    redor do pico quando o plato tem poucas amostras -- ver
    CycleConfig.direction_min_plateau_samples/direction_plateau_fallback_window_s).
    Se nao houver plato nem fallback com dado (ciclo sem `lateral_dynamic`
    disponivel), cai para `lateral_dynamic_at_peak` e depois `lateral_dynamic_mean`.

    O valor e comparado a `deadzone` (tipicamente `effective_direction_deadzone`,
    adaptativo ao ruido da calibracao) usando a MESMA convencao de sinal da
    interface/CSV. Retorna (direcao, valor_usado_na_classificacao) -- o valor
    usado e sempre reportado junto (ver exporter._cycle_summaries), nunca fica
    sem explicacao.
    """
    value = cycle.lateral_dynamic_median_plateau
    if value is None:
        value = cycle.lateral_dynamic_at_peak
    if value is None:
        value = cycle.lateral_dynamic_mean if cycle.lateral_dynamic_mean is not None else 0.0
    return lateral_direction(value, mirrored, deadzone=deadzone), value


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
        self._lateral_baseline: float | None = None  # lateral_absolute neutro (boca fechada)
        self._lateral_baseline_std: float | None = None  # ruido do baseline lateral (calibracao)

        # Faixa dinamica (fallback sem calibracao).
        self._dyn_min = float("inf")
        self._dyn_max = float("-inf")

        # Estado do ciclo em andamento.
        self._cycle_start_t: float | None = None
        self._cycle_start_opening = 0.0
        self._cycle_peak = 0.0
        self._cycle_peak_time = 0.0
        self._cycle_baseline_opening = 0.0
        self._cycle_closed_limit = 0.0
        self._cycle_start_origin = "last_stable_closed_sample"
        self._cycle_start_fallback_reason: str | None = None
        self._cycle_anchor_confirm_gap_s = 0.0
        self._cyc_lat_abs_at_peak: float | None = None
        self._cyc_lat_dyn_at_peak: float | None = None
        self._cyc_lat_dyn_max: float | None = None
        self._cyc_lat_dyn_min: float | None = None
        self._cyc_lat_dyn_abs_max: float | None = None
        self._cyc_lat_dyn_sum = 0.0
        self._cyc_lat_dyn_n = 0
        # Amostras (t, opening, lateral_dynamic) do ciclo INTEIRO (abrindo->
        # fechando), usadas ao final para achar o plato de abertura maxima e
        # a mediana de lateral_dynamic nele (ver _finish_cycle).
        self._cyc_samples: list[tuple[float, float, float | None]] = []

        # -- Recuperacao do inicio real do ciclo (secao 1 do refinamento de
        # precisao) -----------------------------------------------------
        # Ultima amostra VALIDA vista com abertura dentro da banda "realmente
        # fechada" (closed_baseline_limit), atualizada continuamente enquanto
        # NENHUM ciclo esta em andamento e nenhum candidato de abertura esta
        # em curso. E a fonte PRINCIPAL do inicio exportado -- ao contrario
        # do pre-buffer (janela de tempo fixa), nao se perde numa abertura
        # lenta que demore mais que prebuffer_seconds ate confirmar.
        self._last_stable_closed_sample: tuple[float, float, float | None, float | None] | None = None
        # Copia CONGELADA de `_last_stable_closed_sample` no instante em que
        # o sinal sai da banda fechada (inicio candidato). Fica intocada
        # durante toda a abertura candidata; usada como inicio do ciclo se
        # confirmar, ou descartada (junto com `_candidate_samples`) se o
        # sinal cair de volta ao fechado sem confirmar (ruido).
        self._candidate_anchor: tuple[float, float, float | None, float | None] | None = None
        # Amostras acumuladas DURANTE o candidato (desde que saiu da banda
        # fechada ate a confirmacao), para reconstituir pico/lateral daquele
        # trecho quando o ciclo e confirmado.
        self._candidate_samples: list[tuple[float, float, float | None, float | None]] = []

        # PRE-BUFFER: janela deslizante de tempo (config.prebuffer_seconds),
        # mantida so como recurso AUXILIAR (nao a fonte principal do inicio)
        # para o caso em que nao ha `_last_stable_closed_sample` valido (ex.:
        # a sessao comecou com o rosto ja em movimento, sem nenhuma amostra
        # fechada observada ainda). Ver _begin_cycle().
        self._prebuffer: list[tuple[float, float, float | None, float | None]] = []
        # Fim CANDIDATO (dentro de um ciclo confirmado): sequencia de frames
        # consecutivos DENTRO de closed_baseline_limit (banda estreita, nao
        # o limiar de fechamento de 25% da histerese); confirmada apos
        # close_stability_seconds (duracao, nao contagem de frames), usando
        # o PRIMEIRO frame da sequencia como fim exportado. Ver _finish_cycle()
        # -- logica de fechamento INALTERADA nesta rodada.
        self._closing_pending: list[tuple[float, float]] = []

    # -- Calibracao -------------------------------------------------------
    def calibrate(
        self,
        closed_value: float,
        open_value: float,
        lateral_baseline: float | None = None,
        lateral_baseline_std: float | None = None,
    ) -> None:
        """
        Define a faixa de abertura com base em amostras de boca fechada e
        aberta, e opcionalmente o baseline neutro de lateralidade (mediana)
        e seu desvio-padrao (mesma fase de boca fechada da calibracao) --
        usado para adaptar `effective_direction_deadzone` ao ruido real.
        """
        self._baseline = closed_value
        self._span = max(open_value - closed_value, 1e-6)
        if lateral_baseline is not None:
            self._lateral_baseline = lateral_baseline
        if lateral_baseline_std is not None:
            self._lateral_baseline_std = lateral_baseline_std

    def clear_calibration(self) -> None:
        """Remove a calibracao atual (opening, baseline lateral e seu ruido; tecla X)."""
        self._baseline = None
        self._span = None
        self._lateral_baseline = None
        self._lateral_baseline_std = None

    def reset_session(self) -> None:
        """
        Reinicia o estado, os ciclos e a faixa dinamica para uma NOVA sessao,
        preservando a calibracao (baseline/span/lateral_baseline/std) ja
        existente. Usado ao iniciar uma gravacao (R) e ao zerar a sessao
        atual (Z), para que repeticoes nunca sejam herdadas de fora do
        intervalo gravado.
        """
        self.state = MovementState.FECHADO
        self.cycles = []
        self._dyn_min = float("inf")
        self._dyn_max = float("-inf")
        self._cycle_start_t = None
        self._cycle_peak = 0.0
        self._cycle_peak_time = 0.0
        self._cyc_lat_dyn_sum = 0.0
        self._cyc_lat_dyn_n = 0
        self._cyc_samples = []
        self._last_stable_closed_sample = None
        self._candidate_anchor = None
        self._candidate_samples = []
        self._prebuffer = []
        self._closing_pending = []

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

    @property
    def lateral_baseline(self) -> float | None:
        """Lateral_absolute neutro (mediana na fase de boca fechada da calibracao)."""
        return self._lateral_baseline

    @property
    def lateral_baseline_std(self) -> float | None:
        """Desvio-padrao da lateral_absolute na fase de boca fechada da calibracao."""
        return self._lateral_baseline_std

    @property
    def effective_direction_deadzone(self) -> float:
        """
        Limiar efetivo (adaptativo) para classificar direita/esquerda/centro:
        max(direction_deadzone_min, direction_noise_multiplier *
        lateral_baseline_std). Sem baseline_std conhecido (nao calibrado ou
        sem amostras laterais), cai para o piso `direction_deadzone_min`.
        """
        std = self._lateral_baseline_std or 0.0
        return max(self.config.direction_deadzone_min, self.config.direction_noise_multiplier * std)

    def _thresholds(self, opening: float) -> tuple[float, float, float, float]:
        """Retorna (limiar_abrir, limiar_fechar, base, faixa) em unidades de abertura."""
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
        return open_th, close_th, base, span

    def _closed_baseline_limit(self, base: float, span: float) -> float:
        """
        Banda de "boca realmente fechada" (secao 4 do refinamento): mais
        estreita que o limiar de fechamento da histerese (25%), usada para
        validar os limites EXPORTADOS do ciclo (inicio/fim), nao a maquina
        de estados em si.
        """
        return base + self.config.boundary_closed_fraction * span

    def _accumulate_lateral(self, lateral_dynamic: float | None) -> None:
        if lateral_dynamic is None:
            return
        self._cyc_lat_dyn_max = (
            lateral_dynamic if self._cyc_lat_dyn_max is None
            else max(self._cyc_lat_dyn_max, lateral_dynamic)
        )
        self._cyc_lat_dyn_min = (
            lateral_dynamic if self._cyc_lat_dyn_min is None
            else min(self._cyc_lat_dyn_min, lateral_dynamic)
        )
        abs_v = abs(lateral_dynamic)
        self._cyc_lat_dyn_abs_max = (
            abs_v if self._cyc_lat_dyn_abs_max is None else max(self._cyc_lat_dyn_abs_max, abs_v)
        )
        self._cyc_lat_dyn_sum += lateral_dynamic
        self._cyc_lat_dyn_n += 1

    def _begin_cycle(self, base: float, span: float) -> None:
        """
        Confirma o inicio do ciclo usando `_candidate_anchor` -- a copia
        CONGELADA de `_last_stable_closed_sample` no instante em que o sinal
        saiu da banda fechada (nao depende de uma janela de tempo fixa: uma
        abertura lenta, levando mais que prebuffer_seconds ate confirmar, nao
        perde a posicao realmente fechada). Fallback documentado, nesta ordem,
        quando nao ha ancora (nenhuma amostra fechada observada desde o
        ultimo ciclo/reset -- ex.: sessao comecou ja em movimento):
          1. amostra mais antiga do pre-buffer (recurso auxiliar);
          2. o proprio primeiro frame candidato (`_candidate_samples[0]`).
        Nunca inventa um valor: quando cai em fallback, o inicio exportado
        normalmente NAO estara dentro da banda fechada
        (`start_within_baseline=False`) e o motivo fica registrado.
        """
        closed_limit = self._closed_baseline_limit(base, span)
        self._cycle_closed_limit = closed_limit

        replay = list(self._candidate_samples)  # sempre nao-vazio neste ponto (inclui o frame de confirmacao)

        if self._candidate_anchor is not None:
            anchor = self._candidate_anchor
            self._cycle_start_origin = "last_stable_closed_sample"
            self._cycle_start_fallback_reason = None
            full_replay = [anchor] + replay
        elif self._prebuffer:
            anchor = self._prebuffer[0]
            self._cycle_start_origin = "fallback"
            self._cycle_start_fallback_reason = (
                "sem last_stable_closed_sample (nenhuma amostra fechada observada "
                "antes deste movimento); usada a amostra mais antiga do pre-buffer."
            )
            full_replay = [anchor] + replay
        else:
            anchor = replay[0]
            self._cycle_start_origin = "fallback"
            self._cycle_start_fallback_reason = (
                "sem last_stable_closed_sample nem pre-buffer disponivel; usado o "
                "primeiro frame candidato (provavel inicio de sessao ja em movimento)."
            )
            full_replay = replay

        self._cycle_anchor_confirm_gap_s = replay[-1][0] - anchor[0]

        first_t, first_opening, first_abs, first_dyn = anchor
        self._cycle_start_t = first_t
        self._cycle_start_opening = first_opening
        self._cycle_baseline_opening = base
        self._cycle_peak = first_opening
        self._cycle_peak_time = first_t
        self._cyc_lat_abs_at_peak = first_abs
        self._cyc_lat_dyn_at_peak = first_dyn
        self._cyc_lat_dyn_max = None
        self._cyc_lat_dyn_min = None
        self._cyc_lat_dyn_abs_max = None
        self._cyc_lat_dyn_sum = 0.0
        self._cyc_lat_dyn_n = 0
        self._cyc_samples = []
        for pt, popening, pabs, pdyn in full_replay:
            if popening > self._cycle_peak:
                self._cycle_peak = popening
                self._cycle_peak_time = pt
                self._cyc_lat_abs_at_peak = pabs
                self._cyc_lat_dyn_at_peak = pdyn
            self._accumulate_lateral(pdyn)
            self._cyc_samples.append((pt, popening, pdyn))

        self._candidate_anchor = None
        self._candidate_samples = []
        self._prebuffer = []
        self._closing_pending = []

    def _plateau_median(self) -> tuple[float | None, int, bool]:
        """
        Mediana de lateral_dynamic no plato de abertura maxima do ciclo
        (amostras com abertura >= direction_plateau_fraction*pico). Se o
        plato tiver poucas amostras (< direction_min_plateau_samples), cai
        no fallback de uma janela temporal ao redor de peak_time. Retorna
        (mediana, numero_de_amostras_usadas, usou_fallback).
        """
        if not self._cyc_samples or self._cycle_peak <= 1e-9:
            return None, 0, False

        threshold = self.config.direction_plateau_fraction * self._cycle_peak
        plateau_vals = [
            dyn for (_pt, popening, dyn) in self._cyc_samples
            if popening >= threshold and dyn is not None
        ]
        if len(plateau_vals) >= self.config.direction_min_plateau_samples:
            return float(np.median(plateau_vals)), len(plateau_vals), False

        window = self.config.direction_plateau_fallback_window_s
        window_vals = [
            dyn for (pt, _popening, dyn) in self._cyc_samples
            if abs(pt - self._cycle_peak_time) <= window and dyn is not None
        ]
        if window_vals:
            return float(np.median(window_vals)), len(window_vals), True

        # Ultimo recurso: os proprios valores do plato (mesmo poucos), se houver.
        if plateau_vals:
            return float(np.median(plateau_vals)), len(plateau_vals), True
        return None, 0, True

    def _finish_cycle(self, closed_limit: float) -> Cycle | None:
        """
        Confirma o fim do ciclo usando o 1o frame da corrida estavel (banda
        fechada) -- logica de fechamento INALTERADA nesta rodada. Ao
        finalizar, tambem calcula a mediana de lateral_dynamic no plato de
        abertura maxima (ver `_plateau_median`), usada pela classificacao de
        direcao (secao 3 do refinamento de precisao).
        """
        end_t, end_opening = self._closing_pending[0]
        cycle = None
        if self._cycle_start_t is not None:
            duration = end_t - self._cycle_start_t
            if duration >= self.config.min_cycle_seconds:
                lat_mean = (
                    self._cyc_lat_dyn_sum / self._cyc_lat_dyn_n
                    if self._cyc_lat_dyn_n else None
                )
                median_plateau, plateau_n, plateau_fallback = self._plateau_median()
                cycle = Cycle(
                    cycle_id=len(self.cycles) + 1,
                    start_time=self._cycle_start_t,
                    end_time=end_t,
                    peak_time=self._cycle_peak_time,
                    peak_opening=self._cycle_peak,
                    baseline_opening=self._cycle_baseline_opening,
                    start_opening=self._cycle_start_opening,
                    end_opening=end_opening,
                    closed_baseline_limit=self._cycle_closed_limit,
                    start_within_baseline=self._cycle_start_opening <= self._cycle_closed_limit,
                    end_within_baseline=end_opening <= closed_limit,
                    lateral_absolute_at_peak=self._cyc_lat_abs_at_peak,
                    lateral_dynamic_at_peak=self._cyc_lat_dyn_at_peak,
                    lateral_dynamic_max=self._cyc_lat_dyn_max,
                    lateral_dynamic_min=self._cyc_lat_dyn_min,
                    lateral_dynamic_abs_max=self._cyc_lat_dyn_abs_max,
                    lateral_dynamic_mean=lat_mean,
                    start_origin=self._cycle_start_origin,
                    start_fallback_reason=self._cycle_start_fallback_reason,
                    anchor_confirm_gap_s=self._cycle_anchor_confirm_gap_s,
                    lateral_dynamic_median_plateau=median_plateau,
                    plateau_sample_count=plateau_n,
                    plateau_used_fallback=plateau_fallback,
                )
                self.cycles.append(cycle)
        self._cycle_start_t = None
        self._cycle_peak = 0.0
        self._cyc_samples = []
        self._closing_pending = []
        return cycle

    # -- Atualizacao por frame -------------------------------------------
    def update(
        self,
        opening: float,
        t: float,
        lateral_absolute: float | None = None,
        lateral_dynamic: float | None = None,
    ) -> bool:
        """
        Atualiza a maquina de estados com a abertura atual no instante t.
        `lateral_absolute`/`lateral_dynamic` (opcionais) sao acumulados
        durante o ciclo em andamento para as metricas por ciclo (secao 5).
        Retorna True se um ciclo foi concluido neste frame.

        Delimitacao completa do ciclo (boca fechada -> abertura -> pico ->
        fechamento -> boca fechada), preservando a histerese da maquina de
        estados (open_fraction=60% / close_fraction=25%) so para decidir
        QUANDO confirmar abertura/fechamento (evitar ruido), mas usando uma
        banda mais estreita (`closed_baseline_limit`, boundary_closed_fraction
        =5% por padrao) para decidir OS LIMITES EXPORTADOS do ciclo:
          - INICIO: `last_stable_closed_sample` e atualizada em TODO frame
            com abertura dentro da banda fechada, enquanto nenhum ciclo/
            candidato esta em andamento. Assim que o sinal sai da banda, essa
            amostra e CONGELADA em `_candidate_anchor` (nao e mais
            sobrescrita durante a abertura candidata, por mais longa que
            seja -- ao contrario de uma janela de tempo fixa, nao se perde
            numa abertura lenta). Se o limiar de abertura for atingido, o
            ciclo comeca no anchor; se o sinal cair de volta ao fechado sem
            confirmar, o anchor e descartado (ruido) e `last_stable_closed_
            sample` volta a ser atualizado normalmente. O pre-buffer
            (janela de tempo) so entra como fallback auxiliar quando nao ha
            anchor (ver _begin_cycle).
          - FIM: exige `close_stability_seconds` (duracao, nao numero de
            frames -- equivalente em 10/15/30fps) de permanencia CONTINUA
            dentro da banda fechada antes de confirmar; o fim exportado e o
            primeiro frame dessa sequencia estavel. INALTERADO nesta rodada.
        A ramificacao usa `self._cycle_start_t is not None` (nao `self.state`)
        para saber se ha um ciclo em andamento -- `self.state` e so para
        exibicao (fica FECHANDO assim que o sinal cruza o limiar de 25%,
        mesmo antes da confirmacao pela banda estreita, para o biofeedback
        continuar responsivo).
        """
        open_th, close_th, base, span = self._thresholds(opening)
        completed = False

        if self._cycle_start_t is None:
            # Nenhum ciclo em andamento: rastreia a ultima amostra fechada
            # (fonte principal do inicio) e mantem o pre-buffer (auxiliar).
            self.state = MovementState.FECHADO
            closed_limit = self._closed_baseline_limit(base, span)
            sample = (t, opening, lateral_absolute, lateral_dynamic)

            if opening <= closed_limit:
                self._last_stable_closed_sample = sample
                self._candidate_anchor = None
                self._candidate_samples = []
            else:
                if self._candidate_anchor is None:
                    # 1o frame fora da banda fechada: congela a ancora (pode
                    # ser None, se nunca vimos uma amostra fechada ainda).
                    self._candidate_anchor = self._last_stable_closed_sample
                self._candidate_samples.append(sample)

            self._prebuffer.append(sample)
            while self._prebuffer and (t - self._prebuffer[0][0]) > self.config.prebuffer_seconds:
                self._prebuffer.pop(0)

            if opening >= open_th:
                self._begin_cycle(base, span)
                self.state = MovementState.ABRINDO
        else:
            # Ciclo em andamento (abrindo / aberto / fechando-candidato).
            self._cyc_samples.append((t, opening, lateral_dynamic))
            if opening > self._cycle_peak:
                self._cycle_peak = opening
                self._cycle_peak_time = t
                self._cyc_lat_abs_at_peak = lateral_absolute
                self._cyc_lat_dyn_at_peak = lateral_dynamic
            self._accumulate_lateral(lateral_dynamic)

            closed_limit = self._closed_baseline_limit(base, span)
            if opening <= closed_limit:
                self._closing_pending.append((t, opening))
            else:
                self._closing_pending = []  # saiu da banda fechada: reinicia a estabilidade

            if opening <= close_th:
                self.state = MovementState.FECHANDO
            else:
                self.state = MovementState.ABERTO if opening >= open_th else MovementState.ABRINDO

            if self._closing_pending:
                run_span = self._closing_pending[-1][0] - self._closing_pending[0][0]
                if run_span >= self.config.close_stability_seconds:
                    cycle = self._finish_cycle(closed_limit)
                    self.state = MovementState.FECHADO
                    completed = cycle is not None

        return completed

    # -- Estatisticas -----------------------------------------------------
    @property
    def repetitions(self) -> int:
        return len(self.cycles)

    def repeatability(self) -> dict[str, float]:
        """
        Metricas de repetibilidade entre os ciclos detectados.

        Retorna medias, desvios-padrao e coeficiente de variacao (CV) da
        amplitude (pico - referencia "fechado" do ciclo, nao o pico bruto) e
        da duracao, alem de estatisticas do desvio lateral dinamico entre
        ciclos. CV baixo indica movimento mais repetivel.
        """
        if not self.cycles:
            return {}

        amplitudes = np.array([c.amplitude for c in self.cycles], dtype=float)
        durations = np.array([c.duration for c in self.cycles], dtype=float)

        def cv(arr: np.ndarray) -> float:
            m = float(np.mean(arr))
            return float(np.std(arr) / m) if m > 1e-9 else 0.0

        result = {
            "n_ciclos": float(len(self.cycles)),
            "amplitude_media": float(np.mean(amplitudes)),
            "amplitude_dp": float(np.std(amplitudes)),
            "amplitude_cv": cv(amplitudes),
            "duracao_media_s": float(np.mean(durations)),
            "duracao_dp_s": float(np.std(durations)),
            "duracao_cv": cv(durations),
        }

        abs_max_vals = [c.lateral_dynamic_abs_max for c in self.cycles
                         if c.lateral_dynamic_abs_max is not None]
        max_pos_vals = [c.lateral_dynamic_max for c in self.cycles
                         if c.lateral_dynamic_max is not None]
        min_neg_vals = [c.lateral_dynamic_min for c in self.cycles
                         if c.lateral_dynamic_min is not None]
        if abs_max_vals:
            result["lateral_dinamico_absmax_media"] = float(np.mean(abs_max_vals))
        if max_pos_vals:
            result["lateral_dinamico_max_positivo"] = float(np.max(max_pos_vals))
        if min_neg_vals:
            result["lateral_dinamico_min_negativo"] = float(np.min(min_neg_vals))

        return result


# ===========================================================================
# Analise facial frontal (simetria, angulos e proporcoes)
# Funcoes puras que recebem FaceLandmarks e nao dependem do recorder/app.
#
# O sistema trabalha APENAS com a vista frontal. A analise de perfil (plano
# sagital) foi removida: em 2D, sem controle de distancia nem de rotacao da
# cabeca, as projecoes e angulos sagitais nao se mostraram confiaveis.
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


