"""
Decisao por frame compartilhada entre o modo ao vivo (app.py) e a analise
offline (analyze_video.py).

Centralizar esta logica evita que os dois modos divirjam silenciosamente
(por exemplo, um deles gravando uma "medicao" filtrada zero para um frame
invalido enquanto o outro nao) - foi exatamente esse tipo de divergencia que
causou a sessao com frame_valido=0 e abertura_filtrada=0 em todos os frames.

Regra central: um frame invalido NUNCA gera uma nova medicao filtrada. O
filtro fica congelado (mantem o ultimo valor valido apenas para exibicao) e
o valor persistido no CSV para aquele frame e None (viraria NaN/vazio no
CSV e nos graficos), nunca um zero fabricado.

Lateralidade absoluta vs dinamica: `lateral_absolute` (= metrics.lateral_rel)
e a posicao do queixo em relacao a linha media facial, inalterada. Quando ha
um baseline neutro calibrado (`lateral_baseline`, mediana da lateral_absolute
com a boca fechada), tambem se calcula `lateral_dynamic = lateral_absolute -
lateral_baseline`, que fica ~0 na posicao neutra mesmo que a face tenha uma
assimetria estatica (lateral_absolute != 0). `lateral_dynamic_filtered` e
obtido subtraindo o baseline do valor JA filtrado (nao um segundo filtro
independente): como EMA e linear, EMA(x - c) == EMA(x) - c para uma
constante c, entao o resultado e equivalente e nao precisa de outro estado
de filtro.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import QualityConfig
from .filters import EMAFilter
from .landmarks import FaceLandmarks
from .metrics import (
    FrameMetrics,
    FrontalAngles,
    SymmetryMetrics,
    compute_frame_metrics,
    compute_frontal_angles,
    compute_symmetry,
    lateral_direction,
)
from .quality import FrameQuality, QualityResult, assess_quality


@dataclass
class FrameResult:
    metrics: FrameMetrics | None       # None se nenhuma face foi detectada
    quality: QualityResult
    frame_valid: bool
    opening_filtered: float | None     # None quando o frame nao e valido (nunca 0.0 fabricado)
    lateral_filtered: float | None     # = lateral_absolute filtrado
    lateral_baseline: float | None     # baseline neutro usado (None se nao calibrado)
    lateral_dynamic_raw: float | None       # lateral_absolute_raw - baseline
    lateral_dynamic_filtered: float | None  # lateral_absolute_filtered - baseline
    opening_display: float             # ultimo valor valido, so para exibicao (bar/HUD)
    lateral_display: float             # lateral_absolute para exibicao
    lateral_dynamic_display: float     # lateral_dynamic para exibicao
    direction: str                     # "direita"/"esquerda"/"centro"; "centro" se invalido
    # Analise facial frontal do mesmo quadro. Seguem a mesma regra dos demais
    # campos: so sao calculadas em quadro valido; None caso contrario.
    symmetry: SymmetryMetrics | None = None
    angles: FrontalAngles | None = None


def process_frame(
    face: FaceLandmarks | None,
    ref_mm: float | None,
    mirrored: bool,
    quality_config: QualityConfig,
    filt_opening: EMAFilter,
    filt_lateral: EMAFilter,
    filt_face_width: EMAFilter,
    prev_nasion=None,
    lateral_baseline: float | None = None,
) -> FrameResult:
    """Processa um frame: qualidade, metricas brutas e filtragem condicional."""
    quality = assess_quality(face, prev_nasion, quality_config)
    frame_valid = quality.quality == FrameQuality.VALIDA
    m = compute_frame_metrics(face, ref_mm) if face is not None else None

    symmetry = None
    angles = None

    if frame_valid and m is not None:
        opening_filt = filt_opening.update(m.opening_rel)
        lateral_filt = filt_lateral.update(m.lateral_rel)
        filt_face_width.update(m.face_width_px)
        # Simetria e angulos frontais: mesma face, mesmo quadro valido. Ficam
        # aqui (e nao no app) para que o modo ao vivo e a analise offline
        # gravem exatamente as mesmas colunas.
        symmetry = compute_symmetry(face)
        angles = compute_frontal_angles(face)
    else:
        opening_filt = None
        lateral_filt = None

    opening_display = filt_opening.value if filt_opening.value is not None else 0.0
    lateral_display = filt_lateral.value if filt_lateral.value is not None else 0.0

    lateral_dynamic_raw = (
        m.lateral_rel - lateral_baseline
        if (m is not None and lateral_baseline is not None)
        else None
    )
    lateral_dynamic_filtered = (
        lateral_filt - lateral_baseline
        if (lateral_filt is not None and lateral_baseline is not None)
        else None
    )
    lateral_dynamic_display = (
        lateral_display - lateral_baseline if lateral_baseline is not None else lateral_display
    )

    # Direcao: prioriza o desvio DINAMICO (baseline-corrigido) quando
    # disponivel -- e o que importa para o biofeedback/analise funcional do
    # movimento; sem calibracao, cai para o absoluto (comportamento anterior).
    direction_value = (
        lateral_dynamic_filtered if lateral_dynamic_filtered is not None else lateral_filt
    )
    direction = (
        lateral_direction(direction_value, mirrored)
        if frame_valid and direction_value is not None
        else "centro"
    )

    return FrameResult(
        metrics=m,
        quality=quality,
        frame_valid=frame_valid,
        opening_filtered=opening_filt,
        lateral_filtered=lateral_filt,
        lateral_baseline=lateral_baseline,
        lateral_dynamic_raw=lateral_dynamic_raw,
        lateral_dynamic_filtered=lateral_dynamic_filtered,
        opening_display=opening_display,
        lateral_display=lateral_display,
        lateral_dynamic_display=lateral_dynamic_display,
        direction=direction,
        symmetry=symmetry,
        angles=angles,
    )
