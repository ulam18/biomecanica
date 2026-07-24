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
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import QualityConfig
from .filters import EMAFilter
from .landmarks import FaceLandmarks
from .metrics import FrameMetrics, compute_frame_metrics, lateral_direction
from .quality import FrameQuality, QualityResult, assess_quality


@dataclass
class FrameResult:
    metrics: FrameMetrics | None       # None se nenhuma face foi detectada
    quality: QualityResult
    frame_valid: bool
    opening_filtered: float | None     # None quando o frame nao e valido (nunca 0.0 fabricado)
    lateral_filtered: float | None
    opening_display: float             # ultimo valor valido, so para exibicao (bar/HUD)
    lateral_display: float
    direction: str                     # "direita"/"esquerda"/"centro"; "centro" se invalido


def process_frame(
    face: FaceLandmarks | None,
    ref_mm: float | None,
    mirrored: bool,
    quality_config: QualityConfig,
    filt_opening: EMAFilter,
    filt_lateral: EMAFilter,
    filt_face_width: EMAFilter,
    prev_nasion=None,
) -> FrameResult:
    """Processa um frame: qualidade, metricas brutas e filtragem condicional."""
    quality = assess_quality(face, prev_nasion, quality_config)
    frame_valid = quality.quality == FrameQuality.VALIDA
    m = compute_frame_metrics(face, ref_mm) if face is not None else None

    if frame_valid and m is not None:
        opening_filt = filt_opening.update(m.opening_rel)
        lateral_filt = filt_lateral.update(m.lateral_rel)
        filt_face_width.update(m.face_width_px)
    else:
        # Frame invalido: o filtro NAO e atualizado (congela), e nenhuma
        # medicao nova e produzida (None, nao 0.0).
        opening_filt = None
        lateral_filt = None

    opening_display = filt_opening.value if filt_opening.value is not None else 0.0
    lateral_display = filt_lateral.value if filt_lateral.value is not None else 0.0

    direction = (
        lateral_direction(lateral_filt, mirrored)
        if frame_valid and lateral_filt is not None
        else "centro"
    )

    return FrameResult(
        metrics=m,
        quality=quality,
        frame_valid=frame_valid,
        opening_filtered=opening_filt,
        lateral_filtered=lateral_filt,
        opening_display=opening_display,
        lateral_display=lateral_display,
        direction=direction,
    )
