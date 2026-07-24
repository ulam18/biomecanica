"""
Controle de qualidade do frame.

Classifica cada frame em nao detectado / invalido / valido, e produz uma
orientacao curta para o usuario corrigir o posicionamento. Isso evita que
ruido de deteccao (rosto pequeno demais, cabeca inclinada, movimento brusco)
contamine as metricas e a contagem de ciclos.

O tamanho do rosto e avaliado pela razao face_width_px / frame_width_px (nao
pela diagonal da imagem): essa razao e a mesma em qualquer resolucao para a
mesma distancia/posicao real do rosto perante a camera (ver QualityConfig).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import Landmark, QualityConfig
from .landmarks import FaceLandmarks


class FrameQuality(str, Enum):
    NAO_DETECTADA = "face_nao_detectada"
    INVALIDA = "invalida"
    VALIDA = "valida"


@dataclass
class QualityResult:
    quality: FrameQuality
    message: str | None  # orientacao amigavel para o usuario; None quando o frame e valido
    reason: str | None = None  # motivo curto e estavel (para logs/depuracao/testes)
    ratio: float | None = None  # face_width_px / frame_width_px (None se nao calculavel)
    min_ratio: float = 0.0
    max_ratio: float = 0.0


def assess_quality(
    face: FaceLandmarks | None,
    prev_nasion: np.ndarray | None,
    config: QualityConfig | None = None,
) -> QualityResult:
    """Avalia a qualidade do frame atual a partir dos landmarks detectados."""
    config = config or QualityConfig()
    min_r, max_r = config.min_face_width_ratio, config.max_face_width_ratio

    if face is None:
        return QualityResult(
            FrameQuality.NAO_DETECTADA, "Face nao detectada",
            reason="sem_face", ratio=None, min_ratio=min_r, max_ratio=max_r,
        )

    h, w = face.image_height, face.image_width
    eye_l = face.point(Landmark.EYE_OUTER_LEFT)
    eye_r = face.point(Landmark.EYE_OUTER_RIGHT)
    nasion = face.point(Landmark.NASION)
    chin = face.point(Landmark.CHIN)

    pts = np.array([eye_l, eye_r, nasion, chin])
    if (
        np.any(pts[:, 0] < 0)
        or np.any(pts[:, 0] > w)
        or np.any(pts[:, 1] < 0)
        or np.any(pts[:, 1] > h)
    ):
        return QualityResult(
            FrameQuality.INVALIDA, "Centralize o rosto",
            reason="fora_da_imagem", ratio=None, min_ratio=min_r, max_ratio=max_r,
        )

    face_width = float(np.linalg.norm(eye_r - eye_l))
    ratio = face_width / max(float(w), 1e-6)

    if ratio < min_r:
        return QualityResult(
            FrameQuality.INVALIDA,
            f"Aproxime-se da camera (razao {ratio:.3f} < min {min_r:.3f})",
            reason="muito_longe", ratio=ratio, min_ratio=min_r, max_ratio=max_r,
        )
    if ratio > max_r:
        return QualityResult(
            FrameQuality.INVALIDA,
            f"Afaste-se da camera (razao {ratio:.3f} > max {max_r:.3f})",
            reason="muito_perto", ratio=ratio, min_ratio=min_r, max_ratio=max_r,
        )

    dx, dy = eye_r - eye_l
    roll_deg = float(np.degrees(np.arctan2(dy, dx)))
    if abs(roll_deg) > config.max_roll_deg:
        return QualityResult(
            FrameQuality.INVALIDA, "Mantenha a cabeca estavel (nao incline)",
            reason="inclinacao_excessiva", ratio=ratio, min_ratio=min_r, max_ratio=max_r,
        )

    if prev_nasion is not None:
        jump = float(np.linalg.norm(nasion - prev_nasion)) / max(face_width, 1e-6)
        if jump > config.max_global_jump_fraction:
            return QualityResult(
                FrameQuality.INVALIDA, "Mantenha a cabeca estavel",
                reason="movimento_brusco", ratio=ratio, min_ratio=min_r, max_ratio=max_r,
            )

    return QualityResult(
        FrameQuality.VALIDA, None,
        reason=None, ratio=ratio, min_ratio=min_r, max_ratio=max_r,
    )
