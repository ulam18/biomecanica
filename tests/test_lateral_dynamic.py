"""
Testes do refinamento da analise lateral frontal: lateral_absolute vs
lateral_dynamic, baseline neutro, convencao direita/esquerda, calibracao
lateral, reset preservando baseline, e rejeicao de multiplas faces.

Sinteticos (landmarks fabricados), sem webcam/MediaPipe real.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mandibular.calibration import CalibrationAssistant  # noqa: E402
from mandibular.config import Landmark, QualityConfig  # noqa: E402
from mandibular.filters import EMAFilter  # noqa: E402
from mandibular.landmarks import FaceLandmarks  # noqa: E402
from mandibular.metrics import (  # noqa: E402
    CycleDetector,
    compute_frame_metrics,
    cycle_predominant_direction,
    lateral_direction,
)
from mandibular.pipeline import process_frame  # noqa: E402
from mandibular.quality import FrameQuality, assess_quality  # noqa: E402

N_LANDMARKS = 478


def make_face(lateral_px: float = 0.0, opening_px: float = 30.0,
              center=(300.0, 300.0)) -> FaceLandmarks:
    """
    Face sintetica com um deslocamento CONTROLADO do queixo. `lateral_px` >0
    move o queixo para o lado da imagem onde fica o EYE_OUTER_RIGHT (x maior).
    """
    pts = np.zeros((N_LANDMARKS, 2), dtype=np.float32)
    pts[Landmark.EYE_OUTER_LEFT] = (200.0, 200.0)
    pts[Landmark.EYE_OUTER_RIGHT] = (400.0, 200.0)
    pts[Landmark.NASION] = (300.0, 210.0)
    pts[Landmark.CHIN] = (300.0 + lateral_px, 450.0)
    pts[Landmark.UPPER_LIP_INNER] = (300.0, 350.0)
    pts[Landmark.LOWER_LIP_INNER] = (300.0, 350.0 + opening_px)
    pts[Landmark.NOSE_TIP] = (300.0, 300.0)
    pts[Landmark.MOUTH_LEFT] = (270.0, 350.0)
    pts[Landmark.MOUTH_RIGHT] = (330.0, 350.0)
    return FaceLandmarks(points=pts, image_width=600, image_height=600)


# --------------------------------------------------------------------------
# Baseline neutro (calibracao) e lateral_dynamic
# --------------------------------------------------------------------------
def test_calibration_computes_lateral_neutral_baseline():
    """Fase de boca fechada com um desvio lateral CONSTANTE -> baseline = esse valor."""
    ca = CalibrationAssistant()
    ca.start()
    t = ca.phase_start
    result = None
    while ca.active and result is None:
        lateral = 0.08 if ca.phase == 0 else 0.08  # constante nas 2 fases (irrelevante na fase 1)
        opening = 0.05 if ca.phase == 0 else 0.55
        result = ca.update(opening, t, lateral=lateral)
        t += 1.0 / 40
    assert result is not None and result.valid
    assert abs(result.lateral_baseline - 0.08) < 1e-6
    assert result.lateral_baseline_std < 1e-6


def test_calibration_rejects_unstable_lateral_baseline():
    """Lateralidade oscilando muito na fase 'fechada' invalida a calibracao inteira."""
    ca = CalibrationAssistant()
    ca.start()
    t = ca.phase_start
    result = None
    toggle = True
    while ca.active and result is None:
        if ca.phase == 0:
            lateral = 0.20 if toggle else -0.20  # oscila muito -> instavel
            toggle = not toggle
            opening = 0.05
        else:
            lateral = 0.0
            opening = 0.55
        result = ca.update(opening, t, lateral=lateral)
        t += 1.0 / 40
    assert result is not None
    assert not result.valid
    assert "instavel" in result.message


def test_initial_absolute_nonzero_dynamic_near_zero_at_neutral():
    """
    Posicao inicial (neutra) com lateral_absolute != 0 (assimetria estatica
    da face): apos calibrar o baseline nessa mesma posicao, lateral_dynamic
    deve ficar ~0, mesmo com lateral_absolute != 0.
    """
    quality_cfg = QualityConfig()
    filt_o, filt_l, filt_w = EMAFilter(0.9), EMAFilter(0.9), EMAFilter(0.3)

    # Face com assimetria estatica: queixo levemente deslocado mesmo "neutro".
    neutral_face = make_face(lateral_px=16.0, opening_px=10.0)  # ~0.08 rel (queixo/200px)
    m = compute_frame_metrics(neutral_face)
    assert abs(m.lateral_rel) > 0.05  # lateral_absolute claramente != 0

    cycles = CycleDetector()
    cycles.calibrate(0.05, 0.55, lateral_baseline=m.lateral_rel)

    fr = process_frame(
        neutral_face, None, True, quality_cfg, filt_o, filt_l, filt_w,
        lateral_baseline=cycles.lateral_baseline,
    )
    assert fr.frame_valid
    assert abs(fr.lateral_filtered) > 0.05          # absolute preservado (nao apagado)
    assert abs(fr.lateral_dynamic_filtered) < 1e-3   # dynamic ~0 na posicao neutra


# --------------------------------------------------------------------------
# Direita/esquerda anatomica: nao confundir lado da tela com lado anatomico
# --------------------------------------------------------------------------
def test_anatomical_right_normal_image():
    m = compute_frame_metrics(make_face(lateral_px=40.0))
    assert lateral_direction(m.lateral_rel, mirrored=True) == "direita"


def test_anatomical_left_normal_image():
    m = compute_frame_metrics(make_face(lateral_px=-40.0))
    assert lateral_direction(m.lateral_rel, mirrored=True) == "esquerda"


def test_anatomical_right_mirrored_image():
    """A MESMA geometria (queixo do lado do EYE_OUTER_RIGHT), mas rotulada
    como imagem espelhada (mirrored=False): o rotulo deve inverter."""
    m = compute_frame_metrics(make_face(lateral_px=40.0))
    assert lateral_direction(m.lateral_rel, mirrored=False) == "esquerda"


def test_anatomical_left_mirrored_image():
    m = compute_frame_metrics(make_face(lateral_px=-40.0))
    assert lateral_direction(m.lateral_rel, mirrored=False) == "direita"


# --------------------------------------------------------------------------
# Calibracao lateral: reset (Z) preserva, apagar calibracao (X) remove
# --------------------------------------------------------------------------
def test_reset_session_preserves_lateral_baseline():
    cycles = CycleDetector()
    cycles.calibrate(0.05, 0.55, lateral_baseline=0.09)
    cycles.reset_session()
    assert cycles.lateral_baseline == 0.09
    assert cycles.is_calibrated


def test_clear_calibration_removes_lateral_baseline():
    cycles = CycleDetector()
    cycles.calibrate(0.05, 0.55, lateral_baseline=0.09)
    cycles.clear_calibration()
    assert cycles.lateral_baseline is None
    assert not cycles.is_calibrated


# --------------------------------------------------------------------------
# Metricas por ciclo
# --------------------------------------------------------------------------
def test_cycle_metrics_captures_lateral_at_peak_and_extremes():
    det = CycleDetector()
    det.calibrate(0.0, 1.0)
    # abre (lateral vai para +0.3), pico com lateral +0.5, fecha com lateral -0.1
    # (fechamento com abertura <= closed_baseline_limit=0.05; timestamps
    # espacados o suficiente para passar de close_stability_seconds=0.15 e
    # min_cycle_seconds=0.25)
    det.update(0.0, 0.0, lateral_absolute=0.0, lateral_dynamic=0.0)
    det.update(0.7, 0.1, lateral_absolute=0.3, lateral_dynamic=0.3)   # cruza limiar (0.6)
    det.update(0.9, 0.3, lateral_absolute=0.5, lateral_dynamic=0.5)   # novo pico de abertura
    det.update(0.02, 0.6, lateral_absolute=-0.1, lateral_dynamic=-0.1)  # 1o frame na banda fechada
    assert det.repetitions == 0  # fechamento ainda nao confirmado (janela de estabilidade)
    det.update(0.02, 0.9, lateral_absolute=-0.1, lateral_dynamic=-0.1)  # confirma (0.3s >= 0.15s)

    assert det.repetitions == 1
    c = det.cycles[0]
    assert c.cycle_id == 1
    assert abs(c.peak_opening - 0.9) < 1e-9
    assert abs(c.lateral_dynamic_at_peak - 0.5) < 1e-9
    assert abs(c.lateral_dynamic_max - 0.5) < 1e-9
    assert abs(c.lateral_dynamic_min - (-0.1)) < 1e-9
    assert abs(c.lateral_dynamic_abs_max - 0.5) < 1e-9
    assert c.duration > 0
    assert c.opening_velocity > 0
    assert c.closing_velocity > 0


def test_cycle_predominant_direction_uses_shared_convention():
    det = CycleDetector()
    det.calibrate(0.0, 1.0)
    det.update(0.7, 0.0, lateral_dynamic=0.4)
    det.update(0.9, 0.1, lateral_dynamic=0.4)
    det.update(0.02, 0.3, lateral_dynamic=0.4)   # 1o frame na banda fechada (candidato)
    det.update(0.02, 0.5, lateral_dynamic=0.4)   # confirma o fechamento (0.2s >= 0.15s)
    assert det.repetitions == 1
    c = det.cycles[0]
    direction, value = cycle_predominant_direction(c, mirrored=True)
    assert direction == "direita"
    assert abs(value - c.lateral_dynamic_at_peak) < 1e-9  # criterio: valor no pico
    direction, _ = cycle_predominant_direction(c, mirrored=False)
    assert direction == "esquerda"


def test_incomplete_cycle_is_discarded():
    """Um ciclo que abre mas nunca fecha (sessao termina no meio) nao conta."""
    det = CycleDetector(config=None)
    det.calibrate(0.0, 1.0)
    det.update(0.0, 0.0)
    det.update(0.7, 0.1)   # abre
    det.update(0.9, 0.2)   # pico, ainda aberto -- sessao "acaba" aqui
    assert det.repetitions == 0
    assert det.cycles == []


# --------------------------------------------------------------------------
# Multiplas faces
# --------------------------------------------------------------------------
def test_multiple_faces_invalidates_frame():
    face = make_face()
    face.num_faces = 2
    result = assess_quality(face, None, QualityConfig())
    assert result.quality == FrameQuality.INVALIDA
    assert result.reason == "multiplas_faces"
    assert result.message == "Mantenha apenas uma pessoa no enquadramento"


def test_single_face_stays_valid():
    face = make_face()
    assert face.num_faces == 1
    result = assess_quality(face, None, QualityConfig())
    assert result.quality == FrameQuality.VALIDA


def test_multiple_faces_can_be_disabled_via_config():
    face = make_face()
    face.num_faces = 2
    cfg = QualityConfig(reject_multiple_faces=False)
    result = assess_quality(face, None, cfg)
    assert result.quality == FrameQuality.VALIDA


# --------------------------------------------------------------------------
# Deadzone efetiva da direcao (secao 5 do refinamento de precisao): fixa
# (0.02) era sensivel demais a ruido real; agora e adaptativa ao ruido
# medido na fase de boca fechada da calibracao, com piso configuravel.
# --------------------------------------------------------------------------
def test_effective_deadzone_uses_floor_when_calibration_is_clean():
    """Sem ruido conhecido (ou ruido pequeno), o piso direction_deadzone_min domina."""
    cycles = CycleDetector()
    cycles.calibrate(0.0, 1.0, lateral_baseline=0.0)  # sem std informado
    assert cycles.lateral_baseline_std is None
    assert cycles.effective_direction_deadzone == cycles.config.direction_deadzone_min
    assert cycles.config.direction_deadzone_min == 0.03


def test_effective_deadzone_adapts_to_calibration_noise():
    """Ruido de calibracao maior que o piso -> deadzone efetiva cresce (3x o desvio-padrao)."""
    cycles = CycleDetector()
    cycles.calibrate(0.0, 1.0, lateral_baseline=0.0, lateral_baseline_std=0.02)
    expected = max(cycles.config.direction_deadzone_min, 3.0 * 0.02)
    assert abs(cycles.effective_direction_deadzone - expected) < 1e-9
    assert cycles.effective_direction_deadzone > cycles.config.direction_deadzone_min


def test_clear_calibration_resets_lateral_baseline_std():
    cycles = CycleDetector()
    cycles.calibrate(0.0, 1.0, lateral_baseline=0.05, lateral_baseline_std=0.02)
    cycles.clear_calibration()
    assert cycles.lateral_baseline_std is None
    assert cycles.effective_direction_deadzone == cycles.config.direction_deadzone_min


def test_reset_session_preserves_lateral_baseline_std():
    cycles = CycleDetector()
    cycles.calibrate(0.0, 1.0, lateral_baseline=0.05, lateral_baseline_std=0.02)
    cycles.reset_session()
    assert cycles.lateral_baseline_std == 0.02


def test_real_session_reference_classification_with_effective_deadzone():
    """
    Reproduz a classificacao esperada da coleta real (secao 5): com
    lateral_baseline_std pequeno o suficiente para o piso (0.03) dominar,
    +0.0128/+0.0203/+0.0263 -> centro; +0.0797 -> direita; -0.0626 -> esquerda.
    """
    from mandibular.metrics import lateral_direction

    cycles = CycleDetector()
    cycles.calibrate(0.0, 1.0, lateral_baseline=0.0, lateral_baseline_std=0.005)
    deadzone = cycles.effective_direction_deadzone
    assert abs(deadzone - 0.03) < 1e-9  # piso domina (3*0.005=0.015 < 0.03)

    cases = [
        (0.0128, "centro"),
        (0.0203, "centro"),
        (0.0263, "centro"),
        (0.0797, "direita"),
        (-0.0626, "esquerda"),
    ]
    for value, expected in cases:
        assert lateral_direction(value, mirrored=True, deadzone=deadzone) == expected, value


def test_026_classified_as_center_with_effective_deadzone_003():
    from mandibular.metrics import lateral_direction
    assert lateral_direction(0.026, mirrored=True, deadzone=0.03) == "centro"


def test_080_classified_as_direita():
    from mandibular.metrics import lateral_direction
    assert lateral_direction(0.080, mirrored=True, deadzone=0.03) == "direita"


def test_063_negative_classified_as_esquerda():
    from mandibular.metrics import lateral_direction
    assert lateral_direction(-0.063, mirrored=True, deadzone=0.03) == "esquerda"
