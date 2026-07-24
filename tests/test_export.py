"""
Testes de exportacao/graficos com frames invalidos ou sem nenhum frame
valido: um frame invalido nunca pode virar uma "medicao" de valor zero, nem
nos graficos nem no resumo da sessao. Tambem cobre a calibracao ignorando
frames invalidos (usando o mesmo padrao de app.py/analyze_video.py).
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mandibular.calibration import CalibrationAssistant  # noqa: E402
from mandibular.config import Landmark, QualityConfig  # noqa: E402
from mandibular.exporter import export_session  # noqa: E402
from mandibular.filters import EMAFilter  # noqa: E402
from mandibular.landmarks import FaceLandmarks  # noqa: E402
from mandibular.metrics import CycleDetector, MovementState  # noqa: E402
from mandibular.pipeline import process_frame  # noqa: E402
from mandibular.plotting import _series, plot_opening_time  # noqa: E402
from mandibular.recorder import Sample, SessionRecorder  # noqa: E402

N_LANDMARKS = 478


def _sample(i: int, frame_valid: bool, opening_filtered: float | None,
            lateral_filtered: float | None = 0.0) -> Sample:
    return Sample(
        session_id="sessao_teste",
        frame=i,
        timestamp=f"{i * 0.1:.3f}s",
        time_s=i * 0.1,
        face_detected=True,
        frame_valid=frame_valid,
        opening_raw=(20.0 if frame_valid else 5.0),
        opening_rel=(0.20 if frame_valid else 0.02),
        opening_filtered=opening_filtered,
        opening_mm=None,
        lateral_raw=0.0,
        lateral_rel=0.0,
        lateral_filtered=(lateral_filtered if frame_valid else None),
        lateral_mm=None,
        direction="centro",
        cycle_state=MovementState.FECHADO,
        repetitions=0,
        quality_warning=(None if frame_valid else "Aproxime-se da camera (razao 0.020 < min 0.060)"),
    )


# --------------------------------------------------------------------------
# Sessao sem nenhum frame valido
# --------------------------------------------------------------------------
def test_session_without_valid_frames_exports_gracefully():
    rec = SessionRecorder()
    for i in range(10):
        rec.add(_sample(i, frame_valid=False, opening_filtered=None))
    cycles = CycleDetector()

    with tempfile.TemporaryDirectory() as d:
        paths = export_session(rec, cycles, d, "sessao_teste")
        with open(paths["resumo"], encoding="utf-8") as f:
            resumo = json.load(f)

    assert resumo["total_frames"] == 10
    assert resumo["frames_validos"] == 0
    assert resumo["percentual_valido"] == 0.0
    assert resumo["abertura_minima"] is None       # nao fabrica um "0.0"
    assert resumo["abertura_maxima"] is None
    assert not resumo["calibrado"]
    assert "aviso_calibracao" in resumo
    assert resumo["avisos_qualidade"]  # contagem de avisos presente


def test_plot_opening_time_shows_message_when_no_valid_frames():
    rec = SessionRecorder()
    for i in range(5):
        rec.add(_sample(i, frame_valid=False, opening_filtered=None))

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "abertura.png")
        out = plot_opening_time(rec, path)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0


# --------------------------------------------------------------------------
# Graficos: frame invalido vira NaN (lacuna), nunca zero
# --------------------------------------------------------------------------
def test_plot_series_uses_nan_not_zero_for_invalid_frames():
    rec = SessionRecorder()
    rec.add(_sample(0, True, 0.30))
    rec.add(_sample(1, False, None))
    rec.add(_sample(2, True, 0.40))

    t, opening_filt, _lat_filt, _o_raw, _l_raw, valid_mask, _unit = _series(rec, use_mm=False)

    assert opening_filt[0] == 0.30
    assert np.isnan(opening_filt[1])   # NAO e 0.0
    assert opening_filt[2] == 0.40
    assert list(valid_mask) == [True, False, True]


# --------------------------------------------------------------------------
# Calibracao usa somente frames validos
# --------------------------------------------------------------------------
def _face(opening_px: float, scale: float = 1.0, image_size=(600, 600)) -> FaceLandmarks:
    c = np.array([300.0, 300.0], dtype=np.float32)
    pts = np.zeros((N_LANDMARKS, 2), dtype=np.float32)
    pts[Landmark.EYE_OUTER_LEFT] = (200.0, 200.0)
    pts[Landmark.EYE_OUTER_RIGHT] = (400.0, 200.0)
    pts[Landmark.NASION] = (300.0, 210.0)
    pts[Landmark.CHIN] = (300.0, 450.0)
    pts[Landmark.UPPER_LIP_INNER] = (300.0, 350.0)
    pts[Landmark.LOWER_LIP_INNER] = (300.0, 350.0 + opening_px)
    pts = (pts - c) * scale + c
    return FaceLandmarks(points=pts, image_width=image_size[0], image_height=image_size[1])


def test_calibration_rejects_invalid_frames():
    """
    Replica o padrao usado em app.py/analyze_video.py (calib.update() so e
    chamado quando frame_valid=True). Um rosto minusculo demais (invalido)
    intercalado nao pode contaminar a calibracao guiada.
    """
    quality_cfg = QualityConfig()
    filt_o, filt_l, filt_w = EMAFilter(0.5), EMAFilter(0.5), EMAFilter(0.3)
    calib = CalibrationAssistant()
    calib.start()
    # calib.start() ancora phase_start em time.perf_counter(); "t" precisa
    # comecar no mesmo referencial (senao elapsed fica sempre negativo e a
    # calibracao nunca avanca de fase).
    t = calib.phase_start
    result = None
    while calib.active and result is None:
        if calib.phase == 0:
            good = _face(opening_px=10.0)              # boca fechada, rosto normal
        else:
            good = _face(opening_px=110.0)              # boca bem aberta, rosto normal
        bad = _face(opening_px=400.0, scale=0.01)        # rosto minusculo -> invalido

        for face in (bad, good):
            fr = process_frame(face, None, True, quality_cfg, filt_o, filt_l, filt_w)
            if fr.frame_valid:
                result = calib.update(fr.opening_filtered, t)
            t += 1.0 / 30

    assert result is not None
    assert result.valid
    assert result.closed < 0.2   # reflete o frame bom fechado (~0.05)
    assert result.opened > 0.3   # reflete o frame bom aberto (~0.55)
