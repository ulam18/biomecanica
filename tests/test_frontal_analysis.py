"""
Testes das analises faciais frontais/perfil portadas sobre a arquitetura de
pipeline: simetria, angulos frontais (desvio mandibular), proporcoes, perfil,
classificacao (com fontes) e evolucao do paciente.

Sao independentes de webcam (usam landmarks sinteticos) e do test_metrics.py
original (que cobre a pipeline).
"""

import csv
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mandibular.classification import (
    PROFILE_ANGLE_NORMS,
    classify_diduction,
    classify_opening,
    classify_profile,
    classify_profile_prominence,
    classify_symmetry,
)
from mandibular.config import Landmark
from mandibular.evolution import (
    append_evolution,
    evolution_path,
    load_evolution,
    summarize_session,
)
from mandibular.landmarks import FaceLandmarks
from mandibular.metrics import (
    MovementState,
    ProfileAngles,
    compute_frontal_angles,
    compute_profile_angles,
    compute_profile_metrics,
    compute_proportions,
    compute_symmetry,
)
from mandibular.recorder import Sample, SessionRecorder

N = 478


def make_face(opening_px=100.0, lateral_px=0.0, roll_deg=0.0, asym_px=0.0):
    pts = np.zeros((N, 2), dtype=np.float32)
    pts[Landmark.EYE_OUTER_LEFT] = (200.0, 200.0)
    pts[Landmark.EYE_OUTER_RIGHT] = (400.0, 200.0)
    pts[Landmark.EYE_INNER_LEFT] = (260.0, 200.0)
    pts[Landmark.EYE_INNER_RIGHT] = (340.0, 200.0)
    pts[Landmark.NASION] = (300.0, 210.0)
    pts[Landmark.CHIN] = (300.0 + lateral_px, 450.0)
    pts[Landmark.UPPER_LIP_INNER] = (300.0, 350.0)
    pts[Landmark.LOWER_LIP_INNER] = (300.0, 350.0 + opening_px)
    pts[Landmark.NOSE_TIP] = (300.0, 300.0)
    pts[Landmark.MOUTH_LEFT] = (270.0, 350.0)
    pts[Landmark.MOUTH_RIGHT] = (330.0 + asym_px, 350.0)
    pts[Landmark.NOSE_ALA_LEFT] = (280.0, 300.0)
    pts[Landmark.NOSE_ALA_RIGHT] = (320.0, 300.0)
    pts[Landmark.CHEEK_LEFT] = (210.0, 320.0)
    pts[Landmark.CHEEK_RIGHT] = (390.0, 320.0)
    pts[Landmark.FOREHEAD] = (300.0, 120.0)
    pts[Landmark.GLABELA] = (300.0, 180.0)
    pts[Landmark.SUBNASALE] = (300.0, 320.0)
    pts[Landmark.LABIALE_SUP] = (300.0, 345.0)
    pts[Landmark.LABIALE_INF] = (300.0, 365.0)
    pts[Landmark.SUBLABIALE] = (300.0, 405.0)

    if roll_deg:
        c = np.array([300.0, 300.0], dtype=np.float32)
        a = math.radians(roll_deg)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]],
                       dtype=np.float32)
        pts = (pts - c) @ rot.T + c
    return FaceLandmarks(points=pts, image_width=600, image_height=600)


# -- Simetria ---------------------------------------------------------------
def test_symmetry_perfect():
    sym = compute_symmetry(make_face())
    assert sym.index > 0.99 and sym.is_frontal


def test_symmetry_detects_asymmetry():
    assert compute_symmetry(make_face(asym_px=40.0)).index < compute_symmetry(make_face()).index - 0.05


# -- Angulos frontais (desvio mandibular) -----------------------------------
def test_mandibular_deviation_sign():
    assert abs(compute_frontal_angles(make_face(lateral_px=0.0)).mand_deviation_deg) < 0.5
    assert compute_frontal_angles(make_face(lateral_px=60.0)).mand_deviation_deg > 3.0
    assert compute_frontal_angles(make_face(lateral_px=-60.0)).mand_deviation_deg < -3.0


def test_mandibular_deviation_roll_invariance():
    a = compute_frontal_angles(make_face(lateral_px=40.0, roll_deg=0.0))
    b = compute_frontal_angles(make_face(lateral_px=40.0, roll_deg=18.0))
    assert abs(a.mand_deviation_deg - b.mand_deviation_deg) < 0.5


# -- Proporcoes / perfil ----------------------------------------------------
def test_proportions_and_profile_ranges():
    p = compute_proportions(make_face())
    assert p.mouth_within_canon and p.fifths_ratio > 0
    ang = compute_profile_angles(make_face())
    for v in (ang.convexidade_facial, ang.convexidade_total, ang.nasofrontal,
              ang.nasolabial, ang.labiomental):
        assert 0.0 <= v <= 180.0
    assert compute_profile_metrics(make_face()).facing in ("direito", "esquerdo")


# -- Classificacao ----------------------------------------------------------
def test_classifications():
    assert classify_opening(50.0).category == "normal"
    assert classify_opening(30.0).category == "reduzida"
    assert classify_diduction(10.0).category == "normal"
    assert "Classe I" in classify_profile(170.0).category
    assert classify_symmetry(0.95).category == "alta"
    assert "sem corte clinico" in classify_symmetry(0.95).note


def test_profile_prominence_flags():
    base = {k: v[0] for k, v in PROFILE_ANGLE_NORMS.items()}
    a = ProfileAngles(150.0, base["convexidade_total"], base["nasofrontal"],
                      base["nasolabial"], base["labiomental"])
    f = next(x for x in classify_profile_prominence(a) if x.key == "convexidade_facial")
    assert f.out_of_norm and "convexo" in f.category


# -- Evolucao (compativel com o Sample do pipeline) -------------------------
def _sample(lateral_rel, lateral_mm, opening_mm):
    return Sample(
        session_id="s", frame=0, timestamp="t", time_s=0.0,
        face_detected=True, frame_valid=True,
        opening_raw=1.0, opening_rel=0.5, opening_filtered=0.5, opening_mm=opening_mm,
        lateral_raw=1.0, lateral_rel=lateral_rel, lateral_filtered=lateral_rel,
        lateral_mm=lateral_mm, direction="centro",
        cycle_state=MovementState.ABERTO, repetitions=2, quality_warning=None,
    )


def test_evolution_roundtrip_with_pipeline_sample():
    with tempfile.TemporaryDirectory() as d:
        path = evolution_path(d, "Maria Teste")
        for lat_mm in (8.0, 5.0, 3.0):
            rec = SessionRecorder()
            for _ in range(5):
                rec.add(_sample(lat_mm / 60, lat_mm, 55.0))
            append_evolution(path, summarize_session(rec, None, "Maria Teste",
                                                     "2026-07-24 10:00:00"))
        rows = load_evolution(path)
    assert len(rows) == 3
    assert [float(r["desvio_lat_medio_mm"]) for r in rows] == [8.0, 5.0, 3.0]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"OK  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FALHOU  {fn.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} testes passaram.")
    sys.exit(1 if failed else 0)
