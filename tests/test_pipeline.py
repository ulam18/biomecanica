"""
Testes do restante do pipeline (sem webcam): filtro EMA, controle de
qualidade do frame e validacao de sessoes para comparacao.
"""

import csv
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mandibular.config import Landmark, QualityConfig  # noqa: E402
from mandibular.filters import EMAFilter  # noqa: E402
from mandibular.landmarks import FaceLandmarks  # noqa: E402
from mandibular.quality import FrameQuality, assess_quality  # noqa: E402

N_LANDMARKS = 478


def make_face(
    opening_px: float = 0.0,
    lateral_px: float = 0.0,
    scale: float = 1.0,
    roll_deg: float = 0.0,
    center: tuple = (300.0, 300.0),
    image_size: tuple = (600, 600),
) -> FaceLandmarks:
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

    c = np.array(center, dtype=np.float32)
    pts = (pts - c) * scale + c
    if roll_deg:
        a = math.radians(roll_deg)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]],
                       dtype=np.float32)
        pts = (pts - c) @ rot.T + c
    return FaceLandmarks(points=pts, image_width=image_size[0], image_height=image_size[1])


# --------------------------------------------------------------------------
# Filtro EMA
# --------------------------------------------------------------------------
def test_ema_filter_converges_towards_signal():
    f = EMAFilter(alpha=0.5)
    v = None
    for _ in range(30):
        v = f.update(1.0)
    assert abs(v - 1.0) < 1e-6


def test_ema_filter_smooths_single_spike():
    f = EMAFilter(alpha=0.3)
    for _ in range(10):
        f.update(0.2)
    spiked = f.update(0.9)  # ruido isolado
    assert spiked < 0.9  # nao segue o pico integralmente (suaviza)
    assert spiked > 0.2  # mas reage na direcao certa (sem atraso exagerado)


def test_ema_filter_reset():
    f = EMAFilter(alpha=0.5)
    f.update(1.0)
    f.reset()
    assert f.value is None
    assert f.update(0.3) == 0.3  # primeira amostra apos reset nao e suavizada


# --------------------------------------------------------------------------
# Controle de qualidade
# --------------------------------------------------------------------------
def test_quality_face_not_detected():
    result = assess_quality(None, None, QualityConfig())
    assert result.quality == FrameQuality.NAO_DETECTADA


def test_quality_valid_frame():
    face = make_face(opening_px=50.0)
    result = assess_quality(face, None, QualityConfig())
    assert result.quality == FrameQuality.VALIDA
    assert result.message is None


def test_quality_face_too_small():
    face = make_face(opening_px=50.0, scale=0.1)  # afasta muito o rosto da camera
    result = assess_quality(face, None, QualityConfig())
    assert result.quality == FrameQuality.INVALIDA
    assert "Aproxime" in result.message


def test_quality_face_too_close():
    # Imagem larga e baixa para permitir uma face_width proxima da diagonal
    # (proporcao possivel de ocorrer com um rosto muito perto da camera),
    # sem que os pontos fiquem fora dos limites da imagem.
    pts = np.zeros((N_LANDMARKS, 2), dtype=np.float32)
    pts[Landmark.EYE_OUTER_LEFT] = (20.0, 75.0)
    pts[Landmark.EYE_OUTER_RIGHT] = (950.0, 75.0)
    pts[Landmark.NASION] = (485.0, 60.0)
    pts[Landmark.CHIN] = (485.0, 140.0)
    face = FaceLandmarks(points=pts, image_width=1000, image_height=150)

    result = assess_quality(face, None, QualityConfig())
    assert result.quality == FrameQuality.INVALIDA
    assert "Afaste" in result.message


def test_quality_excessive_roll():
    face = make_face(opening_px=50.0, roll_deg=45.0)
    result = assess_quality(face, None, QualityConfig())
    assert result.quality == FrameQuality.INVALIDA
    assert "estavel" in result.message


def test_quality_global_jump_between_frames():
    prev_nasion = np.array([1000.0, 1000.0], dtype=np.float32)  # posicao muito distante
    face = make_face(opening_px=50.0)
    result = assess_quality(face, prev_nasion, QualityConfig())
    assert result.quality == FrameQuality.INVALIDA


# --------------------------------------------------------------------------
# Razao de tamanho facial normalizada pela LARGURA do frame (nao a diagonal)
# --------------------------------------------------------------------------
def make_face_with_ratio(frame_w: int, frame_h: int, ratio: float,
                          opening_px: float = 50.0, lateral_px: float = 0.0) -> FaceLandmarks:
    """Constroi um rosto sintetico centralizado com face_width_px/frame_w == ratio."""
    face_width = ratio * frame_w
    cx, cy = frame_w / 2.0, frame_h / 2.0
    pts = np.zeros((N_LANDMARKS, 2), dtype=np.float32)
    pts[Landmark.EYE_OUTER_LEFT] = (cx - face_width / 2.0, cy - face_width / 4.0)
    pts[Landmark.EYE_OUTER_RIGHT] = (cx + face_width / 2.0, cy - face_width / 4.0)
    pts[Landmark.NASION] = (cx, cy - face_width / 4.0 + face_width * 0.05)
    pts[Landmark.CHIN] = (cx + lateral_px, cy + face_width / 2.0)
    pts[Landmark.UPPER_LIP_INNER] = (cx, cy + face_width * 0.25)
    pts[Landmark.LOWER_LIP_INNER] = (cx, cy + face_width * 0.25 + opening_px)
    return FaceLandmarks(points=pts, image_width=frame_w, image_height=frame_h)


def test_face_ratio_normalized_640x480():
    face = make_face_with_ratio(640, 480, ratio=0.10)
    result = assess_quality(face, None, QualityConfig())
    assert result.ratio is not None
    assert abs(result.ratio - 0.10) < 1e-3
    assert result.quality == FrameQuality.VALIDA


def test_face_ratio_normalized_1280x720():
    face = make_face_with_ratio(1280, 720, ratio=0.10)
    result = assess_quality(face, None, QualityConfig())
    assert result.ratio is not None
    assert abs(result.ratio - 0.10) < 1e-3
    assert result.quality == FrameQuality.VALIDA


def test_face_ratio_invariant_across_resolutions():
    """A mesma proporcao real do rosto perante a camera deve dar a mesma
    razao e a mesma classificacao de qualidade, em qualquer resolucao."""
    r_640 = assess_quality(make_face_with_ratio(640, 480, ratio=0.10), None, QualityConfig())
    r_1280 = assess_quality(make_face_with_ratio(1280, 720, ratio=0.10), None, QualityConfig())
    assert abs(r_640.ratio - r_1280.ratio) < 1e-6
    assert r_640.quality == r_1280.quality == FrameQuality.VALIDA


def test_regression_realistic_webcam_distance_is_no_longer_invalid():
    """
    Reproduz o bug relatado: um rosto a distancia tipica de uso de webcam em
    1280x720 (razao ~0.09, equivalente a ~60cm da camera) NAO pode mais ser
    marcado invalido so por estar "um pouco distante". O limiar antigo
    (0.15 * diagonal da imagem) exigia ~17% da largura do frame, so
    atingivel a menos de ~30cm - por isso a sessao inteira era invalidada.
    """
    face = make_face_with_ratio(1280, 720, ratio=0.09)
    result = assess_quality(face, None, QualityConfig())
    assert result.quality == FrameQuality.VALIDA, result.message


# --------------------------------------------------------------------------
# Comparacao entre sessoes: validacao de colunas
# --------------------------------------------------------------------------
def _write_csv(path: str, columns: list, n_rows: int = 5) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for i in range(n_rows):
            row = []
            for c in columns:
                if c in ("estado_ciclo",):
                    row.append("fechado")
                elif c == "frame":
                    row.append(i)
                else:
                    row.append(f"{0.1 * i:.4f}")
            writer.writerow(row)


def test_compare_sessions_rejects_missing_columns():
    import compare_sessions

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dados.csv")
        _write_csv(path, ["frame", "tempo_s"])  # faltam colunas obrigatorias
        try:
            compare_sessions.load_session(path)
            assert False, "deveria ter lancado ValueError"
        except ValueError:
            pass


def test_compare_sessions_loads_valid_csv():
    import compare_sessions

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dados.csv")
        _write_csv(path, list(compare_sessions.REQUIRED_COLUMNS))
        data = compare_sessions.load_session(path)
        assert len(data.rows) == 5
