"""
Testes de sincronizacao temporal do video anotado (VideoRecorder.write_paced).

Bug corrigido: o pipeline (deteccao facial) raramente processa a mesma taxa
do video de saida (ex.: 30 fps) -- normalmente processa MENOS. Escrever um
frame de video por frame processado faz o MP4 tocar mais rapido que a sessao
real (caso relatado: video 1.77x mais rapido, 496 frames processados a
~17fps codificados a 30fps).

A correcao paceia a escrita pelo tempo real decorrido da sessao
(`target_frame_index = floor(tempo_decorrido * output_fps)`), sem depender de
quantos frames o pipeline efetivamente processou nem de cap.get(CAP_PROP_FPS).
Estes testes sao sinteticos (sem webcam/MediaPipe): simulam sessoes com uma
taxa de processamento fixa e verificam a duracao/contagem do video resultante.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mandibular.recorder import Sample, SessionRecorder  # noqa: E402
from mandibular.metrics import MovementState  # noqa: E402
from mandibular.video_recorder import VideoRecorder  # noqa: E402

FRAME = np.zeros((32, 32, 3), dtype=np.uint8)
OUTPUT_FPS = 30.0


def _sample(i: int, t: float) -> Sample:
    return Sample(
        session_id="s", frame=i, timestamp=f"{t:.3f}s", time_s=t,
        face_detected=True, frame_valid=True,
        opening_raw=1.0, opening_rel=0.3, opening_filtered=0.3, opening_mm=None,
        lateral_raw=0.0, lateral_rel=0.0, lateral_filtered=0.0, lateral_mm=None,
        direction="centro", cycle_state=MovementState.FECHADO, repetitions=0,
        quality_warning=None,
    )


def _simulate_session(session_duration_s: float, processing_fps: float, output_fps: float,
                       video_path: str) -> tuple[SessionRecorder, VideoRecorder]:
    """
    Replica o loop de app.py (grava 1 amostra de CSV por frame processado e
    paceia o video pelo tempo real decorrido), sem camera/MediaPipe.
    """
    recorder = SessionRecorder()
    writer = VideoRecorder(video_path, output_fps, (32, 32))

    n_samples = int(round(session_duration_s * processing_fps))
    for i in range(n_samples):
        t = i / processing_fps
        recorder.add(_sample(i, t))
        target_frame_index = int(t * output_fps)
        writer.write_paced(FRAME, target_frame_index)

    # Encerramento da sessao: preenche ate o fim real (ver app._stop_session).
    target_frame_index = int(session_duration_s * output_fps)
    writer.write_paced(FRAME, target_frame_index)

    return recorder, writer


# --------------------------------------------------------------------------
# Sessao de 10s a 15 fps de processamento, salva a 30 fps (caso do bug real)
# --------------------------------------------------------------------------
def test_slow_pipeline_10s_session_at_15fps_saved_at_30fps():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "video.mp4")
        recorder, writer = _simulate_session(
            session_duration_s=10.0, processing_fps=15.0, output_fps=OUTPUT_FPS,
            video_path=path,
        )
        frame_count = writer.frame_count
        duration = writer.duration_s
        writer.close()

    # CSV mantem so as amostras REALMENTE processadas (sem duplicar).
    assert len(recorder.samples) == 150

    # Video final com ~300 frames (10s * 30fps), nao 150.
    assert abs(frame_count - 300) <= 1, frame_count

    # Duracao do video proxima de 10s (nao ~5s, que seria o bug original).
    assert abs(duration - 10.0) <= 1.0 / OUTPUT_FPS, duration


def test_video_duration_close_to_session_duration_regardless_of_processing_speed():
    """abs(video_duration_s - session_duration_s) <= ~1/output_fps (requisito 11)."""
    for processing_fps in (10.0, 15.0, 24.0, 30.0, 45.0, 60.0):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "video.mp4")
            _rec, writer = _simulate_session(
                session_duration_s=6.0, processing_fps=processing_fps,
                output_fps=OUTPUT_FPS, video_path=path,
            )
            duration = writer.duration_s
            writer.close()
        diff = abs(duration - 6.0)
        assert diff <= 1.0 / OUTPUT_FPS + 1e-9, (processing_fps, diff)


# --------------------------------------------------------------------------
# Pipeline mais RAPIDO que o fps de saida: nao pode acelerar o video
# --------------------------------------------------------------------------
def test_fast_pipeline_does_not_speed_up_video():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "video.mp4")
        recorder, writer = _simulate_session(
            session_duration_s=5.0, processing_fps=60.0, output_fps=OUTPUT_FPS,
            video_path=path,
        )
        frame_count = writer.frame_count
        duration = writer.duration_s
        writer.close()

    assert len(recorder.samples) == 300  # 5s * 60fps processados (CSV completo)
    assert abs(frame_count - 150) <= 1, frame_count  # video continua em 5s * 30fps
    assert abs(duration - 5.0) <= 1.0 / OUTPUT_FPS, duration


# --------------------------------------------------------------------------
# Preenchimento no encerramento da sessao (frames escassos no fim)
# --------------------------------------------------------------------------
def test_session_end_pads_last_interval():
    """
    Se os ultimos frames processados terminam bem antes do fim real da sessao
    (pipeline lento/instavel), o encerramento (R de parada) deve preencher o
    video ate session_duration_s * output_fps, nao deixar o video mais curto.
    """
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "video.mp4")
        writer = VideoRecorder(path, OUTPUT_FPS, (32, 32))

        # So 3 frames processados logo no inicio de uma sessao de 4s.
        for t in (0.0, 0.3, 0.6):
            writer.write_paced(FRAME, int(t * OUTPUT_FPS))
        count_before_end = writer.frame_count
        assert count_before_end < int(4.0 * OUTPUT_FPS)

        session_duration_s = 4.0
        writer.write_paced(FRAME, int(session_duration_s * OUTPUT_FPS))
        frame_count = writer.frame_count
        duration = writer.duration_s
        writer.close()

    assert abs(frame_count - int(session_duration_s * OUTPUT_FPS)) <= 1
    assert abs(duration - session_duration_s) <= 1.0 / OUTPUT_FPS


# --------------------------------------------------------------------------
# write_paced isolado: nao reescreve alem do necessario
# --------------------------------------------------------------------------
def test_write_paced_skips_when_target_already_reached():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "video.mp4")
        writer = VideoRecorder(path, OUTPUT_FPS, (32, 32))
        writer.write_paced(FRAME, 5)
        assert writer.frame_count == 5
        writer.write_paced(FRAME, 5)   # alvo repetido -> nao escreve de novo
        assert writer.frame_count == 5
        writer.write_paced(FRAME, 3)   # alvo menor (nao deveria ocorrer) -> idem
        assert writer.frame_count == 5
        writer.close()


def test_write_paced_duplicates_to_catch_up():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "video.mp4")
        writer = VideoRecorder(path, OUTPUT_FPS, (32, 32))
        writer.write_paced(FRAME, 10)  # pipeline atrasado: pula direto para o alvo 10
        assert writer.frame_count == 10
        writer.close()
