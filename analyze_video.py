"""
Analise offline de um video gravado (sem webcam ao vivo).

Reusa o mesmo pipeline de metricas do modo ao vivo (mandibular.pipeline.process_frame:
deteccao, controle de qualidade, filtragem EMA) para processar cada frame de
um arquivo de video e gerar a mesma pasta de sessao do modo ao vivo:
    - dados.csv, resumo.json, metadados.json;
    - abertura_tempo.png, lateralidade_tempo.png, trajetoria_*.png;
    - video anotado (opcional, --save-video).

Util para validacao com videos controlados e para reprocessar coletas sem
depender da camera.

Exemplos:
    python analyze_video.py coleta.mp4
    python analyze_video.py coleta.mp4 --ref-mm 63 --save-video
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "src")

import cv2  # noqa: E402

from mandibular.config import AppConfig, Landmark  # noqa: E402
from mandibular.exporter import export_session, make_session_id  # noqa: E402
from mandibular.filters import EMAFilter  # noqa: E402
from mandibular.landmarks import FaceMeshDetector  # noqa: E402
from mandibular.metrics import CycleDetector  # noqa: E402
from mandibular.overlay import draw_landmarks, draw_opening_bar, draw_panel  # noqa: E402
from mandibular.pipeline import process_frame  # noqa: E402
from mandibular.recorder import Sample, SessionRecorder  # noqa: E402
from mandibular.video_recorder import VideoRecorder  # noqa: E402

# Faixas de referencia (Biomecanica Funcional, Cap. 16 - ver docs/pesquisa).
REF_ABERTURA_MM = (40.0, 60.0)   # abertura da boca normal
REF_DIDUCAO_MM = (9.0, 12.0)     # amplitude de lateralizacao (diducao) normal


def analyze(
    path: str,
    ref_mm: float | None,
    calib_auto: bool,
    output_dir: str,
    save_video: bool,
    mirrored: bool,
) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video nao encontrado: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir o video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    print(f"Video: {path}  ({fps:.1f} fps, {n_frames} frames)")

    cfg = AppConfig()
    detector = FaceMeshDetector(cfg.detection)

    filt_opening = EMAFilter(cfg.filter.alpha_opening)
    filt_lateral = EMAFilter(cfg.filter.alpha_lateral)
    filt_face_width = EMAFilter(cfg.filter.alpha_face_width)

    # Unica passada de deteccao/filtragem: cada frame gera um FrameResult
    # (metricas, qualidade, valor filtrado ou None se invalido). So os
    # frames VALIDOS entram na amostra usada para a calibracao automatica
    # (percentis) - um rosto longe demais/inclinado nao pode influenciar os
    # limiares de "fechado"/"aberto".
    frame_results: list[tuple[int, float, object, object]] = []  # (idx, t, face, FrameResult)
    valid_openings_rel: list[float] = []
    idx = 0
    faces_ok = 0
    prev_nasion = None
    last_pct = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        ts_ms = int(t * 1000)
        face = detector.process(frame, ts_ms)

        fr = process_frame(
            face, ref_mm, mirrored, cfg.quality,
            filt_opening, filt_lateral, filt_face_width,
            prev_nasion=prev_nasion,
        )
        prev_nasion = face.point(Landmark.NASION) if face is not None else None

        if face is not None:
            faces_ok += 1
        if fr.frame_valid and fr.metrics is not None:
            valid_openings_rel.append(fr.metrics.opening_rel)

        frame_results.append((idx, t, face, fr))
        idx += 1

        if n_frames > 0:
            pct = int(100 * idx / n_frames)
            if pct != last_pct and pct % 10 == 0:
                print(f"  processando: {pct}%")
                last_pct = pct
    cap.release()

    if not frame_results or faces_ok == 0:
        detector.close()
        print("Nenhuma face detectada no video.")
        return

    print(f"Faces detectadas em {faces_ok}/{idx} frames "
          f"({100 * faces_ok / max(idx, 1):.0f}%).  "
          f"Frames validos para calibracao: {len(valid_openings_rel)}.")

    cycles = CycleDetector(cfg.cycle)
    if calib_auto and valid_openings_rel:
        arr = np.array(valid_openings_rel)
        closed = float(np.percentile(arr, 5))
        opened = float(np.percentile(arr, 95))
        cycles.calibrate(closed, opened)
        print(f"Calibracao automatica: fechado~{closed:.3f}  aberto~{opened:.3f} (rel)")
    elif calib_auto:
        print("Calibracao automatica pulada: nenhum frame valido no video "
              "(veja aviso_qualidade no CSV/resumo para o motivo).")

    session_id = make_session_id()
    base = os.path.splitext(os.path.basename(path))[0]
    session_dir = os.path.join(output_dir, f"{base}_{session_id}")
    os.makedirs(session_dir, exist_ok=True)

    video_writer = None
    if save_video:
        video_writer = VideoRecorder(
            os.path.join(session_dir, "video.mp4"), fps, (frame_w, frame_h)
        )

    # Reabre o video para desenhar/gravar (o VideoCapture original ja foi
    # liberado); a deteccao e as metricas NAO sao recalculadas aqui, apenas
    # reusadas de frame_results.
    cap2 = cv2.VideoCapture(path) if save_video else None

    recorder = SessionRecorder()
    for frame_idx, t, face, fr in frame_results:
        frame_for_video = None
        if cap2 is not None:
            ok, frame_for_video = cap2.read()
            if not ok:
                frame_for_video = None

        if fr.frame_valid:
            cycles.update(fr.opening_filtered, t)

        m = fr.metrics
        recorder.add(
            Sample(
                session_id=f"{base}_{session_id}",
                frame=frame_idx,
                timestamp=f"{t:.3f}s",
                time_s=t,
                face_detected=face is not None,
                frame_valid=fr.frame_valid,
                opening_raw=(m.opening_px if m is not None else None),
                opening_rel=(m.opening_rel if m is not None else None),
                opening_filtered=fr.opening_filtered,
                opening_mm=(m.opening_mm if (m is not None and fr.frame_valid) else None),
                lateral_raw=(m.lateral_px if m is not None else None),
                lateral_rel=(m.lateral_rel if m is not None else None),
                lateral_filtered=fr.lateral_filtered,
                lateral_mm=(m.lateral_mm if (m is not None and fr.frame_valid) else None),
                direction=fr.direction,
                cycle_state=cycles.state,
                repetitions=cycles.repetitions,
                quality_warning=fr.quality.message,
            )
        )

        if video_writer is not None and frame_for_video is not None:
            if face is not None:
                draw_landmarks(frame_for_video, face)
                draw_opening_bar(frame_for_video, fr.opening_display, cycles)
            lines = [(f"Abertura: {fr.opening_display:.3f} rel", (255, 255, 255))]
            lines.append((f"Desvio: {fr.lateral_display:+.3f} [{fr.direction}]", (255, 255, 255)))
            lines.append((f"Estado: {cycles.state.value.upper()}", (0, 220, 0)))
            lines.append((f"Repeticoes: {cycles.repetitions}", (255, 255, 255)))
            if not fr.frame_valid:
                lines.append(("Frame invalido - ultimo valor valido exibido", (0, 180, 255)))
            if fr.quality.message:
                lines.append((fr.quality.message, (0, 180, 255)))
            draw_panel(frame_for_video, lines)
            video_writer.write(frame_for_video)

    detector.close()
    if cap2 is not None:
        cap2.release()
    if video_writer is not None:
        video_writer.close()

    paths = export_session(
        recorder,
        cycles,
        session_dir,
        f"{base}_{session_id}",
        ref_mm=ref_mm,
        video_path=(os.path.join(session_dir, "video.mp4") if save_video else None),
        extra_metadata={"origem": "analise_offline", "arquivo_video": path},
    )

    _print_summary(recorder, cycles, ref_mm)
    print(f"\n[export] pasta: {session_dir}")
    for k, v in paths.items():
        print(f"[export] {k}: {v}")


def _print_summary(recorder, cycles, ref_mm) -> None:
    valid_samples = [s for s in recorder.samples if s.frame_valid]
    total = len(recorder.samples)
    pct = (100.0 * len(valid_samples) / total) if total else 0.0

    print("\n" + "=" * 52)
    print("RESUMO DA ANALISE")
    print("=" * 52)
    print(f"Frames totais: {total}  |  Frames validos: {len(valid_samples)} ({pct:.0f}%)")
    if not cycles.is_calibrated:
        print("AVISO: sessao NAO calibrada (repeticoes, se houver, usam faixa dinamica).")

    if not valid_samples:
        print("Nenhum frame valido para calcular metricas de abertura/desvio.")
        return

    openings_rel = np.array([s.opening_rel for s in valid_samples])
    lateral_rel = np.array([s.lateral_rel for s in valid_samples])

    print(f"Repeticoes (ciclos abre/fecha): {cycles.repetitions}")
    print(f"Abertura maxima:  {openings_rel.max():.3f} rel")
    print(f"Desvio lateral:   min {lateral_rel.min():+.3f} / max {lateral_rel.max():+.3f} rel")
    print(f"Desvio lat. medio:{lateral_rel.mean():+.3f} rel (assimetria do caminho)")

    rep = cycles.repeatability()
    if rep:
        print(f"\nRepetibilidade ({int(rep['n_ciclos'])} ciclos):")
        print(f"  amplitude: media {rep['amplitude_media']:.3f}  "
              f"CV {rep['amplitude_cv']:.2f}")
        print(f"  duracao:   media {rep['duracao_media_s']:.2f}s  "
              f"CV {rep['duracao_cv']:.2f}")
        print("  (CV menor = movimento mais repetivel/consistente)")

    if ref_mm is not None:
        mm_samples = [s for s in valid_samples if s.opening_mm is not None]
        if mm_samples:
            ab_mm = max(s.opening_mm for s in mm_samples)
            lat_mm = [s.lateral_mm for s in mm_samples]
            lat_amp_mm = max(lat_mm) - min(lat_mm)
            print("\nComparacao com faixas de referencia (em mm):")
            print(f"  abertura max = {ab_mm:.1f} mm  "
                  f"(referencia normal: {REF_ABERTURA_MM[0]:.0f}-{REF_ABERTURA_MM[1]:.0f} mm)")
            print(f"  amplitude lateral = {lat_amp_mm:.1f} mm  "
                  f"(referencia diducao: {REF_DIDUCAO_MM[0]:.0f}-{REF_DIDUCAO_MM[1]:.0f} mm)")
            print("  >> Valores estimados; apoio, nao diagnostico.")


def main() -> None:
    p = argparse.ArgumentParser(description="Analise offline de video mandibular.")
    p.add_argument("video", help="caminho do arquivo de video")
    p.add_argument("--ref-mm", type=float, default=None,
                   help="distancia real (mm) entre os cantos externos dos olhos")
    p.add_argument("--calib-auto", action="store_true", default=True,
                   help="calibra a faixa pelos percentis do proprio video, so com frames "
                   "validos (padrao: ligado)")
    p.add_argument("--no-calib-auto", dest="calib_auto", action="store_false",
                   help="desliga a calibracao automatica (usa faixa dinamica min/max)")
    p.add_argument("--save-video", action="store_true",
                   help="salva um video anotado (landmarks, valores, estado) na pasta da sessao")
    p.add_argument("--mirrored", action="store_true",
                   help="trata o video como espelhado ao rotular a direcao do desvio")
    p.add_argument("--output", default="resultados", help="pasta de saida")
    a = p.parse_args()
    try:
        analyze(a.video, a.ref_mm, a.calib_auto, a.output, a.save_video, a.mirrored)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
