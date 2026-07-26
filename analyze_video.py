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
from mandibular.metrics import CycleDetector, lateral_direction  # noqa: E402
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
    valid_laterals_rel: list[float] = []
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

        # 1a passada: lateral_baseline ainda e desconhecido (so e definido
        # DEPOIS de ver o video inteiro, via percentis - ver abaixo), entao
        # lateral_dynamic e recalculado na 2a passada (loop de exportacao).
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
            valid_laterals_rel.append(fr.metrics.lateral_rel)

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
    lateral_baseline: float | None = None
    if calib_auto and valid_openings_rel:
        arr = np.array(valid_openings_rel)
        lat_arr = np.array(valid_laterals_rel)
        closed = float(np.percentile(arr, 5))
        opened = float(np.percentile(arr, 95))
        # Baseline lateral: mediana da lateral_absolute nos frames "mais
        # fechados" (abertura <= percentil 25), mesma logica da calibracao
        # guiada ao vivo (mediana, so frames validos), so que sem fases
        # explicitas -- aqui a fase "fechada" e aproximada pelo quartil
        # inferior de abertura observado no proprio video.
        closed_mask = arr <= np.percentile(arr, 25)
        lateral_baseline_std: float | None = None
        if np.any(closed_mask):
            closed_laterals = lat_arr[closed_mask]
            lateral_baseline = float(np.median(closed_laterals))
            lateral_baseline_std = float(np.std(closed_laterals))
        cycles.calibrate(closed, opened, lateral_baseline, lateral_baseline_std)
        print(f"Calibracao automatica: fechado~{closed:.3f}  aberto~{opened:.3f} (rel)"
              + (f"  |  lateral_baseline~{lateral_baseline:.3f}" if lateral_baseline is not None else ""))
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

        m = fr.metrics
        q = fr.quality

        # lateral_dynamic recalculado aqui (baseline so ficou conhecido apos
        # a 1a passada); mesma formula de pipeline.process_frame
        # (absoluto - baseline), sem reprocessar deteccao/MediaPipe.
        lateral_dynamic_raw = (
            m.lateral_rel - lateral_baseline
            if (m is not None and lateral_baseline is not None) else None
        )
        lateral_dynamic_filtered = (
            fr.lateral_filtered - lateral_baseline
            if (fr.lateral_filtered is not None and lateral_baseline is not None) else None
        )
        # direcao: mesma prioridade de pipeline.process_frame (dinamico
        # quando disponivel, senao absoluto) -- recalculada aqui porque a 1a
        # passada rodou sem baseline (ainda desconhecido nesse ponto).
        direction = (
            lateral_direction(lateral_dynamic_filtered, mirrored)
            if fr.frame_valid and lateral_dynamic_filtered is not None
            else fr.direction
        )

        if fr.frame_valid:
            cycles.update(
                fr.opening_filtered, t,
                lateral_absolute=fr.lateral_filtered,
                lateral_dynamic=lateral_dynamic_filtered,
            )

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
                direction=direction,
                cycle_state=cycles.state,
                repetitions=cycles.repetitions,
                quality_warning=q.message,
                quality_reason=q.reason,
                face_size_ratio=q.ratio,
                roll_deg=q.roll_deg,
                yaw_deg=q.yaw_deg,
                pitch_deg=q.pitch_deg,
                lateral_neutral_baseline=lateral_baseline,
                lateral_dynamic_raw=lateral_dynamic_raw,
                lateral_dynamic_filtered=lateral_dynamic_filtered,
            )
        )

        if video_writer is not None and frame_for_video is not None:
            if face is not None:
                draw_landmarks(frame_for_video, face)
                draw_opening_bar(frame_for_video, fr.opening_display, cycles)
            lines = [(f"Abertura: {fr.opening_display:.3f}", (255, 255, 255))]
            lines.append((f"Posicao do queixo: {fr.lateral_display:+.3f}", (255, 255, 255)))
            if lateral_baseline is not None:
                dyn_display = (
                    lateral_dynamic_filtered if lateral_dynamic_filtered is not None
                    else fr.lateral_display - lateral_baseline
                )
                lines.append((f"Movimento desde o neutro: {dyn_display:+.3f}", (255, 255, 255)))
            lines.append((f"Direcao: {direction}", (255, 255, 255)))
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
    video_frame_count = video_writer.frame_count if video_writer is not None else 0
    video_dur = video_writer.duration_s if video_writer is not None else 0.0
    if video_writer is not None:
        video_writer.close()

    session_duration_s = frame_results[-1][1] if frame_results else 0.0
    paths = export_session(
        recorder,
        cycles,
        session_dir,
        f"{base}_{session_id}",
        ref_mm=ref_mm,
        video_path=(os.path.join(session_dir, "video.mp4") if save_video else None),
        mirrored=mirrored,
        extra_metadata={
            "origem": "analise_offline",
            "arquivo_video": path,
            "resolucao": [frame_w, frame_h],
            "espelhado": mirrored,
            "duracao_sessao_s": session_duration_s,
            "repeticoes_sessao": cycles.repetitions,
            "video_habilitado": save_video,
            "video_frames": video_frame_count,
            "video_fps": fps,
            "video_duracao_s": video_dur,
            "quality_thresholds": {
                "min_face_width_ratio": cfg.quality.min_face_width_ratio,
                "max_face_width_ratio": cfg.quality.max_face_width_ratio,
                "max_roll_deg": cfg.quality.max_roll_deg,
                "max_yaw_deg": cfg.quality.max_yaw_deg,
                "max_pitch_deg": cfg.quality.max_pitch_deg,
                "max_global_jump_fraction": cfg.quality.max_global_jump_fraction,
                "reject_multiple_faces": cfg.quality.reject_multiple_faces,
            },
        },
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
