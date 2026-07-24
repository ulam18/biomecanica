"""
Aplicacao em tempo real: captura por webcam, deteccao facial, filtragem,
controle de qualidade, calculo das metricas, deteccao de ciclos, biofeedback,
interface visual e exportacao de dados (CSV, JSON, graficos e video opcional).

Controles do teclado:
    C  - calibrar (assistente: boca fechada -> boca aberta)
    R  - iniciar/parar gravacao dos dados da sessao
    E  - exportar a sessao gravada
    Z  - zerar sessao (amostras, filtros e contagem de repeticoes)
    V  - ligar/desligar gravacao do video anotado
    Q / ESC - sair
"""

from __future__ import annotations

import os
import time
from datetime import datetime

import cv2
import numpy as np

from .calibration import CalibrationAssistant
from .config import AppConfig, Landmark
from .exporter import export_session, make_session_id
from .feedback import biofeedback_messages
from .filters import EMAFilter
from .landmarks import FaceMeshDetector
from .metrics import CycleDetector, MovementState
from .overlay import C_ALERTA, C_OK, C_REC, C_TEXTO, draw_landmarks, draw_opening_bar, draw_panel
from .pipeline import process_frame
from .recorder import Sample, SessionRecorder
from .video_recorder import VideoRecorder


class MandibularApp:
    def __init__(self, config: AppConfig | None = None):
        self.cfg = config or AppConfig()
        self.detector = FaceMeshDetector(self.cfg.detection)
        self.cycles = CycleDetector(self.cfg.cycle)
        self.recorder = SessionRecorder()
        self.calib = CalibrationAssistant()

        self.filt_opening = EMAFilter(self.cfg.filter.alpha_opening)
        self.filt_lateral = EMAFilter(self.cfg.filter.alpha_lateral)
        self.filt_face_width = EMAFilter(self.cfg.filter.alpha_face_width)

        self.recording = False
        self.video_recording = False
        self.video_writer: VideoRecorder | None = None
        self.session_id: str | None = None
        self.session_dir: str | None = None

        self.frame_idx = 0
        self.t0 = time.perf_counter()
        self._last_ts_ms = -1
        self._prev_nasion: np.ndarray | None = None
        self.last_status = ""

    # -- Sessao -------------------------------------------------------------
    def _ensure_session(self) -> None:
        if self.session_id is None:
            self.session_id = make_session_id()
            self.session_dir = os.path.join(self.cfg.output_dir, self.session_id)
            os.makedirs(self.session_dir, exist_ok=True)

    # -- Loop principal -------------------------------------------------------
    def run(self) -> None:
        cap = cv2.VideoCapture(self.cfg.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)

        if not cap.isOpened():
            raise RuntimeError(
                f"Nao foi possivel abrir a camera (indice {self.cfg.camera_index})."
            )
        cam_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

        win = "Reconhecimento Mandibular - Biomecanica"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if self.cfg.flip_horizontal:
                    frame = cv2.flip(frame, 1)

                now = time.perf_counter()
                t = now - self.t0
                ts_ms = max(int(t * 1000), self._last_ts_ms + 1)
                self._last_ts_ms = ts_ms
                face = self.detector.process(frame, ts_ms)

                fr = process_frame(
                    face,
                    self.cfg.reference_distance_mm,
                    self.cfg.flip_horizontal,
                    self.cfg.quality,
                    self.filt_opening,
                    self.filt_lateral,
                    self.filt_face_width,
                    prev_nasion=self._prev_nasion,
                )
                self._prev_nasion = face.point(Landmark.NASION) if face is not None else None

                if face is not None:
                    draw_landmarks(frame, face)

                if self.calib.active and fr.frame_valid:
                    result = self.calib.update(fr.opening_filtered, now)
                    if result is not None:
                        if result.valid:
                            self.cycles.calibrate(result.closed, result.opened)
                        self.last_status = result.message

                if not self.calib.active and fr.frame_valid:
                    self.cycles.update(fr.opening_filtered, t)

                feedback = (
                    []
                    if self.calib.active
                    else biofeedback_messages(fr.opening_display, fr.direction, self.cycles, fr.quality)
                )

                if face is not None:
                    draw_opening_bar(frame, fr.opening_display, self.cycles)
                self._draw_hud(frame, fr, feedback, now)

                if self.recording:
                    self._ensure_session()
                    m = fr.metrics
                    self.recorder.add(
                        Sample(
                            session_id=self.session_id,
                            frame=self.frame_idx,
                            timestamp=datetime.now().isoformat(timespec="milliseconds"),
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
                            cycle_state=self.cycles.state,
                            repetitions=self.cycles.repetitions,
                            quality_warning=fr.quality.message,
                        )
                    )

                if self.video_recording and self.video_writer is not None:
                    self.video_writer.write(frame)

                self.frame_idx += 1
                cv2.imshow(win, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):  # q ou ESC
                    break
                elif key == ord("c"):
                    self.calib.start()
                    self.last_status = "Iniciando calibracao..."
                elif key == ord("r"):
                    self.recording = not self.recording
                    if self.recording:
                        self._ensure_session()
                    self.last_status = (
                        "Gravacao iniciada." if self.recording else "Gravacao pausada."
                    )
                elif key == ord("v"):
                    self._toggle_video(frame, cam_fps)
                elif key == ord("e"):
                    self._export()
                elif key == ord("z"):
                    self._reset()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()
            if self.video_writer is not None:
                self.video_writer.close()
                self.video_writer = None

    # -- HUD ------------------------------------------------------------
    def _draw_hud(self, frame, fr, feedback: list[str], now: float) -> None:
        m, quality, direction = fr.metrics, fr.quality, fr.direction
        state = self.cycles.state
        state_color = {
            MovementState.FECHADO: C_TEXTO,
            MovementState.ABRINDO: C_OK,
            MovementState.ABERTO: C_OK,
            MovementState.FECHANDO: C_ALERTA,
        }[state]

        lines: list[tuple[str, tuple]] = []

        if m is None:
            lines.append(("Face nao detectada", C_ALERTA))
        else:
            if m.opening_mm is not None:
                lines.append((f"Abertura: {m.opening_rel:.3f} rel  ({m.opening_mm:.1f} mm)", C_TEXTO))
                lines.append((f"Desvio lat.: {m.lateral_rel:+.3f} rel  ({m.lateral_mm:+.1f} mm)  [{direction}]", C_TEXTO))
            else:
                lines.append((f"Abertura: {m.opening_rel:.3f} (rel. larg. facial)", C_TEXTO))
                lines.append((f"Desvio lateral: {m.lateral_rel:+.3f}  [{direction}]", C_TEXTO))
            if not fr.frame_valid:
                lines.append((
                    "Frame invalido - exibindo ultimo valor valido (NAO e nova medicao)",
                    C_ALERTA,
                ))
            # Depuracao: razao de tamanho facial atual, limite minimo e motivo exato.
            if quality.ratio is not None:
                lines.append((
                    f"Razao facial: {quality.ratio:.3f}  (min {quality.min_ratio:.2f} / max {quality.max_ratio:.2f})",
                    C_TEXTO,
                ))
            if quality.message:
                lines.append((f"{quality.message}", C_ALERTA))

        lines.append((f"Estado: {state.value.upper()}", state_color))
        if self.cycles.is_calibrated:
            lines.append((f"Repeticoes: {self.cycles.repetitions}", C_TEXTO))
        else:
            lines.append((
                f"Repeticoes (NAO calibrado, faixa dinamica): {self.cycles.repetitions}",
                C_ALERTA,
            ))
        lines.append((
            "Calibrado: sim" if self.cycles.is_calibrated else "Calibrado: nao (tecle C)",
            C_OK if self.cycles.is_calibrated else C_ALERTA,
        ))

        for msg in feedback:
            lines.append((msg, C_ALERTA))

        rec_bits = []
        rec_bits.append("REC dados" if self.recording else None)
        rec_bits.append("REC video" if self.video_recording else None)
        rec_txt = " | ".join(b for b in rec_bits if b)
        lines.append((
            rec_txt if rec_txt else "[C]alibrar [R]ec [V]ideo [E]xport [Z]erar [Q]sair",
            C_REC if rec_txt else C_TEXTO,
        ))

        if self.calib.active:
            lines = [(self.calib.instruction(now), C_ALERTA)] + lines
        if self.last_status:
            lines.append((self.last_status, C_OK))

        draw_panel(frame, lines)

    # -- Acoes ------------------------------------------------------------
    def _toggle_video(self, frame, cam_fps: float) -> None:
        if self.video_recording:
            if self.video_writer is not None:
                self.video_writer.close()
                self.video_writer = None
            self.video_recording = False
            self.last_status = "Gravacao de video pausada."
            return

        self._ensure_session()
        h, w = frame.shape[:2]
        video_path = os.path.join(self.session_dir, "video.mp4")
        try:
            self.video_writer = VideoRecorder(video_path, cam_fps, (w, h))
            self.video_recording = True
            self.last_status = "Gravacao de video iniciada."
        except RuntimeError as exc:
            self.last_status = f"Falha ao iniciar video: {exc}"

    def _export(self) -> None:
        if self.recorder.is_empty:
            self.last_status = "Nada gravado para exportar (tecle R primeiro)."
            return
        self._ensure_session()

        video_path = None
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
            self.video_recording = False
            video_path = os.path.join(self.session_dir, "video.mp4")

        try:
            paths = export_session(
                self.recorder,
                self.cycles,
                self.session_dir,
                self.session_id,
                ref_mm=self.cfg.reference_distance_mm,
                video_path=video_path,
                extra_metadata={
                    "camera_index": self.cfg.camera_index,
                    "resolucao": [self.cfg.frame_width, self.cfg.frame_height],
                    "espelhado": self.cfg.flip_horizontal,
                },
            )
        except ValueError as exc:
            self.last_status = str(exc)
            return

        rep = self.cycles.repeatability()
        rep_txt = ""
        if rep:
            rep_txt = (
                f" | ciclos={int(rep['n_ciclos'])}"
                f" ampl.CV={rep['amplitude_cv']:.2f}"
                f" dur.CV={rep['duracao_cv']:.2f}"
            )
        self.last_status = f"Exportado: {self.session_id}{rep_txt}"
        print(f"[export] pasta: {self.session_dir}")
        for k, v in paths.items():
            print(f"[export] {k}: {v}")
        if rep:
            print(f"[export] repetibilidade: {rep}")

    def _reset(self) -> None:
        self.recorder.clear()
        self.cycles = CycleDetector(self.cfg.cycle)
        self.filt_opening.reset()
        self.filt_lateral.reset()
        self.filt_face_width.reset()
        self.frame_idx = 0
        self.t0 = time.perf_counter()
        self._last_ts_ms = -1
        self._prev_nasion = None
        self.recording = False
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        self.video_recording = False
        self.session_id = None
        self.session_dir = None
        self.last_status = "Sessao zerada."
