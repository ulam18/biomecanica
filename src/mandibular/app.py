"""
Aplicacao em tempo real: captura por webcam, deteccao facial, filtragem,
controle de qualidade, calculo das metricas, deteccao de ciclos, biofeedback,
interface visual e exportacao de dados (CSV, JSON, graficos e video opcional).

A interface pode ser operada de duas formas equivalentes:
    - clicando nos botoes da faixa inferior da janela (padrao no modo simples);
    - pelo teclado, para quem prefere atalhos (modo tecnico).

Controles do teclado:
    C  - calibrar (assistente: boca fechada -> boca aberta)
    X  - apagar a calibracao atual
    V  - habilita/desabilita a gravacao de video para a PROXIMA sessao
    R  - inicia/encerra uma sessao (dados + video, SINCRONIZADOS)
    E  - exporta a sessao encerrada (CSV, resumo, metadados, graficos, video)
    Z  - zera a sessao atual (preserva a calibracao)
    Q / ESC - sair (finaliza com seguranca uma sessao ainda ativa)

Sincronizacao R/vídeo (ver secao 11 do escopo): pressionar R gera um
session_id, zera o contador local de frames/tempo e os ciclos (preservando a
calibracao), e -- se V estiver habilitado -- abre o VideoRecorder no MESMO
instante em que a gravacao de dados comeca. R novamente encerra os dois ao
mesmo tempo. O timestamp entregue ao MediaPipe (`_last_ts_ms`/`t0`) e global
e NUNCA e resetado por R/Z: resetar o "relogio" que o FaceLandmarker usa
(modo VIDEO exige timestamps estritamente crescentes) travaria a deteccao.

No modo simples (`cfg.simple_ui`, usado pelo launcher/menu clinico), os
botoes da faixa inferior chamam essas mesmas acoes (R="gravar", V="video",
E="exportar", C="calibrar", Z="zerar"), e o HUD tecnico e substituido por um
painel em linguagem direta, sem numeros de depuracao.
"""

from __future__ import annotations

import os
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .calibration import CalibrationAssistant
from .config import AppConfig, Landmark
from .exporter import export_session, make_session_id
from .feedback import biofeedback_messages
from .filters import EMAFilter
from .landmarks import FaceMeshDetector
from .metrics import CycleDetector, MovementState, lateral_direction
from .overlay import (
    BARRA_ALTURA,
    C_ALERTA,
    C_HEADER,
    C_OK,
    C_REC,
    C_TEXTO,
    ButtonBar,
    draw_landmarks,
    draw_opening_bar,
    draw_panel,
)
from .pipeline import process_frame
from .recorder import Sample, SessionRecorder
from .video_recorder import VideoRecorder

# Botoes da faixa inferior: (acao, rotulo). Os rotulos evitam jargao tecnico
# porque a aplicacao tambem e usada por profissional de saude sem familiaridade
# com o codigo ou com atalhos de teclado. As acoes casam com as teclas C/R/V/E/Z.
BOTOES = [
    ("calibrar", "1. CALIBRAR"),
    ("gravar", "2. GRAVAR"),
    ("exportar", "3. SALVAR"),
    ("zerar", "RECOMECAR"),
    ("sair", "SAIR"),
]

# Avisos de qualidade sem numeros de depuracao, para o painel do modo simples.
# A chave e o `reason` do QualityResult (quality.py), nao a mensagem tecnica.
AVISOS_SIMPLES = {
    "sem_face": "Rosto nao encontrado - centralize o rosto",
    "multiplas_faces": "Certifique-se de que so uma pessoa aparece na camera",
    "fora_da_imagem": "Centralize o rosto na imagem",
    "muito_longe": "Aproxime o paciente da camera",
    "muito_perto": "Afaste um pouco o paciente da camera",
    "inclinacao_excessiva": "Peca para manter a cabeca reta",
    "yaw_excessivo": "Peca para o paciente olhar de frente para a camera",
    "pitch_excessivo": "Peca para o paciente nivelar a cabeca (nem para cima, nem para baixo)",
    "movimento_brusco": "Peca para manter a cabeca parada",
}


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

        # -- Estado da sessao (R) ---------------------------------------
        self.recording = False
        self.video_enabled_next = False   # V: vale para a PROXIMA sessao iniciada
        self.video_enabled = False        # decisao congelada da sessao atual/ultima
        self.video_recording = False      # video_writer ativo nesta sessao
        self.video_writer: VideoRecorder | None = None
        self.session_id: str | None = None
        self.session_dir: str | None = None
        self._session_frame_idx = 0
        self._session_start_t: float | None = None      # em segundos, mesmo relogio de `t`
        self._session_start_wall: datetime | None = None
        self._session_end_wall: datetime | None = None
        self._session_duration_s: float | None = None
        # Capturados no encerramento do video (R de parada), antes de fechar
        # o VideoRecorder, para irem para os metadados na exportacao (E).
        self._video_frame_count = 0
        self._video_fps = 0.0
        self._video_duration_s = 0.0

        # -- Relogio global (NUNCA resetado por R/Z: o FaceLandmarker em modo
        # VIDEO exige timestamps estritamente crescentes durante toda a vida
        # do detector) --------------------------------------------------
        self.frame_idx = 0
        self.t0 = time.perf_counter()
        self._last_ts_ms = -1
        self._prev_nasion: np.ndarray | None = None
        self.last_status = ""

        self._fps_filter = EMAFilter(alpha=0.1)
        self._last_frame_time: float | None = None
        self._last_frame_shape: tuple[int, int] | None = None

        # -- Interface por botoes/mouse (modo simples) -----------------------
        self.buttons = ButtonBar(BOTOES)
        self._clique: str | None = None   # acao pendente vinda do mouse
        self.last_report: str | None = None

    # -- Sessao ---------------------------------------------------------
    def _start_session(self, t: float) -> None:
        self.session_id = make_session_id()
        self.session_dir = os.path.join(self.cfg.output_dir, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)

        self.recorder.clear()
        self.cycles.reset_session()  # preserva calibracao; zera ciclos/estado
        self._session_frame_idx = 0
        self._session_start_t = t
        self._session_start_wall = datetime.now()
        self._session_end_wall = None
        self._session_duration_s = None

        self.video_enabled = self.video_enabled_next
        self.video_writer = None
        self.video_recording = False
        if self.video_enabled:
            h, w = self._last_frame_shape or (self.cfg.frame_height, self.cfg.frame_width)
            video_path = os.path.join(self.session_dir, "video.mp4")
            try:
                # FPS de CODIFICACAO (nao o fps real de processamento do
                # pipeline -- cap.get(CAP_PROP_FPS) so descreve a captura da
                # webcam, nunca a velocidade de inferencia; ver write_paced).
                self.video_writer = VideoRecorder(
                    video_path, self.cfg.video_output_fps, (w, h)
                )
                self.video_recording = True
            except RuntimeError as exc:
                self.last_status = f"Falha ao iniciar video: {exc}; sessao segue sem video."

        self.recording = True
        self.last_status = (
            "Sessao iniciada (dados + video)." if self.video_recording
            else "Sessao iniciada (somente dados)."
        )

    def _stop_session(self, t: float, frame=None) -> None:
        self.recording = False
        self._session_end_wall = datetime.now()
        self._session_duration_s = (
            t - self._session_start_t if self._session_start_t is not None else 0.0
        )

        if self.video_writer is not None:
            if frame is not None:
                # Preenche ate o ultimo indice correspondente ao fim real da
                # sessao (o ultimo frame recebido "vale" ate esse instante).
                target = int(self._session_duration_s * self.video_writer.fps)
                self.video_writer.write_paced(frame, target)
            self._video_frame_count = self.video_writer.frame_count
            self._video_fps = self.video_writer.fps
            self._video_duration_s = self.video_writer.duration_s
            self.video_writer.close()
            self.video_writer = None
        else:
            self._video_frame_count = 0
            self._video_fps = 0.0
            self._video_duration_s = 0.0
        self.video_recording = False

        self.last_status = (
            "Sessao encerrada. Salve para gerar o relatorio."
            if self.cfg.simple_ui
            else "Sessao encerrada. Tecle E para exportar."
        )

    def _clear_session(self, t: float, frame=None) -> None:
        """Zera a sessao atual (amostras, video, contadores), preservando a calibracao."""
        if self.recording:
            self._stop_session(t, frame)
        self.recorder.clear()
        self.cycles.reset_session()
        self._session_frame_idx = 0
        self._session_start_t = None
        self._session_start_wall = None
        self._session_end_wall = None
        self._session_duration_s = None
        self.session_id = None
        self.session_dir = None
        self._video_frame_count = 0
        self._video_fps = 0.0
        self._video_duration_s = 0.0
        self.last_status = "Sessao zerada (calibracao preservada)."

    def _clear_calibration(self) -> None:
        self.cycles.clear_calibration()
        self.last_status = "Calibracao removida. Calibre novamente antes de gravar."

    # -- Loop principal -------------------------------------------------------
    def run(self) -> None:
        cap = cv2.VideoCapture(self.cfg.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.frame_height)

        if not cap.isOpened():
            raise RuntimeError(
                f"Nao foi possivel abrir a camera (indice {self.cfg.camera_index})."
            )

        win = "Reconhecimento Mandibular - Biomecanica (frontal)"
        # WINDOW_AUTOSIZE no modo simples: a janela nao e redimensionavel, o que
        # garante que a coordenada do clique corresponda ao pixel da imagem
        # (necessario para acertar os botoes do rodape).
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE if self.cfg.simple_ui else cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, self._on_mouse)

        now = time.perf_counter()  # garante que 'now'/'frame' existam mesmo se o loop nao rodar
        frame = None
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if self.cfg.flip_horizontal:
                    frame = cv2.flip(frame, 1)
                self._last_frame_shape = frame.shape[:2]

                now = time.perf_counter()
                if self._last_frame_time is not None:
                    dt = now - self._last_frame_time
                    if dt > 1e-6:
                        self._fps_filter.update(1.0 / dt)
                self._last_frame_time = now

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
                    lateral_baseline=self.cycles.lateral_baseline,
                )
                self._prev_nasion = face.point(Landmark.NASION) if face is not None else None

                if face is not None:
                    draw_landmarks(frame, face)

                if self.calib.active and fr.frame_valid:
                    result = self.calib.update(fr.opening_filtered, now, lateral=fr.lateral_filtered)
                    if result is not None:
                        if result.valid:
                            self.cycles.calibrate(
                                result.closed, result.opened,
                                result.lateral_baseline, result.lateral_baseline_std,
                            )
                        self.last_status = result.message

                t_session = (
                    t - self._session_start_t if self._session_start_t is not None else 0.0
                )
                if self.recording and not self.calib.active and fr.frame_valid:
                    self.cycles.update(
                        fr.opening_filtered, t_session,
                        lateral_absolute=fr.lateral_filtered,
                        lateral_dynamic=fr.lateral_dynamic_filtered,
                    )

                feedback = (
                    []
                    if self.calib.active
                    else biofeedback_messages(fr.opening_display, fr.direction, self.cycles, fr.quality)
                )

                if face is not None:
                    draw_opening_bar(
                        frame, fr.opening_display, self.cycles,
                        margem_inferior=BARRA_ALTURA if self.cfg.show_buttons else 0,
                    )
                self._draw_hud(frame, fr, feedback, now)

                if self.recording:
                    m = fr.metrics
                    q = fr.quality
                    self.recorder.add(
                        Sample(
                            session_id=self.session_id,
                            frame=self._session_frame_idx,
                            timestamp=datetime.now().isoformat(timespec="milliseconds"),
                            time_s=t_session,
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
                            quality_warning=q.message,
                            quality_reason=q.reason,
                            face_size_ratio=q.ratio,
                            roll_deg=q.roll_deg,
                            yaw_deg=q.yaw_deg,
                            pitch_deg=q.pitch_deg,
                            lateral_neutral_baseline=fr.lateral_baseline,
                            lateral_dynamic_raw=fr.lateral_dynamic_raw,
                            lateral_dynamic_filtered=fr.lateral_dynamic_filtered,
                            symmetry_index=(fr.symmetry.index if fr.symmetry else None),
                            midline_offset_rel=(
                                fr.symmetry.midline_offset_rel if fr.symmetry else None
                            ),
                            cant_deg=(fr.symmetry.cant_deg if fr.symmetry else None),
                            mand_deviation_deg=(
                                fr.angles.mand_deviation_deg if fr.angles else None
                            ),
                            is_frontal=(fr.symmetry.is_frontal if fr.symmetry else None),
                            biofeedback=(" | ".join(feedback) if feedback else None),
                        )
                    )
                    self._session_frame_idx += 1

                    if self.video_recording and self.video_writer is not None:
                        # Pacing pelo tempo real da sessao (nao 1 frame de
                        # video por frame processado): ver VideoRecorder.write_paced.
                        target_frame_index = int(t_session * self.video_writer.fps)
                        self.video_writer.write_paced(frame, target_frame_index)

                if self.cfg.show_buttons:
                    self.buttons.draw(frame, self._button_states())

                self.frame_idx += 1
                cv2.imshow(win, frame)

                key = cv2.waitKey(1) & 0xFF
                acao = self._clique or self._action_for_key(key)
                self._clique = None
                if acao == "sair":
                    break
                if acao is not None:
                    self._handle_action(acao, t, frame)
        finally:
            # Finalizacao segura: nao perder silenciosamente uma sessao ativa.
            if self.recording:
                self._stop_session(now - self.t0, frame)
            if not self.recorder.is_empty:
                print("[aviso] finalizando e exportando a sessao ativa antes de sair.")
                self._export()

            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()
            if self.video_writer is not None:
                self.video_writer.close()
                self.video_writer = None

    # -- Acoes: mouse e teclado convergem aqui --------------------------------
    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and self.cfg.show_buttons:
            self._clique = self.buttons.hit(x, y)

    @staticmethod
    def _action_for_key(key: int) -> str | None:
        return {
            ord("q"): "sair", 27: "sair",
            ord("c"): "calibrar",
            ord("x"): "apagar_calibracao",
            ord("r"): "gravar",
            ord("v"): "video",
            ord("e"): "exportar",
            ord("z"): "zerar",
        }.get(key)

    def _handle_action(self, acao: str, t: float, frame) -> None:
        if acao == "calibrar":
            self.calib.start()
            self.last_status = "Iniciando calibracao..."
        elif acao == "apagar_calibracao":
            self._clear_calibration()
        elif acao == "gravar":
            if not self.recording:
                self._start_session(t)
            else:
                self._stop_session(t, frame)
        elif acao == "video":
            self.video_enabled_next = not self.video_enabled_next
            if self.cfg.simple_ui:
                self.last_status = (
                    "Video sera gravado na proxima sessao."
                    if self.video_enabled_next
                    else "Video NAO sera gravado na proxima sessao."
                )
            else:
                self.last_status = (
                    "Video HABILITADO para a proxima sessao."
                    if self.video_enabled_next
                    else "Video DESABILITADO para a proxima sessao."
                )
        elif acao == "exportar":
            self._export()
        elif acao == "zerar":
            self._clear_session(t, frame)

    def _button_states(self) -> dict[str, str]:
        """Sinaliza a gravacao em curso e sugere o proximo passo ao usuario."""
        estados: dict[str, str] = {}
        if self.recording:
            estados["gravar"] = "ativo"
        elif not self.cycles.is_calibrated:
            estados["calibrar"] = "principal"
        else:
            estados["gravar"] = "principal"
        if not self.recording and not self.recorder.is_empty:
            estados["exportar"] = "principal"
        return estados

    # -- HUD ------------------------------------------------------------
    def _draw_hud(self, frame, fr, feedback: list[str], now: float) -> None:
        if self.cfg.simple_ui:
            self._draw_hud_simple(frame, fr, feedback, now)
        else:
            self._draw_hud_tecnico(frame, fr, feedback, now)

    def _draw_hud_simple(self, frame, fr, feedback: list[str], now: float) -> None:
        """
        Painel em linguagem direta, sem jargao nem numeros de depuracao, para
        uso clinico. Os dados completos continuam indo para o CSV e o relatorio.
        """
        m, quality = fr.metrics, fr.quality
        lines: list[tuple[str, tuple]] = []

        if self.calib.active:
            lines.append((self.calib.instruction(now), C_ALERTA))
            draw_panel(frame, lines)
            return

        if m is None:
            lines.append((AVISOS_SIMPLES["sem_face"], C_ALERTA))
        else:
            if m.opening_mm is not None:
                lines.append((f"Abertura da boca: {m.opening_mm:.0f} mm", C_TEXTO))
            else:
                lines.append((f"Abertura da boca: {m.opening_rel:.2f}", C_TEXTO))
            lado = {"direita": "para a direita", "esquerda": "para a esquerda"}.get(
                fr.direction, "centralizado"
            )
            lines.append((f"Queixo: {lado}", C_TEXTO))
            if quality.message:
                lines.append((
                    AVISOS_SIMPLES.get(quality.reason, quality.message), C_ALERTA,
                ))

        lines.append((f"Repeticoes: {self.cycles.repetitions}", C_TEXTO))

        if not self.cycles.is_calibrated:
            lines.append(("Passo 1: clique em CALIBRAR", C_ALERTA))
        elif not self.recording:
            lines.append(("Passo 2: clique em GRAVAR", C_OK))
        else:
            lines.append(("GRAVANDO - clique em GRAVAR de novo para parar", C_REC))

        for msg in feedback:
            lines.append((msg, C_ALERTA))
        if self.last_status:
            lines.append((self.last_status, C_OK))

        draw_panel(frame, lines)

    def _draw_hud_tecnico(self, frame, fr, feedback: list[str], now: float) -> None:
        """
        Painel organizado em secoes (ABERTURA / LATERALIDADE / QUALIDADE /
        SESSAO / CONTROLES), uma informacao por linha -- nunca varios valores
        concatenados numa linha so, para nao cortar texto. Nomes exibidos ao
        usuario (secao 1 do refinamento):
            "Lateral absoluto"  -> "Posicao do queixo em relacao a linha media"
            "Lateral dinamico"  -> "Movimento do queixo desde a posicao neutra"
            "Baseline lateral"  -> "Posicao neutra calibrada"
        (nomes internos/colunas do CSV continuam os mesmos, por compatibilidade.)
        """
        m, quality = fr.metrics, fr.quality
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
            lines.append(("ABERTURA", C_HEADER))
            if m.opening_mm is not None:
                lines.append((f"Abertura: {m.opening_rel:.3f} rel  ({m.opening_mm:.1f} mm)", C_TEXTO))
            else:
                lines.append((f"Abertura: {m.opening_rel:.3f}", C_TEXTO))
            lines.append((f"Estado: {state.value.upper()}", state_color))
            if self.cycles.is_calibrated:
                lines.append((f"Repeticoes: {self.cycles.repetitions}", C_TEXTO))
            else:
                lines.append((f"Repeticoes (nao calibrado): {self.cycles.repetitions}", C_ALERTA))
            lines.append(("", C_TEXTO))

            lines.append(("LATERALIDADE", C_HEADER))
            side = lateral_direction(fr.lateral_display, self.cfg.flip_horizontal)
            lines.append((f"Posicao do queixo: {fr.lateral_display:+.3f}", C_TEXTO))
            lines.append((f"Lado da linha media: {side}", C_TEXTO))
            if fr.lateral_baseline is not None:
                lines.append((f"Movimento desde o neutro: {fr.lateral_dynamic_display:+.3f}", C_TEXTO))
                lines.append((f"Direcao do movimento: {fr.direction}", C_TEXTO))
                lines.append((f"Posicao neutra calibrada: {fr.lateral_baseline:+.3f}", C_TEXTO))
            else:
                lines.append(("Posicao neutra calibrada: nao calibrada (tecle C)", C_ALERTA))
            lines.append(("", C_TEXTO))

            if not fr.frame_valid:
                lines.append((
                    "Frame invalido - exibindo ultimo valor valido (NAO e nova medicao)",
                    C_ALERTA,
                ))

            lines.append(("QUALIDADE", C_HEADER))
            lines.append((
                f"Roll: {quality.roll_deg:+.0f} graus" if quality.roll_deg is not None
                else "Roll: indisponivel",
                C_TEXTO,
            ))
            lines.append((
                f"Yaw: {quality.yaw_deg:+.0f} graus" if quality.yaw_deg is not None
                else "Yaw: indisponivel",
                C_TEXTO,
            ))
            lines.append((
                f"Pitch: {quality.pitch_deg:+.0f} graus" if quality.pitch_deg is not None
                else "Pitch: indisponivel",
                C_TEXTO,
            ))
            if quality.ratio is not None:
                lines.append((
                    f"Razao facial: {quality.ratio:.3f} (min {quality.min_ratio:.2f} / max {quality.max_ratio:.2f})",
                    C_TEXTO,
                ))
            lines.append((
                "Frame: valido" if fr.frame_valid else f"Frame: invalido ({quality.reason or '?'})",
                C_OK if fr.frame_valid else C_ALERTA,
            ))
            if quality.message:
                lines.append((quality.message, C_ALERTA))
            fps = self._fps_filter.value or 0.0
            lines.append((f"FPS: {fps:.0f}", C_TEXTO))
            lines.append(("", C_TEXTO))

        lines.append(("SESSAO", C_HEADER))
        lines.append((
            "Calibrado: sim" if self.cycles.is_calibrated else "Calibrado: nao (tecle C)",
            C_OK if self.cycles.is_calibrated else C_ALERTA,
        ))
        lines.append((
            f"Sessao: {'gravando' if self.recording else 'parada'}",
            C_REC if self.recording else C_TEXTO,
        ))
        lines.append((
            f"Video proxima sessao: {'ON' if self.video_enabled_next else 'OFF'}", C_TEXTO
        ))
        rec_bits = []
        rec_bits.append("REC dados" if self.recording else None)
        rec_bits.append("REC video" if self.video_recording else None)
        rec_txt = " | ".join(b for b in rec_bits if b)
        if rec_txt:
            lines.append((rec_txt, C_REC))
        lines.append(("", C_TEXTO))

        for msg in feedback:
            lines.append((msg, C_ALERTA))
        if feedback:
            lines.append(("", C_TEXTO))

        lines.append(("CONTROLES", C_HEADER))
        lines.append(("C calibrar | X apagar calibracao", C_TEXTO))
        lines.append(("V video | R iniciar/parar", C_TEXTO))
        lines.append(("E exportar | Z zerar | Q sair", C_TEXTO))

        if self.calib.active:
            lines = [(self.calib.instruction(now), C_ALERTA), ("", C_TEXTO)] + lines
        if self.last_status:
            lines.append(("", C_TEXTO))
            lines.append((self.last_status, C_OK))

        draw_panel(frame, lines)

    # -- Acoes ------------------------------------------------------------
    def _export(self) -> None:
        if self.recording:
            self.last_status = (
                "Pare a sessao antes de salvar (clique em GRAVAR de novo)."
                if self.cfg.simple_ui
                else "Pare a sessao (tecle R) antes de exportar."
            )
            return
        if self.recorder.is_empty:
            self.last_status = (
                "Nada para salvar - clique em GRAVAR primeiro."
                if self.cfg.simple_ui
                else "Nada gravado para exportar (tecle R primeiro)."
            )
            return
        self._ensure_session_dir()

        video_path = None
        if self._video_frame_count > 0:
            candidate = os.path.join(self.session_dir, "video.mp4")
            if os.path.exists(candidate):
                video_path = candidate

        processed_samples = self._session_frame_idx
        session_duration_s = self._session_duration_s or 0.0
        processing_fps_medio = (
            processed_samples / session_duration_s if session_duration_s > 1e-6 else 0.0
        )
        # video_duration_s = video_frames_written / video_output_fps (nao
        # cap.get(CAP_PROP_FPS): esse so descreve a captura da webcam, nunca a
        # velocidade real de inferencia do pipeline).
        video_duration_s = self._video_duration_s
        video_session_duration_difference_s = abs(video_duration_s - session_duration_s)

        try:
            paths = export_session(
                self.recorder,
                self.cycles,
                self.session_dir,
                self.session_id,
                ref_mm=self.cfg.reference_distance_mm,
                video_path=video_path,
                mirrored=self.cfg.flip_horizontal,
                extra_metadata={
                    "camera_index": self.cfg.camera_index,
                    "resolucao": [self.cfg.frame_width, self.cfg.frame_height],
                    "espelhado": self.cfg.flip_horizontal,
                    "quality_thresholds": {
                        "min_face_width_ratio": self.cfg.quality.min_face_width_ratio,
                        "max_face_width_ratio": self.cfg.quality.max_face_width_ratio,
                        "max_roll_deg": self.cfg.quality.max_roll_deg,
                        "max_yaw_deg": self.cfg.quality.max_yaw_deg,
                        "max_pitch_deg": self.cfg.quality.max_pitch_deg,
                        "max_global_jump_fraction": self.cfg.quality.max_global_jump_fraction,
                        "reject_multiple_faces": self.cfg.quality.reject_multiple_faces,
                    },
                    "inicio_sessao": (
                        self._session_start_wall.isoformat(timespec="seconds")
                        if self._session_start_wall else None
                    ),
                    "fim_sessao": (
                        self._session_end_wall.isoformat(timespec="seconds")
                        if self._session_end_wall else None
                    ),
                    "repeticoes_sessao": self.cycles.repetitions,
                    "video_habilitado": self.video_enabled,
                    "processed_samples": processed_samples,
                    "processing_fps_medio": round(processing_fps_medio, 3),
                    "video_output_fps": self._video_fps or self.cfg.video_output_fps,
                    "video_frames_written": self._video_frame_count,
                    "video_duration_s": round(video_duration_s, 4),
                    "session_duration_s": round(session_duration_s, 4),
                    "video_session_duration_difference_s": round(
                        video_session_duration_difference_s, 4
                    ),
                },
                paciente=self.cfg.patient,
                history_dir=self.cfg.output_dir,
            )
        except ValueError as exc:
            self.last_status = str(exc)
            return

        # Prefere o PDF: e o documento que vai para o prontuario. O HTML fica
        # na pasta como alternativa navegavel.
        self.last_report = paths.get("relatorio_pdf") or paths.get("relatorio")
        if self.cfg.open_report and self.last_report:
            webbrowser.open(Path(self.last_report).resolve().as_uri())

        rep = self.cycles.repeatability()
        rep_txt = ""
        if rep:
            rep_txt = (
                f" | ciclos={int(rep['n_ciclos'])}"
                f" ampl.CV={rep['amplitude_cv']:.2f}"
                f" dur.CV={rep['duracao_cv']:.2f}"
            )
        if not self.cfg.simple_ui:
            self.last_status = f"Exportado: {self.session_id}{rep_txt}"
        elif self.cfg.open_report:
            self.last_status = "Sessao salva! O relatorio em PDF foi aberto."
        else:
            self.last_status = "Sessao salva na pasta de resultados."
        print(f"[export] pasta: {self.session_dir}")
        for k, v in paths.items():
            print(f"[export] {k}: {v}")
        if rep:
            print(f"[export] repetibilidade: {rep}")
        if video_path is not None:
            print(
                f"[export] processamento~{processing_fps_medio:.1f}fps "
                f"({processed_samples} amostras / {session_duration_s:.2f}s)  |  "
                f"video={video_duration_s:.2f}s @ {self._video_fps:.0f}fps "
                f"(diff={video_session_duration_difference_s:.3f}s)"
            )

        if self.cfg.simple_ui:
            # No uso clinico, SALVAR encerra o ciclo: a proxima sessao comeca
            # do zero (nova pasta), mas a calibracao do paciente e preservada
            # -- ela ja foi zerada de amostras/ciclos por _clear_session, so
            # falta soltar o session_id/dir para o proximo _start_session
            # criar uma pasta nova em vez de reaproveitar esta. _clear_session
            # sobrescreveria a mensagem de sucesso com "sessao zerada"; guarda
            # e restaura para o profissional ver a confirmacao do salvamento.
            status_salvo = self.last_status
            self._clear_session(time.perf_counter() - self.t0)
            self.last_status = status_salvo

    def _ensure_session_dir(self) -> None:
        if self.session_id is None:
            self.session_id = make_session_id()
        if self.session_dir is None:
            self.session_dir = os.path.join(self.cfg.output_dir, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
