"""
Gravacao opcional do video anotado (landmarks, valores, estado, biofeedback).

Wrapper fino sobre cv2.VideoWriter, garantindo que o arquivo seja liberado
corretamente mesmo se a sessao terminar com erro.
"""

from __future__ import annotations

import os

import cv2


class VideoRecorder:
    """Grava frames anotados em um arquivo de video (mp4, codec mp4v)."""

    def __init__(self, path: str, fps: float, frame_size: tuple[int, int]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.fps = max(fps, 1.0)
        self.frame_count = 0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, self.fps, frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Nao foi possivel criar o arquivo de video: {path}")

    def write(self, frame) -> None:
        self._writer.write(frame)
        self.frame_count += 1

    def write_paced(self, frame, target_frame_index: int) -> None:
        """
        Escreve `frame` (repetindo-o quantas vezes for preciso) ate que
        `frame_count` alcance `target_frame_index`.

        O pipeline (deteccao + MediaPipe) raramente processa a `self.fps`
        (30 fps) de saida do video -- normalmente processa MENOS. Escrever um
        frame de video por frame processado faz o MP4 tocar mais rapido que a
        sessao real (ver bug: video 1.77x mais rapido). Pacing pelo tempo real
        decorrido (`target_frame_index = floor(tempo_decorrido * fps_saida)`)
        garante que a duracao do video acompanhe a duracao real da sessao,
        independente da velocidade do pipeline:
          - pipeline mais LENTO que `self.fps`: o alvo avancou varios indices
            desde a ultima escrita -> este frame e duplicado para preencher o
            intervalo (o CSV nao duplica nada, so o video).
          - pipeline mais RAPIDO que `self.fps`: o alvo ainda nao avancou ->
            nada e escrito agora (o frame anterior "vale" por mais tempo).
        """
        while self.frame_count < target_frame_index:
            self.write(frame)

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
