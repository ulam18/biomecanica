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
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, max(fps, 1.0), frame_size)
        if not self._writer.isOpened():
            raise RuntimeError(f"Nao foi possivel criar o arquivo de video: {path}")

    def write(self, frame) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
