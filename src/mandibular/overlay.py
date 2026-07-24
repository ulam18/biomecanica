"""
Funcoes de desenho da interface (landmarks, painel HUD, barra de biofeedback).

Extraidas para um modulo proprio para serem reutilizadas tanto no modo ao
vivo (app.py) quanto na geracao de video anotado na analise offline
(analyze_video.py), evitando duplicar a logica de desenho.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import HIGHLIGHT_POINTS, Landmark
from .metrics import CycleDetector

# Cores (BGR).
C_PONTO = (0, 255, 0)
C_LINHA = (255, 200, 0)
C_TEXTO = (255, 255, 255)
C_PAINEL = (30, 30, 30)
C_REC = (0, 0, 255)
C_OK = (0, 220, 0)
C_ALERTA = (0, 180, 255)


def draw_landmarks(frame: np.ndarray, face) -> None:
    """Desenha a linha media, o eixo inter-ocular, a linha dos labios e os pontos destacados."""
    for a, b in [
        (Landmark.NASION, Landmark.CHIN),
        (Landmark.EYE_OUTER_LEFT, Landmark.EYE_OUTER_RIGHT),
        (Landmark.UPPER_LIP_INNER, Landmark.LOWER_LIP_INNER),
    ]:
        pa = tuple(np.round(face.point(a)).astype(int))
        pb = tuple(np.round(face.point(b)).astype(int))
        cv2.line(frame, pa, pb, C_LINHA, 1, cv2.LINE_AA)

    for idx in HIGHLIGHT_POINTS:
        p = tuple(np.round(face.point(idx)).astype(int))
        cv2.circle(frame, p, 3, C_PONTO, -1, cv2.LINE_AA)


def draw_panel(frame: np.ndarray, lines: list[tuple[str, tuple]]) -> None:
    """Desenha um painel semi-transparente com uma linha de texto por item."""
    pad = 12
    line_h = 26
    w = 380
    h = pad * 2 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + w, 10 + h), C_PAINEL, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    y = 10 + pad + 18
    for text, color in lines:
        cv2.putText(frame, text, (10 + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 1, cv2.LINE_AA)
        y += line_h


def draw_opening_bar(frame: np.ndarray, opening_rel: float, cycles: CycleDetector) -> None:
    """Barra de biofeedback da abertura (0..faixa calibrada, ou 0..0.6 sem calibracao)."""
    h, w = frame.shape[:2]
    x0, y0 = w - 60, 60
    bar_h = h - 120
    cv2.rectangle(frame, (x0, y0), (x0 + 30, y0 + bar_h), (80, 80, 80), 1)

    if cycles.is_calibrated and cycles.baseline is not None and cycles.span:
        frac = (opening_rel - cycles.baseline) / cycles.span
    else:
        frac = opening_rel / 0.6
    frac = float(np.clip(frac, 0.0, 1.0))

    fill = int(bar_h * frac)
    cv2.rectangle(frame, (x0, y0 + bar_h - fill), (x0 + 30, y0 + bar_h), C_OK, -1)
    cv2.putText(frame, "abertura", (x0 - 20, y0 - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXTO, 1, cv2.LINE_AA)
