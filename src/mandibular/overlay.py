"""
Funcoes de desenho da interface (landmarks, painel HUD, barra de biofeedback,
botoes clicaveis).

Extraidas para um modulo proprio para serem reutilizadas tanto no modo ao
vivo (app.py) quanto na geracao de video anotado na analise offline
(analyze_video.py), evitando duplicar a logica de desenho.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
C_HEADER = (0, 200, 200)  # titulo de secao do HUD (ex.: "ABERTURA", "LATERALIDADE")

C_BOTAO = (70, 70, 70)
C_BOTAO_ATIVO = (40, 40, 190)
C_BOTAO_DESTAQUE = (40, 130, 40)
C_BOTAO_BORDA = (200, 200, 200)

BARRA_ALTURA = 64  # altura da faixa de botoes no rodape


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
    """
    Desenha um painel semi-transparente com uma linha de texto por item
    (uma informacao por linha -- nunca concatenar varios valores numa
    linha longa, para o texto nao ficar cortado). Linhas vazias ("") viram
    um pequeno espaco em branco, usadas para separar secoes do HUD.
    """
    pad = 12
    line_h = 24
    blank_h = 10
    w = 460  # largo o suficiente para as linhas mais compridas (ex.: qualidade) nao cortarem
    h = pad * 2 + sum(blank_h if text == "" else line_h for text, _ in lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + w, 10 + h), C_PAINEL, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    y = 10 + pad + 16
    for text, color in lines:
        if text == "":
            y += blank_h
            continue
        cv2.putText(frame, text, (10 + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)
        y += line_h


@dataclass
class Button:
    """Um botao clicavel na faixa inferior da janela."""
    action: str                       # acao disparada ("calibrar", "gravar", ...)
    label: str                        # texto exibido
    rect: tuple = field(default=(0, 0, 0, 0))   # (x0, y0, x1, y1), preenchido no layout


class ButtonBar:
    """
    Faixa de botoes no rodape da janela, para uso sem teclado.

    O layout e recalculado a cada frame (a janela pode mudar de tamanho) e o
    teste de clique usa as coordenadas da IMAGEM -- por isso a janela do modo
    simples e criada com WINDOW_AUTOSIZE, garantindo mapeamento 1:1 entre o
    pixel clicado e o pixel da imagem.
    """

    def __init__(self, items: list[tuple[str, str]]) -> None:
        self.buttons = [Button(action, label) for action, label in items]

    def layout(self, width: int, height: int) -> None:
        n = len(self.buttons)
        if n == 0:
            return
        margem, vao = 10, 8
        largura = (width - 2 * margem - vao * (n - 1)) // n
        y0 = height - BARRA_ALTURA + 8
        y1 = height - 10
        for i, b in enumerate(self.buttons):
            x0 = margem + i * (largura + vao)
            b.rect = (x0, y0, x0 + largura, y1)

    def draw(self, frame: np.ndarray, destaque: dict[str, str] | None = None) -> None:
        """
        Desenha os botoes. `destaque` mapeia acao -> "ativo" | "principal",
        permitindo sinalizar a gravacao em curso ou a acao recomendada.
        """
        destaque = destaque or {}
        h, w = frame.shape[:2]
        self.layout(w, h)

        faixa = frame.copy()
        cv2.rectangle(faixa, (0, h - BARRA_ALTURA), (w, h), (25, 25, 25), -1)
        cv2.addWeighted(faixa, 0.75, frame, 0.25, 0, frame)

        for b in self.buttons:
            x0, y0, x1, y1 = b.rect
            estado = destaque.get(b.action)
            cor = {
                "ativo": C_BOTAO_ATIVO,
                "principal": C_BOTAO_DESTAQUE,
            }.get(estado, C_BOTAO)
            cv2.rectangle(frame, (x0, y0), (x1, y1), cor, -1)
            cv2.rectangle(frame, (x0, y0), (x1, y1), C_BOTAO_BORDA, 1)

            escala = 0.62
            (tw, th), _ = cv2.getTextSize(b.label, cv2.FONT_HERSHEY_SIMPLEX, escala, 2)
            while tw > (x1 - x0) - 12 and escala > 0.35:
                escala -= 0.04
                (tw, th), _ = cv2.getTextSize(b.label, cv2.FONT_HERSHEY_SIMPLEX, escala, 2)
            tx = x0 + ((x1 - x0) - tw) // 2
            ty = y0 + ((y1 - y0) + th) // 2
            cv2.putText(frame, b.label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        escala, C_TEXTO, 2, cv2.LINE_AA)

    def hit(self, x: int, y: int) -> str | None:
        """Retorna a acao do botao sob o ponto (x, y), ou None."""
        for b in self.buttons:
            x0, y0, x1, y1 = b.rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                return b.action
        return None


def draw_opening_bar(
    frame: np.ndarray,
    opening_rel: float,
    cycles: CycleDetector,
    margem_inferior: int = 0,
) -> None:
    """
    Barra de biofeedback da abertura (0..faixa calibrada, ou 0..0.6 sem calibracao).

    `margem_inferior` reserva espaco no rodape (usado quando a faixa de botoes
    esta visivel, para que a barra nao fique por baixo dela).
    """
    h, w = frame.shape[:2]
    x0, y0 = w - 60, 60
    bar_h = h - 120 - margem_inferior
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
