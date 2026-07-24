"""
Assistente de calibracao guiada (boca fechada -> boca aberta).

A calibracao funcional (limiares de abertura para a maquina de estados e o
biofeedback) e sempre feita nesta escala relativa (normalizada pela largura
facial). A conversao para milimetros (`--ref-mm`) e independente e opcional.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class CalibrationResult:
    closed: float
    opened: float
    valid: bool
    message: str


class CalibrationAssistant:
    """
    Calibracao em duas fases, cada uma com contagem regressiva:
        fase 0 - boca fechada (coleta amostras, usa a mediana);
        fase 1 - abertura maxima confortavel (coleta amostras, usa o maximo).

    Uma calibracao so e aceita se a diferenca entre aberto e fechado for
    grande o suficiente (MIN_DIFFERENCE); caso contrario e marcada invalida
    e o usuario pode repetir (tecla C).
    """

    HOLD_SECONDS = 1.5
    MIN_DIFFERENCE = 0.05  # diferenca minima (unidades relativas) entre aberto e fechado

    def __init__(self) -> None:
        self.active = False
        self.phase = 0  # 0 = fechado, 1 = aberto
        self.phase_start = 0.0
        self.closed_samples: list[float] = []
        self.open_samples: list[float] = []
        self.last_result: CalibrationResult | None = None

    def start(self) -> None:
        self.active = True
        self.phase = 0
        self.phase_start = time.perf_counter()
        self.closed_samples.clear()
        self.open_samples.clear()
        self.last_result = None

    def update(self, opening: float, now: float) -> CalibrationResult | None:
        """
        Coleta uma amostra da fase atual. Retorna o resultado (valido ou nao)
        quando a calibracao termina; caso contrario, None.
        """
        if not self.active:
            return None

        elapsed = now - self.phase_start
        if self.phase == 0:
            self.closed_samples.append(opening)
            if elapsed >= self.HOLD_SECONDS:
                self.phase = 1
                self.phase_start = now
            return None

        self.open_samples.append(opening)
        if elapsed < self.HOLD_SECONDS:
            return None

        self.active = False
        closed = float(np.median(self.closed_samples)) if self.closed_samples else 0.0
        opened = float(np.max(self.open_samples)) if self.open_samples else closed
        valid = (opened - closed) >= self.MIN_DIFFERENCE
        message = (
            "Calibracao concluida."
            if valid
            else "Calibracao invalida: abra mais a boca e repita (tecle C)."
        )
        self.last_result = CalibrationResult(closed, opened, valid, message)
        return self.last_result

    def instruction(self, now: float) -> str:
        remaining = max(0.0, self.HOLD_SECONDS - (now - self.phase_start))
        if self.phase == 0:
            return f"CALIBRANDO: mantenha a BOCA FECHADA ({remaining:.1f}s)"
        return f"CALIBRANDO: abra a BOCA ao maximo ({remaining:.1f}s)"
