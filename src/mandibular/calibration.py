"""
Assistente de calibracao guiada (boca fechada -> boca aberta).

A calibracao funcional (limiares de abertura para a maquina de estados e o
biofeedback) e sempre feita nesta escala relativa (normalizada pela largura
facial). A conversao para milimetros (`--ref-mm`) e independente e opcional.

Durante a fase de boca fechada, tambem coleta amostras de lateralidade
(`lateral_absolute`) para calcular o baseline neutro (`lateral_neutral_baseline`,
usado para `lateral_dynamic = lateral_absolute - baseline`). Se a lateralidade
variar demais nessa fase (cabeca/mandibula instavel), a calibracao inteira e
rejeitada -- um baseline neutro instavel invalidaria toda a analise dinamica
subsequente.
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
    lateral_baseline: float | None = None      # mediana da lateral_absolute (boca fechada)
    lateral_baseline_std: float | None = None  # desvio-padrao na mesma fase (estabilidade)


class CalibrationAssistant:
    """
    Calibracao em duas fases, cada uma com contagem regressiva:
        fase 0 - boca fechada (coleta amostras de abertura E lateralidade,
                 usa a mediana de cada uma);
        fase 1 - abertura maxima confortavel (coleta amostras, usa o maximo).

    Uma calibracao so e aceita se:
      - a diferenca entre aberto e fechado for grande o suficiente
        (MIN_DIFFERENCE); e
      - a lateralidade na fase de boca fechada for estavel (desvio-padrao ate
        MAX_LATERAL_BASELINE_STD, em unidades relativas -- valor permissivo:
        ruido tipico de deteccao fica bem abaixo disso, mas uma cabeca
        oscilando ou uma mordida assimetrica sendo mantida durante a fase
        "fechada" fica acima).
    Caso contrario e marcada invalida e o usuario pode repetir (tecla C).
    """

    HOLD_SECONDS = 1.5
    MIN_DIFFERENCE = 0.05  # diferenca minima (unidades relativas) entre aberto e fechado
    MAX_LATERAL_BASELINE_STD = 0.05  # desvio-padrao maximo aceitavel do lateral na fase fechada

    def __init__(self) -> None:
        self.active = False
        self.phase = 0  # 0 = fechado, 1 = aberto
        self.phase_start = 0.0
        self.closed_samples: list[float] = []
        self.open_samples: list[float] = []
        self.closed_lateral_samples: list[float] = []
        self.last_result: CalibrationResult | None = None

    def start(self) -> None:
        self.active = True
        self.phase = 0
        self.phase_start = time.perf_counter()
        self.closed_samples.clear()
        self.open_samples.clear()
        self.closed_lateral_samples.clear()
        self.last_result = None

    def update(
        self, opening: float, now: float, lateral: float | None = None
    ) -> CalibrationResult | None:
        """
        Coleta uma amostra da fase atual (`lateral` = lateral_absolute do
        frame, so relevante na fase 0). Retorna o resultado (valido ou nao)
        quando a calibracao termina; caso contrario, None.
        """
        if not self.active:
            return None

        elapsed = now - self.phase_start
        if self.phase == 0:
            self.closed_samples.append(opening)
            if lateral is not None:
                self.closed_lateral_samples.append(lateral)
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
        opening_valid = (opened - closed) >= self.MIN_DIFFERENCE

        lateral_baseline: float | None = None
        lateral_std: float | None = None
        lateral_valid = True
        if self.closed_lateral_samples:
            lateral_baseline = float(np.median(self.closed_lateral_samples))
            lateral_std = float(np.std(self.closed_lateral_samples))
            lateral_valid = lateral_std <= self.MAX_LATERAL_BASELINE_STD

        valid = opening_valid and lateral_valid
        if not opening_valid:
            message = "Calibracao invalida: abra mais a boca e repita a calibracao."
        elif not lateral_valid:
            message = (
                f"Calibracao invalida: mandibula/cabeca instavel na fase fechada "
                f"(desvio lateral {lateral_std:.3f} > {self.MAX_LATERAL_BASELINE_STD:.3f}); "
                f"mantenha-se parado e repita a calibracao."
            )
        else:
            message = "Calibracao concluida."

        self.last_result = CalibrationResult(
            closed, opened, valid, message, lateral_baseline, lateral_std
        )
        return self.last_result

    def instruction(self, now: float) -> str:
        remaining = max(0.0, self.HOLD_SECONDS - (now - self.phase_start))
        if self.phase == 0:
            return f"CALIBRANDO: mantenha a BOCA FECHADA ({remaining:.1f}s)"
        return f"CALIBRANDO: abra a BOCA ao maximo ({remaining:.1f}s)"
