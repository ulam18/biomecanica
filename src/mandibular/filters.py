"""
Filtragem simples dos sinais biomecanicos (media movel exponencial - EMA).

Reduz o ruido de deteccao dos landmarks quadro a quadro sem introduzir atraso
perceptivel, ao contrario de uma media movel de janela larga.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EMAFilter:
    """Media movel exponencial: y[n] = alpha*x[n] + (1-alpha)*y[n-1]."""

    alpha: float = 0.5

    def __post_init__(self) -> None:
        self._value: float | None = None

    def reset(self) -> None:
        self._value = None

    def update(self, x: float) -> float:
        if self._value is None:
            self._value = x
        else:
            self._value = self.alpha * x + (1.0 - self.alpha) * self._value
        return self._value

    @property
    def value(self) -> float | None:
        return self._value
