"""
Biofeedback funcional simples, em linguagem nao diagnostica.

As mensagens descrevem apenas o que foi observado (posicionamento, faixa de
abertura treinada, direcao do desvio, repetibilidade) - nunca um julgamento
clinico ou sugestao de diagnostico de DTM.
"""

from __future__ import annotations

from .metrics import CycleDetector
from .quality import FrameQuality, QualityResult

REPEATABILITY_CV_WARNING = 0.35


def biofeedback_messages(
    opening_rel: float,
    direction: str,
    cycles: CycleDetector,
    quality: QualityResult,
) -> list[str]:
    """Gera mensagens curtas de apoio para exibir na interface."""
    if quality.quality != FrameQuality.VALIDA:
        return [quality.message] if quality.message else []

    msgs: list[str] = []

    if cycles.is_calibrated and cycles.baseline is not None and cycles.span:
        frac = (opening_rel - cycles.baseline) / cycles.span
        if frac >= cycles.config.open_fraction:
            msgs.append("Abertura dentro da faixa treinada")
        elif frac < cycles.config.close_fraction:
            pass  # boca fechada: nada a reportar
        else:
            msgs.append("Abertura abaixo da faixa treinada")

    if direction == "direita":
        msgs.append("Desvio para a direita")
    elif direction == "esquerda":
        msgs.append("Desvio para a esquerda")

    rep = cycles.repeatability()
    if rep and rep.get("amplitude_cv", 0.0) > REPEATABILITY_CV_WARNING:
        msgs.append("Movimento pouco repetivel entre os ciclos")

    return msgs
