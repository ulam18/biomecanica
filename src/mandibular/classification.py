"""
Criterios de classificacao simples das metricas (Semana 3 do cronograma).

Traduz os valores numericos calculados em categorias interpretaveis
(reduzida / normal / ampla, leve / acentuada, etc.), sempre acompanhadas da
FONTE e de uma ressalva. Duas regras de veracidade guiam este modulo:

1. So classifica com base em valor de referencia com fonte identificavel.
   - Abertura bucal (40-60 mm) e diducao (9-12 mm): Dufour & Pillu,
     *Biomecanica Funcional*, cap. 16, p. 553 (ver docs/pesquisa_biomecanica.md).

2. NAO inventa limiar clinico onde a literatura nao fornece. Para SIMETRIA
   facial, as fontes consultadas afirmam que "todas as faces sao assimetricas"
   e NAO trazem corte mm/% que separe fisiologico de patologico. Por isso a
   classificacao de simetria e apresentada como faixa DIDATICA (comparacao
   intrasujeito), explicitamente nao-clinica.

Todas as saidas sao de APOIO/DIDATICAS e nao constituem diagnostico.
"""

from __future__ import annotations

from dataclasses import dataclass

# -- Valores de referencia (com fonte) --------------------------------------
ABERTURA_NORMAL_MM = (40.0, 60.0)   # Dufour & Pillu, p. 553
DIDUCAO_NORMAL_MM = (9.0, 12.0)     # Dufour & Pillu, p. 553

# Desvio de mento: ordem de grandeza anedotica (casos cirurgicos severos:
# 6-7 mm) citada em Carlini & Gomes (2005). NAO e um limiar de referencia; usado
# apenas para contextualizar a magnitude, com ressalva explicita.
DESVIO_MENTO_SEVERO_MM = 6.0


@dataclass
class Classification:
    """Resultado de uma classificacao de metrica."""
    metric: str        # nome da metrica ("Abertura", "Diducao", ...)
    value: str         # valor formatado
    category: str      # categoria atribuida
    level: str         # "ok" | "atencao" | "info"
    note: str          # fonte e/ou ressalva

    def __str__(self) -> str:
        return f"{self.metric}: {self.value} -> {self.category} ({self.note})"


def classify_opening(opening_mm: float | None) -> Classification:
    """Classifica a abertura bucal maxima frente a faixa 40-60 mm (p. 553)."""
    if opening_mm is None:
        return Classification(
            "Abertura", "sem calibracao", "indefinida", "info",
            "informe --ref-mm para classificar em mm",
        )
    lo, hi = ABERTURA_NORMAL_MM
    if opening_mm < lo:
        cat, level = "reduzida", "atencao"
    elif opening_mm > hi:
        cat, level = "ampla", "atencao"
    else:
        cat, level = "normal", "ok"
    return Classification(
        "Abertura", f"{opening_mm:.1f} mm", cat, level,
        f"ref. {lo:.0f}-{hi:.0f} mm (Dufour & Pillu, p.553); estimativa",
    )


def classify_diduction(amplitude_mm: float | None) -> Classification:
    """Classifica a amplitude de lateralizacao frente a 9-12 mm (p. 553)."""
    if amplitude_mm is None:
        return Classification(
            "Diducao", "sem calibracao", "indefinida", "info",
            "informe --ref-mm para classificar em mm",
        )
    lo, hi = DIDUCAO_NORMAL_MM
    if amplitude_mm < lo:
        cat, level = "reduzida", "atencao"
    elif amplitude_mm > hi:
        cat, level = "aumentada", "atencao"
    else:
        cat, level = "normal", "ok"
    return Classification(
        "Diducao", f"{amplitude_mm:.1f} mm", cat, level,
        f"ref. {lo:.0f}-{hi:.0f} mm (Dufour & Pillu, p.553); estimativa",
    )


def classify_symmetry(index: float, good: float = 0.90, warn: float = 0.80) -> Classification:
    """
    Faixa DIDATICA de simetria (0..1) -- NAO e classificacao clinica.

    As fontes consultadas nao fornecem limiar que separe assimetria fisiologica
    de patologica ("todas as faces sao assimetricas", Carlini & Gomes, 2005).
    As faixas abaixo servem para comparacao intrasujeito e biofeedback.
    """
    pct = index * 100
    if index >= good:
        cat, level = "alta", "ok"
    elif index >= warn:
        cat, level = "moderada", "atencao"
    else:
        cat, level = "baixa", "atencao"
    return Classification(
        "Simetria", f"{pct:.0f}%", cat, level,
        "faixa didatica (sem corte clinico na literatura); intrasujeito",
    )


def classify_chin_deviation(deviation_mm: float | None) -> Classification:
    """
    Contextualiza o desvio lateral do mento (com sinal), se houver calibracao.

    NAO ha limiar de referencia nas fontes; usa-se apenas a ordem de grandeza
    anedotica de casos severos (6-7 mm, Carlini & Gomes, 2005) para contexto.
    """
    if deviation_mm is None:
        return Classification(
            "Desvio do mento", "sem calibracao", "indefinida", "info",
            "informe --ref-mm para estimar em mm",
        )
    lado = "direita" if deviation_mm >= 0 else "esquerda"
    mag = abs(deviation_mm)
    if mag >= DESVIO_MENTO_SEVERO_MM:
        cat, level = f"acentuado p/ {lado}", "atencao"
    else:
        cat, level = f"leve p/ {lado}", "info"
    return Classification(
        "Desvio do mento", f"{deviation_mm:+.1f} mm", cat, level,
        "sem corte clinico; 6-7 mm em casos severos (Carlini & Gomes); estimativa",
    )
