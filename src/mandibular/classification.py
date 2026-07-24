"""
Criterios de classificacao simples das metricas (Semana 3 do cronograma).

Traduz os valores numericos calculados em categorias interpretaveis
(reduzida / normal / ampla, Classe I / II / III, etc.), sempre acompanhadas da
FONTE e de uma ressalva. Duas regras de veracidade guiam este modulo:

1. So classifica com base em valor de referencia com fonte identificavel.
   - Abertura bucal (40-60 mm) e diducao (9-12 mm): Dufour & Pillu,
     *Biomecanica Funcional*, cap. 16, p. 553 (ver docs/pesquisa_biomecanica.md).
   - Angulo de perfil (165 deg / 175 deg): material de analise facial
     (ver docs/pesquisa_biomecanica.md, secao de fontes de simetria).

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
PERFIL_CONVEXO_MAX = 165.0          # < 165 deg: perfil convexo (Classe II)
PERFIL_CONCAVO_MIN = 175.0          # > 175 deg: perfil concavo (Classe III)

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


def classify_profile(convexity_deg: float) -> Classification:
    """
    Classifica o angulo de perfil (Glabela-Subnasal-Pogonio).

    < 165 deg: perfil convexo (Classe II); > 175 deg: concavo (Classe III);
    entre eles: reto (Classe I). Fonte: material de analise facial.
    """
    if convexity_deg < PERFIL_CONVEXO_MAX:
        cat, level = "convexo (Classe II)", "info"
    elif convexity_deg > PERFIL_CONCAVO_MIN:
        cat, level = "concavo (Classe III)", "info"
    else:
        cat, level = "reto (Classe I)", "ok"
    return Classification(
        "Perfil", f"{convexity_deg:.0f} graus", cat, level,
        "ref. 165-175 graus (analise facial); estimativa 2D",
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


# ---------------------------------------------------------------------------
# Angulos de perfil (tecido mole) -> triagem de proeminencia
# ---------------------------------------------------------------------------
# Normas (media, desvio-padrao, fonte). Servem para a triagem "fora do padrao"
# (|z| > 2). São ESPECÍFICAS de populacao/idade -- valores default abaixo, com a
# fonte; a analise aceita um conjunto de normas alternativo.
#
# Fontes:
#  - convexidade G-Sn-Pg ~168+-5: atlas de tecido mole / EJO 2008; concordante
#    com a fotogrametria de Ngeow & Aljunid (PMC4298958).
#  - demais angulos: Ngeow & Aljunid, fotogrametria (PMC4298958). Nasolabial e
#    labiomental têm alta variabilidade (DP grande), por isso disparam menos.
PROFILE_ANGLE_NORMS = {
    "convexidade_facial": (168.0, 5.0, "G-Sn-Pg ~168+-5 (EJO 2008; PMC4298958)"),
    "convexidade_total": (147.0, 5.0, "G-Prn-Pg ~147+-5 (PMC4298958)"),
    "nasofrontal": (144.0, 5.0, "G-N-Prn ~144+-5 (PMC4298958)"),
    "nasolabial": (100.0, 11.0, "~Cm-Sn-Ls ~100+-11, alta variabilidade (PMC4298958)"),
    "labiomental": (134.0, 12.0, "Li-Sm-Pg ~134+-12, alta variabilidade (PMC4298958)"),
}

_PROFILE_ANGLE_LABELS = {
    "convexidade_facial": "Convexidade facial",
    "convexidade_total": "Convexidade total (c/ nariz)",
    "nasofrontal": "Angulo nasofrontal",
    "nasolabial": "Angulo nasolabial",
    "labiomental": "Angulo labiomental",
}

# Interpretacao (significado quando abaixo / acima da faixa normal).
_PROFILE_ANGLE_INTERP = {
    "convexidade_facial": ("perfil convexo (Classe II)", "perfil concavo (Classe III)"),
    "convexidade_total": ("nariz/perfil proeminente", "perfil retraido"),
    "nasofrontal": ("raiz do nariz proeminente", "raiz do nariz rasa"),
    "nasolabial": ("labio sup./nariz protruso", "labio sup. retruido"),
    "labiomental": ("labio inf./mento proeminente", "sulco mentolabial raso"),
}


@dataclass
class AngleFinding:
    """Achado de um angulo de perfil frente a norma."""
    key: str
    label: str
    value: float       # angulo medido (graus)
    mean: float        # media da norma
    sd: float          # desvio-padrao da norma
    z: float           # (valor - media) / sd -> "distancia" ao normal
    category: str      # "dentro do padrao" ou interpretacao da alteracao
    level: str         # "ok" | "atencao"
    note: str          # fonte + ressalva

    @property
    def out_of_norm(self) -> bool:
        return abs(self.z) >= 2.0


def classify_profile_prominence(angles, norms: dict | None = None) -> list[AngleFinding]:
    """
    Classifica cada angulo de perfil frente a norma, sinalizando proeminencia
    fora do padrao (|z| >= 2, ~fora de media +- 2 DP).

    `angles` e um ProfileAngles (mandibular.metrics). Retorna uma lista de
    AngleFinding, uma por angulo. O z-score serve de base para o biofeedback:
    quanto mais proximo de 0, mais dentro do padrao -> permite acompanhar a
    evolucao entre sessoes.
    """
    norms = norms or PROFILE_ANGLE_NORMS
    values = {
        "convexidade_facial": angles.convexidade_facial,
        "convexidade_total": angles.convexidade_total,
        "nasofrontal": angles.nasofrontal,
        "nasolabial": angles.nasolabial,
        "labiomental": angles.labiomental,
    }

    findings: list[AngleFinding] = []
    for key, value in values.items():
        mean, sd, src = norms[key]
        z = (value - mean) / sd if sd > 1e-6 else 0.0
        low_txt, high_txt = _PROFILE_ANGLE_INTERP[key]
        if z <= -2.0:
            cat, level = low_txt, "atencao"
        elif z >= 2.0:
            cat, level = high_txt, "atencao"
        else:
            cat, level = "dentro do padrao", "ok"
        findings.append(
            AngleFinding(
                key=key,
                label=_PROFILE_ANGLE_LABELS[key],
                value=float(value),
                mean=mean,
                sd=sd,
                z=float(z),
                category=cat,
                level=level,
                note=f"norma {src}; estimativa 2D",
            )
        )
    return findings
