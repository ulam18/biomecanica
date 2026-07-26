"""
Gravacao das medidas por frame e exportacao para CSV.

Cada linha registrada corresponde a um frame processado (com face valida ou
nao), permitindo reconstruir a trajetoria temporal do movimento mandibular,
auditar a qualidade da coleta e comparar sessoes.

Unidades: colunas "*_relativo"/"*_filtrado"/"*_bruto" (abertura e lateral,
absoluta e dinamica) sao adimensionais, normalizadas pela distancia
INTEROCULAR (cantos externos dos olhos) -- ver metrics.REL_UNIT_DESCRIPTION.
Colunas "*_mm" so existem com calibracao de escala (--ref-mm). Convencao de
sinal da lateralidade: positivo/negativo mapeiam para direita/esquerda
anatomica de acordo com `mirrored` (ver coluna `anatomical_direction`, ja
calculada com a correcao; NAO deduza o lado so pelo sinal bruto).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

from .metrics import MovementState

CSV_COLUMNS = [
    "session_id",
    "frame",
    "timestamp",
    "tempo_s",
    "face_detectada",
    "frame_valido",
    "motivo_qualidade",
    "razao_facial",
    "roll_deg",
    "yaw_deg",
    "pitch_deg",
    "abertura_bruta",
    "abertura_relativa",
    "abertura_filtrada",
    "abertura_mm",
    "desvio_lateral_bruto",
    "desvio_lateral_relativo",
    "desvio_lateral_filtrado",
    "desvio_lateral_mm",
    "direcao",
    # -- Lateralidade absoluta vs dinamica (secao 2 do refinamento frontal).
    # "absolute_raw/filtered" sao ALIAS de desvio_lateral_relativo/filtrado
    # (mesma unidade relativa; nomes em ingles pedidos explicitamente).
    # "dynamic_*" = absolute - lateral_neutral_baseline (~0 na posicao neutra).
    "lateral_neutral_baseline",
    "lateral_absolute_raw",
    "lateral_absolute_filtered",
    "lateral_dynamic_raw",
    "lateral_dynamic_filtered",
    "anatomical_direction",  # alias de "direcao"
    "cycle_id",  # ciclo COMPLETO ao qual este frame pertence (vazio se nenhum)
    "estado_ciclo",
    "repeticoes",
    "aviso_qualidade",
]


@dataclass
class Sample:
    """
    Uma amostra temporal do movimento (item 12 do escopo funcional).

    Os campos "*_filtered" sao None quando frame_valid=False: um frame
    invalido nao produz uma nova medicao filtrada (o filtro fica congelado),
    entao gravar 0.0 seria fabricar um dado que nunca foi medido. Os campos
    "*_raw"/"*_rel" tambem sao None quando face_detected=False (sem face nao
    ha o que medir); quando ha face mas o frame e invalido por outro motivo
    (rosto longe, inclinado etc.), o valor bruto calculado e preservado para
    auditoria, mesmo que a medicao nao seja considerada confiavel.
    """
    session_id: str
    frame: int
    timestamp: str
    time_s: float
    face_detected: bool
    frame_valid: bool
    opening_raw: float | None
    opening_rel: float | None
    opening_filtered: float | None
    opening_mm: float | None
    lateral_raw: float | None
    lateral_rel: float | None
    lateral_filtered: float | None
    lateral_mm: float | None
    direction: str
    cycle_state: MovementState
    repetitions: int
    quality_warning: str | None
    quality_reason: str | None = None  # motivo curto/estavel (ver quality.QualityResult.reason)
    face_size_ratio: float | None = None  # face_width_px / frame_width_px
    roll_deg: float | None = None
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    lateral_neutral_baseline: float | None = None  # baseline usado neste frame (None se nao calibrado)
    lateral_dynamic_raw: float | None = None
    lateral_dynamic_filtered: float | None = None
    cycle_id: int | None = None  # preenchido por assign_cycle_ids() na exportacao

    def to_row(self) -> list:
        def fmt(v: float | None, decimals: int) -> str:
            return "" if v is None else f"{v:.{decimals}f}"

        return [
            self.session_id,
            self.frame,
            self.timestamp,
            f"{self.time_s:.4f}",
            int(self.face_detected),
            int(self.frame_valid),
            self.quality_reason or "",
            fmt(self.face_size_ratio, 4),
            fmt(self.roll_deg, 2),
            fmt(self.yaw_deg, 2),
            fmt(self.pitch_deg, 2),
            fmt(self.opening_raw, 4),
            fmt(self.opening_rel, 6),
            fmt(self.opening_filtered, 6),
            fmt(self.opening_mm, 3),
            fmt(self.lateral_raw, 4),
            fmt(self.lateral_rel, 6),
            fmt(self.lateral_filtered, 6),
            fmt(self.lateral_mm, 3),
            self.direction,
            fmt(self.lateral_neutral_baseline, 6),
            fmt(self.lateral_rel, 6),          # lateral_absolute_raw (alias)
            fmt(self.lateral_filtered, 6),     # lateral_absolute_filtered (alias)
            fmt(self.lateral_dynamic_raw, 6),
            fmt(self.lateral_dynamic_filtered, 6),
            self.direction,                    # anatomical_direction (alias)
            "" if self.cycle_id is None else self.cycle_id,
            self.cycle_state.value,
            self.repetitions,
            self.quality_warning or "",
        ]


def assign_cycle_ids(samples: list[Sample], cycles: list) -> None:
    """
    Marca em cada amostra (`sample.cycle_id`) a QUAL ciclo COMPLETO ela
    pertence, casando `time_s` com o intervalo [start_time, end_time] de cada
    `metrics.Cycle`. Amostras fora de qualquer ciclo completo (fora de
    sessao, ciclo incompleto/em andamento) ficam com cycle_id=None.

    Feito como pos-processamento (na exportacao) em vez de em tempo real:
    um ciclo so e conhecido DEPOIS de completo, entao nao ha como rotular a
    amostra no momento em que ela e gravada.
    """
    if not cycles:
        return
    for s in samples:
        for c in cycles:
            if c.start_time <= s.time_s <= c.end_time:
                s.cycle_id = c.cycle_id
                break


@dataclass
class SessionRecorder:
    """Acumula as amostras de uma sessao e as exporta."""
    samples: list[Sample] = field(default_factory=list)

    def add(self, sample: Sample) -> None:
        self.samples.append(sample)

    @property
    def is_empty(self) -> bool:
        return len(self.samples) == 0

    def clear(self) -> None:
        self.samples.clear()

    def to_csv(self, path: str) -> str:
        """Exporta as amostras para um arquivo CSV. Retorna o caminho."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
            for s in self.samples:
                writer.writerow(s.to_row())
        return path
