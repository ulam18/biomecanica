"""
Gravacao das medidas por frame e exportacao para CSV.

Cada linha registrada corresponde a um frame processado (com face valida ou
nao), permitindo reconstruir a trajetoria temporal do movimento mandibular,
auditar a qualidade da coleta e comparar sessoes.
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
    "abertura_bruta",
    "abertura_relativa",
    "abertura_filtrada",
    "abertura_mm",
    "desvio_lateral_bruto",
    "desvio_lateral_relativo",
    "desvio_lateral_filtrado",
    "desvio_lateral_mm",
    "direcao",
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
            fmt(self.opening_raw, 4),
            fmt(self.opening_rel, 6),
            fmt(self.opening_filtered, 6),
            fmt(self.opening_mm, 3),
            fmt(self.lateral_raw, 4),
            fmt(self.lateral_rel, 6),
            fmt(self.lateral_filtered, 6),
            fmt(self.lateral_mm, 3),
            self.direction,
            self.cycle_state.value,
            self.repetitions,
            self.quality_warning or "",
        ]


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
