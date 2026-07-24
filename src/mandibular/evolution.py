"""
Acompanhamento da evolucao do paciente entre sessoes.

A proposta do projeto pede "exportar dados para documentacao da evolucao do
paciente". Enquanto o CSV de sessao registra a serie temporal de UMA coleta,
este modulo agrega cada sessao em UMA linha-resumo e a acumula num historico
por paciente, permitindo comparar sessoes ao longo do tratamento.

Cenario tipico: paciente com desvio mandibular lateral. A cada sessao registra-se
o desvio (lateral, em mm/relativo, e angular) e a simetria; o grafico de evolucao
mostra a tendencia -- idealmente o desvio caminhando para 0 (mento centrado) =
melhora. Os valores sao de APOIO/acompanhamento, nao diagnostico.

A logica de dados (resumo, gravacao, leitura) fica aqui, sem matplotlib, para
ser testavel sem dependencias graficas; o grafico vive em `plotting.py`.
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, fields

import numpy as np

from .recorder import SessionRecorder


@dataclass
class SessionSummary:
    """Resumo de uma sessao (uma linha no historico do paciente)."""
    data: str                          # AAAA-MM-DD HH:MM:SS
    paciente: str
    n_amostras: int
    abertura_max_rel: float
    abertura_max_mm: float | None
    # Desvio lateral do mento (deslocamento mandibular) -- foco do acompanhamento
    desvio_lat_medio_rel: float
    desvio_lat_absmax_rel: float
    desvio_lat_medio_mm: float | None
    desvio_mandib_medio_deg: float | None    # desvio angular medio
    desvio_mandib_absmax_deg: float | None
    simetria_media: float | None             # 0..1
    repeticoes: int
    amplitude_cv: float | None
    duracao_cv: float | None


def summarize_session(
    recorder: SessionRecorder,
    cycles,
    paciente: str,
    data: str,
) -> SessionSummary:
    """
    Agrega as amostras de uma sessao em um SessionSummary.

    Robusto ao Sample do pipeline: campos podem ser None (frames invalidos) e
    metricas de simetria/angulo podem nem existir no Sample -- usa-se
    getattr(..., None) e filtra-se None antes de agregar.
    """
    s = recorder.samples
    if not s:
        raise ValueError("Sessao vazia: nada a resumir.")

    def col(attr: str) -> list[float]:
        return [v for v in (getattr(x, attr, None) for x in s) if v is not None]

    op_rel = col("opening_rel")
    lat_rel = col("lateral_rel")
    if not op_rel or not lat_rel:
        raise ValueError("Sessao sem frames validos: nada a resumir.")

    def mean_of(attr: str) -> float | None:
        vals = col(attr)
        return float(np.mean(vals)) if vals else None

    def absmax_of(attr: str) -> float | None:
        vals = col(attr)
        return float(np.max(np.abs(vals))) if vals else None

    op_mm = col("opening_mm")
    lat_rel_arr = np.array(lat_rel, dtype=float)
    rep = cycles.repeatability() if cycles is not None else {}

    return SessionSummary(
        data=data,
        paciente=paciente,
        n_amostras=len(s),
        abertura_max_rel=float(np.max(op_rel)),
        abertura_max_mm=(float(np.max(op_mm)) if op_mm else None),
        desvio_lat_medio_rel=float(lat_rel_arr.mean()),
        desvio_lat_absmax_rel=float(np.max(np.abs(lat_rel_arr))),
        desvio_lat_medio_mm=mean_of("lateral_mm"),
        desvio_mandib_medio_deg=mean_of("mand_deviation_deg"),
        desvio_mandib_absmax_deg=absmax_of("mand_deviation_deg"),
        simetria_media=mean_of("symmetry_index"),
        repeticoes=(cycles.repetitions if cycles is not None else 0),
        amplitude_cv=(rep.get("amplitude_cv") if rep else None),
        duracao_cv=(rep.get("duracao_cv") if rep else None),
    )


_HEADER = [f.name for f in fields(SessionSummary)]


def append_evolution(csv_path: str, summary: SessionSummary) -> str:
    """
    Acrescenta uma sessao ao historico do paciente (cria o cabecalho se novo).
    Retorna o caminho do arquivo.
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(_HEADER)
        d = asdict(summary)
        writer.writerow(["" if d[k] is None else d[k] for k in _HEADER])
    return csv_path


def load_evolution(csv_path: str) -> list[dict]:
    """Le o historico do paciente como uma lista de dicionarios (por sessao)."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def evolution_path(output_dir: str, paciente: str) -> str:
    """Caminho padrao do historico de um paciente."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in paciente).strip("_")
    return os.path.join(output_dir, f"evolucao_{safe or 'paciente'}.csv")
