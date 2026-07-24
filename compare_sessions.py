"""
Compara duas ou mais sessoes exportadas (dados.csv), lado a lado.

Uso:
    python compare_sessions.py sessao1/dados.csv sessao2/dados.csv
    python compare_sessions.py sessao1/dados.csv sessao2/dados.csv --normalized-time

Gera uma tabela-resumo no terminal e um grafico comparativo (abertura e
desvio lateral filtrados ao longo do tempo). Nao emite diagnostico.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REQUIRED_COLUMNS = {
    "frame",
    "tempo_s",
    "abertura_relativa",
    "abertura_filtrada",
    "desvio_lateral_relativo",
    "desvio_lateral_filtrado",
    "estado_ciclo",
    "repeticoes",
}


@dataclass
class SessionData:
    name: str
    path: str
    rows: list[dict]

    def col(self, name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in self.rows], dtype=float)


def load_session(path: str) -> SessionData:
    """Le e valida um dados.csv de sessao. Lanca erro claro se invalido."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo nao encontrado: {path}")

    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"CSV vazio: {path}")

    missing = REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        raise ValueError(f"Colunas ausentes em {path}: {sorted(missing)}")

    session_dir = os.path.basename(os.path.dirname(os.path.abspath(path)))
    name = session_dir or os.path.basename(path)
    return SessionData(name=name, path=path, rows=rows)


def _cv(values: np.ndarray) -> float:
    if len(values) == 0 or abs(float(np.mean(values))) < 1e-9:
        return 0.0
    return float(np.std(values) / np.mean(values))


def session_stats(s: SessionData) -> dict:
    """Reconstroi amplitude/duracao por ciclo a partir das transicoes de 'repeticoes'."""
    opening = s.col("abertura_filtrada")
    lateral = s.col("desvio_lateral_filtrado")
    reps = s.col("repeticoes")
    t = s.col("tempo_s")

    amplitudes: list[float] = []
    duracoes: list[float] = []
    prev_reps = 0
    cycle_start_t = float(t[0]) if len(t) else 0.0
    cycle_peak = float(opening[0]) if len(opening) else 0.0
    for i in range(len(s.rows)):
        cur_reps = int(reps[i])
        cycle_peak = max(cycle_peak, float(opening[i]))
        if cur_reps > prev_reps:
            amplitudes.append(cycle_peak)
            duracoes.append(float(t[i]) - cycle_start_t)
            cycle_start_t = float(t[i])
            cycle_peak = float(opening[i])
            prev_reps = cur_reps

    return {
        "sessao": s.name,
        "n_amostras": len(s.rows),
        "abertura_max": float(np.max(opening)) if len(opening) else 0.0,
        "abertura_media": float(np.mean(opening)) if len(opening) else 0.0,
        "desvio_lateral_max_abs": float(np.max(np.abs(lateral))) if len(lateral) else 0.0,
        "repeticoes": int(reps.max()) if len(reps) else 0,
        "amplitude_media": float(np.mean(amplitudes)) if amplitudes else 0.0,
        "amplitude_cv": _cv(np.array(amplitudes)),
        "duracao_media_s": float(np.mean(duracoes)) if duracoes else 0.0,
        "duracao_cv": _cv(np.array(duracoes)),
    }


def print_summary_table(stats_list: list[dict]) -> None:
    cols = [
        "sessao", "n_amostras", "abertura_max", "abertura_media",
        "desvio_lateral_max_abs", "repeticoes", "amplitude_media",
        "amplitude_cv", "duracao_media_s", "duracao_cv",
    ]

    def fmt(v) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    widths = {c: max(len(c), *(len(fmt(s[c])) for s in stats_list)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for s in stats_list:
        print("  ".join(fmt(s[c]).ljust(widths[c]) for c in cols))


def plot_comparison(sessions: list[SessionData], out_path: str, normalize_time: bool) -> str:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for s in sessions:
        t = s.col("tempo_s")
        if normalize_time and len(t) and t[-1] > 0:
            t = t / t[-1]
        ax1.plot(t, s.col("abertura_filtrada"), label=s.name, linewidth=1.3)
        ax2.plot(t, s.col("desvio_lateral_filtrado"), label=s.name, linewidth=1.3)

    ax1.set_ylabel("Abertura (filtrada)")
    ax1.set_title("Comparacao entre sessoes")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_ylabel("Desvio lateral (filtrado)")
    ax2.set_xlabel("Tempo normalizado [0-1]" if normalize_time else "Tempo [s]")
    ax2.axhline(0.0, color="gray", linewidth=0.8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Compara duas ou mais sessoes exportadas.")
    p.add_argument("sessions", nargs="+", help="caminhos dos dados.csv das sessoes (>=2)")
    p.add_argument("--output", default="resultados/comparacao.png", help="grafico comparativo de saida")
    p.add_argument(
        "--normalized-time", action="store_true",
        help="normaliza o eixo do tempo (0 a 1) para comparar sessoes de duracao diferente",
    )
    a = p.parse_args()

    if len(a.sessions) < 2:
        print("Informe pelo menos duas sessoes para comparar.", file=sys.stderr)
        sys.exit(1)

    try:
        sessions = [load_session(path) for path in a.sessions]
    except (FileNotFoundError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)

    stats_list = [session_stats(s) for s in sessions]

    print("\nCOMPARACAO ENTRE SESSOES (apoio funcional; nao e diagnostico)\n")
    print_summary_table(stats_list)

    out = plot_comparison(sessions, a.output, a.normalized_time)
    print(f"\n[export] grafico comparativo: {out}")


if __name__ == "__main__":
    main()
