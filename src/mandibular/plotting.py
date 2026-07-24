"""
Geracao de graficos da evolucao temporal do movimento mandibular.

Usa matplotlib com backend nao interativo ("Agg") para salvar imagens sem
depender de uma janela grafica.

IMPORTANTE: frames invalidos tem "*_filtered" = None (ver recorder.Sample).
Esses pontos viram NaN aqui, o que faz o matplotlib abrir uma lacuna no
grafico (sem desenhar uma linha) em vez de uma queda artificial para zero -
uma sessao sem nenhum frame valido nao deve parecer "toda zerada".
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .recorder import SessionRecorder  # noqa: E402


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _to_nan_array(values) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _series(recorder: SessionRecorder, use_mm: bool):
    """Monta as series (tempo, filtrado, bruto, mascara de validade, unidade)."""
    t = np.array([s.time_s for s in recorder.samples], dtype=float)
    valid_mask = np.array([s.frame_valid for s in recorder.samples], dtype=bool)

    if use_mm:
        opening_filt = _to_nan_array([s.opening_mm for s in recorder.samples])
        lateral_filt = _to_nan_array([s.lateral_mm for s in recorder.samples])
        opening_raw = opening_filt
        lateral_raw = lateral_filt
        unit = "mm"
    else:
        opening_filt = _to_nan_array([s.opening_filtered for s in recorder.samples])
        lateral_filt = _to_nan_array([s.lateral_filtered for s in recorder.samples])
        opening_raw = _to_nan_array([s.opening_rel for s in recorder.samples])
        lateral_raw = _to_nan_array([s.lateral_rel for s in recorder.samples])
        unit = "rel. (larg. facial)"

    return t, opening_filt, lateral_filt, opening_raw, lateral_raw, valid_mask, unit


def _shade_invalid(ax, t: np.ndarray, valid_mask: np.ndarray) -> None:
    """Sombra em cinza os trechos de tempo com frame invalido."""
    if len(t) == 0:
        return
    in_gap = False
    gap_start = t[0]
    for i, ok in enumerate(valid_mask):
        if not ok and not in_gap:
            in_gap, gap_start = True, t[i]
        elif ok and in_gap:
            in_gap = False
            ax.axvspan(gap_start, t[i], color="gray", alpha=0.12, linewidth=0)
    if in_gap:
        ax.axvspan(gap_start, t[-1], color="gray", alpha=0.12, linewidth=0)


def _no_valid_frames_figure(path: str, title: str) -> str:
    """Grafico substituto quando nao ha nenhum frame valido - nunca uma linha em zero."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5, "Sem frames validos", ha="center", va="center",
            fontsize=16, color="#a00000", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_opening_time(recorder: SessionRecorder, path: str, use_mm: bool = False) -> str:
    """Abertura bucal ao longo do tempo: bruta (fina) + filtrada (principal, so onde valida)."""
    if recorder.is_empty:
        raise ValueError("Nao ha amostras para plotar.")
    _ensure_parent(path)

    t, opening_filt, _lat_filt, opening_raw, _lat_raw, valid_mask, unit = _series(recorder, use_mm)

    if np.all(np.isnan(opening_filt)):
        return _no_valid_frames_figure(path, "Abertura bucal - evolucao temporal")

    fig, ax = plt.subplots(figsize=(10, 4))
    _shade_invalid(ax, t, valid_mask)
    ax.plot(t, opening_raw, color="#1f77b4", linewidth=0.8, alpha=0.35, label="bruto")
    ax.plot(t, opening_filt, color="#1f77b4", linewidth=1.6, label="filtrado (valido)")
    ax.set_ylabel(f"Abertura bucal [{unit}]")
    ax.set_xlabel("Tempo [s]")
    ax.set_title("Abertura bucal - evolucao temporal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    prev = 0
    for s in recorder.samples:
        if s.repetitions > prev:
            ax.axvline(s.time_s, color="#d62728", alpha=0.35, linestyle="--")
            prev = s.repetitions

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def plot_evolution(rows: list[dict], path: str, patient: str = "") -> str:
    """
    Grafico de evolucao do paciente ENTRE sessoes (a partir do historico em
    evolution.load_evolution). Mostra o desvio mandibular (lateral e angular) e
    a simetria ao longo das sessoes, com alvo em 0 para o desvio -- a melhora do
    tratamento aparece como aproximacao de 0.
    """
    if not rows:
        raise ValueError("Historico vazio: nada a plotar.")
    _ensure_parent(path)

    x = list(range(1, len(rows) + 1))
    labels = [str(r.get("data", ""))[:10] for r in rows]

    lat_mm = [_to_float(r.get("desvio_lat_medio_mm")) for r in rows]
    if all(v is not None for v in lat_mm):
        lat, lat_unit = lat_mm, "mm"
    else:
        lat = [_to_float(r.get("desvio_lat_medio_rel")) for r in rows]
        lat_unit = "rel."
    ang = [_to_float(r.get("desvio_mandib_medio_deg")) for r in rows]
    sym = [_to_float(r.get("simetria_media")) for r in rows]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax1.set_title((f"Evolucao do paciente {patient}").strip() or "Evolucao do paciente")

    ax1.plot(x, lat, "o-", color="#2ca02c")
    ax1.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    ax1.set_ylabel(f"Desvio lateral do\nmento [{lat_unit}]")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, ang, "o-", color="#1f77b4")
    ax2.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    ax2.set_ylabel("Desvio mandibular\nangular [graus]")
    ax2.grid(True, alpha=0.3)

    sym_pct = [None if v is None else v * 100 for v in sym]
    ax3.plot(x, sym_pct, "o-", color="#9467bd")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Simetria [%]")
    ax3.set_xlabel("Sessao")
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{i}\n{d}" for i, d in zip(x, labels)], fontsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_lateral_time(recorder: SessionRecorder, path: str, use_mm: bool = False) -> str:
    """Desvio lateral ao longo do tempo: bruto (fino) + filtrado (principal, so onde valido)."""
    if recorder.is_empty:
        raise ValueError("Nao ha amostras para plotar.")
    _ensure_parent(path)

    t, _open_filt, lateral_filt, _open_raw, lateral_raw, valid_mask, unit = _series(recorder, use_mm)

    if np.all(np.isnan(lateral_filt)):
        return _no_valid_frames_figure(path, "Desvio lateral da mandibula - evolucao temporal")

    fig, ax = plt.subplots(figsize=(10, 4))
    _shade_invalid(ax, t, valid_mask)
    ax.plot(t, lateral_raw, color="#2ca02c", linewidth=0.8, alpha=0.35, label="bruto")
    ax.plot(t, lateral_filt, color="#2ca02c", linewidth=1.6, label="filtrado (valido)")
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_ylabel(f"Desvio lateral [{unit}]")
    ax.set_xlabel("Tempo [s]")
    ax.set_title("Desvio lateral da mandibula - evolucao temporal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_trajectory(recorder: SessionRecorder, path: str, use_mm: bool = False) -> str:
    """Trajetoria abertura x desvio lateral, usando apenas frames validos."""
    if recorder.is_empty:
        raise ValueError("Nao ha amostras para plotar.")
    _ensure_parent(path)

    _t, opening_filt, lateral_filt, _o_raw, _l_raw, _valid_mask, unit = _series(recorder, use_mm)

    mask = ~(np.isnan(opening_filt) | np.isnan(lateral_filt))
    if not np.any(mask):
        return _no_valid_frames_figure(path, "Trajetoria: abertura x desvio lateral")

    opening = opening_filt[mask]
    lateral = lateral_filt[mask]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(lateral, opening, color="#9467bd", linewidth=0.8, alpha=0.7)
    ax.scatter(lateral, opening, c=np.arange(len(opening)), cmap="viridis", s=6)
    ax.axvline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel(f"Desvio lateral [{unit}]")
    ax.set_ylabel(f"Abertura bucal [{unit}]")
    ax.set_title("Trajetoria: abertura x desvio lateral (apenas frames validos)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
