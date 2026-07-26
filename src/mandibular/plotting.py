"""
Geracao de graficos da evolucao temporal do movimento mandibular.

Usa matplotlib com backend nao interativo ("Agg") para salvar imagens sem
depender de uma janela grafica.

IMPORTANTE: frames invalidos tem "*_filtered" = None (ver recorder.Sample).
Esses pontos viram NaN aqui, o que faz o matplotlib abrir uma lacuna no
grafico (sem desenhar uma linha) em vez de uma queda artificial para zero -
uma sessao sem nenhum frame valido nao deve parecer "toda zerada".

Rotulos curtos ("Abertura relativa (adim.)" etc.) + nota discreta no rodape
de cada grafico substituem os rotulos longos anteriores ("[rel. (larg.
facial)]"), que ficavam cortados. A explicacao completa (landmarks 33-263)
fica nos metadados (ver metrics.REL_UNIT_DESCRIPTION). `bbox_inches="tight"`
em todo `savefig` garante que titulo/eixos/nota nao sejam cortados.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import lateral_direction  # noqa: E402
from .recorder import SessionRecorder  # noqa: E402

INTEROCULAR_NOTE = "Medidas adimensionais normalizadas pela distancia interocular."


def _opening_axis_label(use_mm: bool) -> str:
    return "Abertura (mm)" if use_mm else "Abertura relativa (adim.)"


def _lateral_absolute_axis_label(use_mm: bool) -> str:
    return "Posicao lateral (mm)" if use_mm else "Posicao lateral relativa (adim.)"


LATERAL_DYNAMIC_AXIS_LABEL = "Movimento desde o neutro (adim.)"


def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def _to_nan_array(values) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _sign_convention_text(mirrored: bool) -> str:
    pos = lateral_direction(1.0, mirrored)
    neg = lateral_direction(-1.0, mirrored)
    return f"positivo={pos} anatomica | negativo={neg} anatomica"


def _annotate_footer(ax, mirrored: bool | None = None) -> None:
    """
    Nota discreta abaixo do grafico (unidade +, quando aplicavel, a
    convencao de sinal desta sessao). Fica FORA da area dos eixos
    (transform=ax.transAxes, y<0); `bbox_inches="tight"` no savefig garante
    que ela nao seja cortada.
    """
    text = INTEROCULAR_NOTE
    if mirrored is not None:
        text += "  " + _sign_convention_text(mirrored)
    ax.text(
        0.5, -0.16, text, transform=ax.transAxes,
        ha="center", va="top", fontsize=7, color="gray",
    )


def _lateral_neutral_baseline(recorder: SessionRecorder) -> float | None:
    for s in recorder.samples:
        if s.lateral_neutral_baseline is not None:
            return s.lateral_neutral_baseline
    return None


def _series(recorder: SessionRecorder, use_mm: bool):
    """Monta as series (tempo, filtrado, bruto, mascara de validade, unidade)."""
    t = np.array([s.time_s for s in recorder.samples], dtype=float)
    valid_mask = np.array([s.frame_valid for s in recorder.samples], dtype=bool)

    if use_mm:
        opening_filt = _to_nan_array([s.opening_mm for s in recorder.samples])
        lateral_filt = _to_nan_array([s.lateral_mm for s in recorder.samples])
        opening_raw = opening_filt
        lateral_raw = lateral_filt
    else:
        opening_filt = _to_nan_array([s.opening_filtered for s in recorder.samples])
        lateral_filt = _to_nan_array([s.lateral_filtered for s in recorder.samples])
        opening_raw = _to_nan_array([s.opening_rel for s in recorder.samples])
        lateral_raw = _to_nan_array([s.lateral_rel for s in recorder.samples])

    return t, opening_filt, lateral_filt, opening_raw, lateral_raw, valid_mask


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


def _save(fig, path: str) -> str:
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _no_valid_frames_figure(path: str, title: str) -> str:
    """Grafico substituto quando nao ha nenhum frame valido - nunca uma linha em zero."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.text(0.5, 0.5, "Sem frames validos", ha="center", va="center",
            fontsize=16, color="#a00000", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return _save(fig, path)


def plot_opening_time(recorder: SessionRecorder, path: str, use_mm: bool = False) -> str:
    """Abertura bucal ao longo do tempo: bruta (fina) + filtrada (principal, so onde valida)."""
    if recorder.is_empty:
        raise ValueError("Nao ha amostras para plotar.")
    _ensure_parent(path)

    t, opening_filt, _lat_filt, opening_raw, _lat_raw, valid_mask = _series(recorder, use_mm)

    if np.all(np.isnan(opening_filt)):
        return _no_valid_frames_figure(path, "Abertura bucal - evolucao temporal")

    fig, ax = plt.subplots(figsize=(10, 4))
    _shade_invalid(ax, t, valid_mask)
    ax.plot(t, opening_raw, color="#1f77b4", linewidth=0.8, alpha=0.35, label="bruto")
    ax.plot(t, opening_filt, color="#1f77b4", linewidth=1.6, label="filtrado (valido)")
    ax.set_ylabel(_opening_axis_label(use_mm))
    ax.set_xlabel("Tempo [s]")
    ax.set_title("Abertura bucal - evolucao temporal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    prev = 0
    for s in recorder.samples:
        if s.repetitions > prev:
            ax.axvline(s.time_s, color="#d62728", alpha=0.35, linestyle="--")
            prev = s.repetitions

    _annotate_footer(ax)
    fig.tight_layout()
    return _save(fig, path)


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
        lat_unit = "adim."
    ang = [_to_float(r.get("desvio_mandib_medio_deg")) for r in rows]
    sym = [_to_float(r.get("simetria_media")) for r in rows]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax1.set_title((f"Evolucao do paciente {patient}").strip() or "Evolucao do paciente")

    ax1.plot(x, lat, "o-", color="#2ca02c")
    ax1.axhline(0.0, color="gray", linewidth=1.0, linestyle="--")
    ax1.set_ylabel(f"Desvio lateral do\nmento ({lat_unit})")
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
    return _save(fig, path)


def plot_lateral_time(recorder: SessionRecorder, path: str, use_mm: bool = False,
                       mirrored: bool = True) -> str:
    """
    Posicao do queixo em relacao a linha media facial (lateral_absolute) ao
    longo do tempo: bruta (fina) + filtrada (principal, so onde valida).
    Linha horizontal em `lateral_neutral_baseline` (posicao neutra
    calibrada), quando disponivel.
    """
    if recorder.is_empty:
        raise ValueError("Nao ha amostras para plotar.")
    _ensure_parent(path)

    t, _open_filt, lateral_filt, _open_raw, lateral_raw, valid_mask = _series(recorder, use_mm)

    if np.all(np.isnan(lateral_filt)):
        return _no_valid_frames_figure(
            path, "Posicao do queixo em relacao a linha media facial - evolucao temporal"
        )

    fig, ax = plt.subplots(figsize=(10, 4))
    _shade_invalid(ax, t, valid_mask)
    ax.plot(t, lateral_raw, color="#2ca02c", linewidth=0.8, alpha=0.35, label="bruto")
    ax.plot(t, lateral_filt, color="#2ca02c", linewidth=1.6, label="filtrado (valido)")
    baseline = _lateral_neutral_baseline(recorder)
    if baseline is not None and not use_mm:
        ax.axhline(baseline, color="gray", linewidth=1.0, linestyle="--",
                    label="posicao neutra calibrada")
    ax.set_ylabel(_lateral_absolute_axis_label(use_mm))
    ax.set_xlabel("Tempo [s]")
    ax.set_title("Posicao do queixo em relacao a linha media facial - evolucao temporal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    _annotate_footer(ax, mirrored)

    fig.tight_layout()
    return _save(fig, path)


def _dynamic_series(recorder: SessionRecorder):
    """
    Series de lateral_dynamic (sempre em unidade relativa: o baseline neutro
    e definido nessa escala pela calibracao, sem equivalente em mm).
    """
    t = np.array([s.time_s for s in recorder.samples], dtype=float)
    valid_mask = np.array([s.frame_valid for s in recorder.samples], dtype=bool)
    dyn_filt = _to_nan_array([s.lateral_dynamic_filtered for s in recorder.samples])
    dyn_raw = _to_nan_array([s.lateral_dynamic_raw for s in recorder.samples])
    return t, dyn_filt, dyn_raw, valid_mask


def plot_lateral_dynamic_time(recorder: SessionRecorder, path: str, mirrored: bool = True) -> str:
    """
    Movimento do queixo desde a posicao neutra calibrada (lateral_dynamic =
    lateral_absolute - lateral_neutral_baseline) ao longo do tempo: fica ~0
    na posicao neutra mesmo com assimetria facial estatica. Sem baseline
    calibrado (sessao nao calibrada), todos os pontos sao NaN.
    """
    if recorder.is_empty:
        raise ValueError("Nao ha amostras para plotar.")
    _ensure_parent(path)

    t, dyn_filt, dyn_raw, valid_mask = _dynamic_series(recorder)

    if np.all(np.isnan(dyn_filt)):
        return _no_valid_frames_figure(
            path,
            "Movimento do queixo desde a posicao neutra calibrada - evolucao temporal "
            "(sem baseline/frames validos)",
        )

    fig, ax = plt.subplots(figsize=(10, 4))
    _shade_invalid(ax, t, valid_mask)
    ax.plot(t, dyn_raw, color="#e377c2", linewidth=0.8, alpha=0.35, label="bruto")
    ax.plot(t, dyn_filt, color="#e377c2", linewidth=1.6, label="filtrado (valido)")
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", label="posicao neutra (0)")
    ax.set_ylabel(LATERAL_DYNAMIC_AXIS_LABEL)
    ax.set_xlabel("Tempo [s]")
    ax.set_title("Movimento do queixo desde a posicao neutra calibrada - evolucao temporal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    _annotate_footer(ax, mirrored)

    fig.tight_layout()
    return _save(fig, path)


def plot_trajectory_dynamic(recorder: SessionRecorder, path: str, mirrored: bool = True) -> str:
    """Trajetoria: abertura x movimento desde o neutro, usando so frames validos."""
    if recorder.is_empty:
        raise ValueError("Nao ha amostras para plotar.")
    _ensure_parent(path)

    _t, opening_filt, _lat_filt, _o_raw, _l_raw, _valid_mask = _series(recorder, use_mm=False)
    _t2, dyn_filt, _dyn_raw, _valid_mask2 = _dynamic_series(recorder)

    mask = ~(np.isnan(opening_filt) | np.isnan(dyn_filt))
    if not np.any(mask):
        return _no_valid_frames_figure(
            path, "Trajetoria: abertura x movimento desde o neutro (sem baseline/frames validos)"
        )

    opening = opening_filt[mask]
    lateral = dyn_filt[mask]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(lateral, opening, color="#8c564b", linewidth=0.8, alpha=0.7)
    ax.scatter(lateral, opening, c=np.arange(len(opening)), cmap="viridis", s=6)
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(LATERAL_DYNAMIC_AXIS_LABEL)
    ax.set_ylabel(_opening_axis_label(False))
    ax.set_title("Trajetoria: abertura x movimento desde o neutro (frames validos)")
    ax.grid(True, alpha=0.3)
    _annotate_footer(ax, mirrored)

    fig.tight_layout()
    return _save(fig, path)


def _cycle_progress_series(recorder: SessionRecorder, cycle) -> tuple[np.ndarray, np.ndarray]:
    """
    Progresso normalizado [0,1] de um ciclo, POR FASE:
        0%   = inicio EXPORTADO do ciclo (ver metrics.Cycle.start_opening,
               idealmente dentro da banda de boca realmente fechada);
        50%  = pico da abertura (ou centro do plato, se houver amostras
               dentro de 95% do maximo -- robusto a ruido no exato ponto de
               maior abertura);
        100% = fim EXPORTADO do ciclo (idem, `end_opening`).
    A fase de abertura (0-50%) e a de fechamento (50-100%) sao reamostradas
    SEPARADAMENTE (cada uma em relacao a sua propria duracao), entao o pico
    fica sempre alinhado em 50% mesmo que abertura e fechamento tenham
    duracoes diferentes.

    Usa somente frames validos com cycle_id == cycle.cycle_id (atribuido por
    recorder.assign_cycle_ids, chamado antes destes graficos em
    exporter.export_session). Retorna (progresso, abertura) ordenados por
    tempo; arrays vazios se o ciclo nao tiver >=2 amostras validas.
    """
    pts = [
        (s.time_s, s.opening_filtered) for s in recorder.samples
        if s.cycle_id == cycle.cycle_id and s.frame_valid and s.opening_filtered is not None
    ]
    if len(pts) < 2:
        return np.array([]), np.array([])
    pts.sort(key=lambda p: p[0])
    times = np.array([p[0] for p in pts])
    openings = np.array([p[1] for p in pts])

    max_op = float(openings.max())
    plateau_mask = openings >= 0.95 * max_op
    plateau_times = times[plateau_mask]
    peak_t = float((plateau_times.min() + plateau_times.max()) / 2.0)

    opening_span = max(peak_t - cycle.start_time, 1e-6)
    closing_span = max(cycle.end_time - peak_t, 1e-6)
    progress = np.where(
        times <= peak_t,
        0.5 * (times - cycle.start_time) / opening_span,
        0.5 + 0.5 * (times - peak_t) / closing_span,
    )
    progress = np.clip(progress, 0.0, 1.0)
    return progress, openings


def _style_cycle_progress_axes(ax, title: str, start_ok: bool = True, end_ok: bool = True) -> None:
    """
    `start_ok`/`end_ok`: so escreve "(fechado)" no rotulo de 0%/100% se os
    ciclos efetivamente plotados atenderam o criterio de banda fechada
    (`Cycle.start_within_baseline`/`end_within_baseline`) -- nunca afirmar
    "fechado" sem verificar (secao 4 do refinamento).
    """
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Progresso normalizado do ciclo [%]")
    ax.set_ylabel(_opening_axis_label(False))
    ax.set_title(title)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    start_lbl = "0%\ninicio (fechado)" if start_ok else "0%\ninicio"
    end_lbl = "100%\nfim (fechado)" if end_ok else "100%\nfim"
    ax.set_xticklabels([start_lbl, "50%\npico", end_lbl], fontsize=8)
    ax.grid(True, alpha=0.3)
    _annotate_footer(ax)


def plot_cycles_normalized(recorder: SessionRecorder, cycles, path: str) -> str:
    """
    Um traco por ciclo COMPLETO (abertura filtrada x progresso normalizado
    por fase, ver `_cycle_progress_series`), cada um com uma cor, com o pico
    alinhado em 50%. Usa somente frames validos; um ciclo sem amostras
    validas suficientes e simplesmente omitido (nunca se liga um trecho
    invalido nem se inventa zero).
    """
    _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    plotted_cycles = []

    for c in cycles.cycles:
        progress, openings = _cycle_progress_series(recorder, c)
        if len(progress) < 2:
            continue
        color = cmap(len(plotted_cycles) % 10)
        ax.plot(progress * 100, openings, color=color, linewidth=1.3, label=f"ciclo {c.cycle_id}")
        peak_idx = int(np.argmax(openings))
        ax.scatter([progress[peak_idx] * 100], [openings[peak_idx]],
                    color=color, s=35, marker="o", zorder=5)
        ax.scatter([progress[0] * 100, progress[-1] * 100], [openings[0], openings[-1]],
                    color=color, s=18, marker="s", zorder=4)
        plotted_cycles.append(c)

    if not plotted_cycles:
        plt.close(fig)
        return _no_valid_frames_figure(path, "Ciclos individuais (sem ciclos completos validos)")

    start_ok = all(c.start_within_baseline for c in plotted_cycles)
    end_ok = all(c.end_within_baseline for c in plotted_cycles)
    _style_cycle_progress_axes(
        ax, f"Ciclos individuais (n={len(plotted_cycles)}; quadrado=inicio/fim, bola=pico)",
        start_ok=start_ok, end_ok=end_ok,
    )
    if len(plotted_cycles) <= 10:
        ax.legend(loc="upper right", fontsize=7, ncol=2)

    fig.tight_layout()
    return _save(fig, path)


def plot_cycles_mean_band(recorder: SessionRecorder, cycles, path: str, n_points: int = 50) -> str:
    """
    Curva media dos ciclos completos (abertura x progresso normalizado por
    fase, pico sempre em 50%), com faixa de variabilidade (+-1 desvio-padrao)
    entre ciclos. A media comeca e termina proxima do baseline de boca
    fechada porque cada ciclo, individualmente, ja e delimitado do fechado ao
    fechado (ver metrics.CycleDetector -- pre-buffer / fechamento estavel).
    """
    _ensure_parent(path)
    grid = np.linspace(0.0, 1.0, n_points)
    interpolated = []
    plotted_cycles = []

    for c in cycles.cycles:
        progress, openings = _cycle_progress_series(recorder, c)
        if len(progress) < 2:
            continue
        order = np.argsort(progress)
        interpolated.append(np.interp(grid, progress[order], openings[order]))
        plotted_cycles.append(c)

    if not interpolated:
        return _no_valid_frames_figure(path, "Curva media dos ciclos (sem ciclos completos validos)")

    stacked = np.vstack(interpolated)
    mean_curve = stacked.mean(axis=0)
    std_curve = stacked.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(grid * 100, mean_curve - std_curve, mean_curve + std_curve,
                     color="#1f77b4", alpha=0.25, label="+-1 dp")
    ax.plot(grid * 100, mean_curve, color="#1f77b4", linewidth=2.0, label="media")
    start_ok = all(c.start_within_baseline for c in plotted_cycles)
    end_ok = all(c.end_within_baseline for c in plotted_cycles)
    _style_cycle_progress_axes(
        ax, f"Curva media dos ciclos (n={len(plotted_cycles)})", start_ok=start_ok, end_ok=end_ok
    )
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    return _save(fig, path)


def plot_trajectory(recorder: SessionRecorder, path: str, use_mm: bool = False,
                     mirrored: bool = True) -> str:
    """Trajetoria: abertura x posicao do queixo, usando apenas frames validos."""
    if recorder.is_empty:
        raise ValueError("Nao ha amostras para plotar.")
    _ensure_parent(path)

    _t, opening_filt, lateral_filt, _o_raw, _l_raw, _valid_mask = _series(recorder, use_mm)

    mask = ~(np.isnan(opening_filt) | np.isnan(lateral_filt))
    if not np.any(mask):
        return _no_valid_frames_figure(path, "Trajetoria: abertura x posicao do queixo")

    opening = opening_filt[mask]
    lateral = lateral_filt[mask]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(lateral, opening, color="#9467bd", linewidth=0.8, alpha=0.7)
    ax.scatter(lateral, opening, c=np.arange(len(opening)), cmap="viridis", s=6)
    baseline = _lateral_neutral_baseline(recorder)
    if baseline is not None and not use_mm:
        ax.axvline(baseline, color="gray", linewidth=1.0, linestyle="--")
    ax.set_xlabel(_lateral_absolute_axis_label(use_mm))
    ax.set_ylabel(_opening_axis_label(use_mm))
    ax.set_title("Trajetoria: abertura x posicao do queixo (apenas frames validos)")
    ax.grid(True, alpha=0.3)
    _annotate_footer(ax, mirrored)

    fig.tight_layout()
    return _save(fig, path)
