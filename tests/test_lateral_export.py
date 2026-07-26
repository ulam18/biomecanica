"""
Testes de exportacao/graficos das novas colunas e metricas de lateralidade
dinamica e ciclos (secoes 5/6/9/10 do refinamento frontal), e de
compatibilidade de compare_sessions.py com sessoes antigas (sem
lateral_dynamic_filtered).
"""

from __future__ import annotations

import csv
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import inspect

from mandibular import plotting  # noqa: E402
from mandibular.exporter import LATERAL_SIGN_CONVENTION, export_session  # noqa: E402
from mandibular.metrics import CycleDetector, MovementState  # noqa: E402
from mandibular.recorder import Sample, SessionRecorder, assign_cycle_ids  # noqa: E402


def _sample(i: int, t: float, opening: float, lateral: float, baseline: float | None) -> Sample:
    dyn = (lateral - baseline) if baseline is not None else None
    return Sample(
        session_id="s", frame=i, timestamp=f"{t:.3f}s", time_s=t,
        face_detected=True, frame_valid=True,
        opening_raw=opening * 200, opening_rel=opening, opening_filtered=opening, opening_mm=None,
        lateral_raw=lateral * 200, lateral_rel=lateral, lateral_filtered=lateral, lateral_mm=None,
        direction="centro", cycle_state=MovementState.FECHADO, repetitions=0,
        quality_warning=None, quality_reason=None, face_size_ratio=0.3,
        roll_deg=0.0, yaw_deg=None, pitch_deg=None,
        lateral_neutral_baseline=baseline, lateral_dynamic_raw=dyn, lateral_dynamic_filtered=dyn,
    )


def _build_session_with_one_cycle():
    """Sessao sintetica com um ciclo completo de abertura/fechamento e baseline lateral."""
    rec = SessionRecorder()
    cycles = CycleDetector()
    cycles.calibrate(0.0, 1.0, lateral_baseline=0.05)

    schedule = [
        (0.0, 0.0, 0.05), (0.1, 0.7, 0.10), (0.3, 0.9, 0.20), (0.6, 0.02, 0.02),
    ]
    for i, (t, opening, lateral) in enumerate(schedule):
        cycles.update(
            opening, t,
            lateral_absolute=lateral,
            lateral_dynamic=lateral - 0.05,
        )
        rec.add(_sample(i, t, opening, lateral, baseline=0.05))
    # Confirma o fechamento (close_stability_seconds=0.15s, banda <=5%): o
    # ciclo gravado usa t=0.6 (1o frame na banda fechada) como fim real; este
    # 5o update (0.2s depois) so confirma a estabilidade.
    cycles.update(0.02, 0.8, lateral_absolute=0.02, lateral_dynamic=0.02 - 0.05)
    return rec, cycles


# --------------------------------------------------------------------------
# CSV: novas colunas presentes e preenchidas corretamente
# --------------------------------------------------------------------------
def test_csv_export_includes_new_lateral_and_cycle_columns():
    rec, cycles = _build_session_with_one_cycle()
    assert cycles.repetitions == 1

    with tempfile.TemporaryDirectory() as d:
        paths = export_session(rec, cycles, d, "sessao_teste", mirrored=True)
        with open(paths["csv"], encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    header = rows[0].keys()
    for col in (
        "lateral_neutral_baseline", "lateral_absolute_raw", "lateral_absolute_filtered",
        "lateral_dynamic_raw", "lateral_dynamic_filtered", "anatomical_direction", "cycle_id",
    ):
        assert col in header, col

    assert all(r["lateral_neutral_baseline"] == "0.050000" for r in rows)
    # com o pre-buffer, o inicio exportado do ciclo recua ate a amostra
    # fechada em t=0.0 (nao o frame tardio do cruzamento do limiar em t=0.1):
    # todas as 4 amostras (t=0.0,0.1,0.3,0.6) ficam dentro do ciclo [0.0, 0.6].
    assert all(r["cycle_id"] == "1" for r in rows)


def test_resumo_includes_per_cycle_metrics_and_dynamic_extremes():
    rec, cycles = _build_session_with_one_cycle()
    with tempfile.TemporaryDirectory() as d:
        paths = export_session(rec, cycles, d, "sessao_teste", mirrored=True)
        with open(paths["resumo"], encoding="utf-8") as f:
            resumo = json.load(f)

    assert len(resumo["ciclos"]) == 1
    c0 = resumo["ciclos"][0]
    assert c0["cycle_id"] == 1
    assert "velocidade_media_abertura" in c0
    assert "direcao_predominante" in c0
    assert resumo["desvio_dinamico_max_direita"] is not None
    assert resumo["desvio_dinamico_max_esquerda"] is not None
    assert resumo["desvio_dinamico_medio_abs"] is not None


def test_metadados_includes_lateral_baseline_and_convention():
    rec, cycles = _build_session_with_one_cycle()
    with tempfile.TemporaryDirectory() as d:
        paths = export_session(rec, cycles, d, "sessao_teste", mirrored=False)
        with open(paths["metadados"], encoding="utf-8") as f:
            meta = json.load(f)

    assert meta["lateral_neutral_baseline"] == 0.05
    assert meta["mirrored"] is False
    assert "lateral_sign_convention" in meta
    assert "esquerda" in meta["lateral_sign_convention"] or "direita" in meta["lateral_sign_convention"]
    assert "unidade_metrica_relativa" in meta
    assert "interocular" in meta["unidade_metrica_relativa"]


def test_lateral_sign_convention_is_fixed_independent_of_mirrored():
    """
    Secao 7: a convencao anatomica ("positivo=direita anatomica; negativo=
    esquerda anatomica") deve ser a MESMA string nos metadados,
    independente de mirrored -- mirrored fica em campo separado.
    """
    rec, cycles = _build_session_with_one_cycle()
    metas = {}
    for mirrored in (True, False):
        with tempfile.TemporaryDirectory() as d:
            paths = export_session(rec, cycles, d, "sessao_teste", mirrored=mirrored)
            with open(paths["metadados"], encoding="utf-8") as f:
                metas[mirrored] = json.load(f)

    assert metas[True]["lateral_sign_convention"] == metas[False]["lateral_sign_convention"]
    assert metas[True]["lateral_sign_convention"] == LATERAL_SIGN_CONVENTION
    assert "mirrored" not in LATERAL_SIGN_CONVENTION.lower()
    assert metas[True]["mirrored"] is True
    assert metas[False]["mirrored"] is False


def test_signed_and_magnitude_fields_are_coherent():
    """
    Secao 7: movimento_max_direita/min_esquerda sao assinados (positivo=
    direita, negativo=esquerda, por definicao); magnitude_max_* sao sempre
    >=0 e iguais ao valor absoluto dos assinados.
    """
    rec, cycles = _build_session_with_one_cycle()
    with tempfile.TemporaryDirectory() as d:
        paths = export_session(rec, cycles, d, "sessao_teste", mirrored=True)
        with open(paths["resumo"], encoding="utf-8") as f:
            resumo = json.load(f)

    assert resumo["movimento_max_direita"] >= 0
    assert resumo["movimento_min_esquerda"] <= 0
    assert resumo["magnitude_max_direita"] >= 0
    assert resumo["magnitude_max_esquerda"] >= 0
    assert abs(resumo["magnitude_max_direita"] - max(resumo["movimento_max_direita"], 0.0)) < 1e-9
    assert abs(resumo["magnitude_max_esquerda"] - abs(min(resumo["movimento_min_esquerda"], 0.0))) < 1e-9
    # campos antigos preservados, mesmo valor das magnitudes
    assert resumo["desvio_dinamico_max_direita"] == resumo["magnitude_max_direita"]
    assert resumo["desvio_dinamico_max_esquerda"] == resumo["magnitude_max_esquerda"]


def test_predominant_direction_criteria_documented_in_resumo():
    """Secao 6: criterio, limiar e valor usado devem estar sempre reportados (auditavel)."""
    rec, cycles = _build_session_with_one_cycle()
    with tempfile.TemporaryDirectory() as d:
        paths = export_session(rec, cycles, d, "sessao_teste", mirrored=True)
        with open(paths["resumo"], encoding="utf-8") as f:
            resumo = json.load(f)

    assert resumo["criterio_direcao_predominante"] == "mediana_lateral_dynamic_plato_95pct"
    assert "descricao_criterio_direcao" in resumo
    assert "95%" in resumo["descricao_criterio_direcao"]
    assert resumo["limiar_direcao"] == cycles.effective_direction_deadzone
    assert resumo["effective_direction_deadzone"] == cycles.effective_direction_deadzone
    assert resumo["direction_deadzone_min"] == cycles.config.direction_deadzone_min
    c0 = resumo["ciclos"][0]
    assert "valor_usado_na_classificacao" in c0
    assert c0["valor_usado_na_classificacao"] == c0["lateral_dynamic_no_pico"]


def test_renamed_graph_titles_present_in_plotting_source():
    """
    Secao 5: verifica que os novos titulos/rotulos apresentados ao usuario
    estao de fato no codigo (nao apenas planejados) -- checagem direta do
    texto fonte das funcoes de plotagem renomeadas.
    """
    source = inspect.getsource(plotting)
    for expected in (
        "Posicao do queixo em relacao a linha media facial",
        "Movimento do queixo desde a posicao neutra calibrada",
        "Trajetoria: abertura x posicao do queixo",
        "Trajetoria: abertura x movimento desde o neutro",
        "Progresso normalizado do ciclo [%]",
        "Abertura relativa (adim.)",
        "Posicao lateral relativa (adim.)",
        "Movimento desde o neutro (adim.)",
        "Medidas adimensionais normalizadas pela distancia interocular.",
    ):
        assert expected in source, expected


def test_graphs_saved_with_bbox_inches_tight_to_avoid_clipped_labels():
    """
    Secao 6: todo grafico deve salvar com bbox_inches="tight" (garante que
    titulo/eixos/nota de rodape nao fiquem cortados). Verifica na fonte
    (funcao central `_save`, usada por todas as plot_*) em vez de inspecionar
    pixels do PNG.
    """
    source = inspect.getsource(plotting._save)
    assert 'bbox_inches="tight"' in source


def test_short_axis_labels_do_not_include_long_form_unit_text():
    """Os rotulos curtos nao devem reintroduzir o texto longo antigo, ja cortado."""
    assert plotting._opening_axis_label(False) == "Abertura relativa (adim.)"
    assert plotting._lateral_absolute_axis_label(False) == "Posicao lateral relativa (adim.)"
    assert plotting.LATERAL_DYNAMIC_AXIS_LABEL == "Movimento desde o neutro (adim.)"
    for label in (
        plotting._opening_axis_label(False),
        plotting._lateral_absolute_axis_label(False),
        plotting.LATERAL_DYNAMIC_AXIS_LABEL,
    ):
        assert len(label) <= 40


# --------------------------------------------------------------------------
# Graficos com ciclos: gerados, sem inventar zero
# --------------------------------------------------------------------------
def test_export_generates_cycle_plots_when_cycles_exist():
    rec, cycles = _build_session_with_one_cycle()
    with tempfile.TemporaryDirectory() as d:
        paths = export_session(rec, cycles, d, "sessao_teste", mirrored=True)
        assert "ciclos_png" in paths
        assert "ciclos_media_png" in paths
        assert os.path.exists(paths["ciclos_png"])
        assert os.path.getsize(paths["ciclos_png"]) > 0
        assert os.path.exists(paths["ciclos_media_png"])


def test_export_skips_cycle_plots_when_no_cycles():
    rec = SessionRecorder()
    cycles = CycleDetector()
    rec.add(_sample(0, 0.0, 0.1, 0.0, baseline=None))
    with tempfile.TemporaryDirectory() as d:
        paths = export_session(rec, cycles, d, "sessao_teste", mirrored=True)
        assert "ciclos_png" not in paths
        assert "ciclos_media_png" not in paths


# --------------------------------------------------------------------------
# assign_cycle_ids: amostras fora de qualquer ciclo completo ficam sem id
# --------------------------------------------------------------------------
def test_assign_cycle_ids_leaves_out_of_cycle_samples_unset():
    rec, cycles = _build_session_with_one_cycle()
    extra = _sample(99, 10.0, 0.05, 0.05, baseline=0.05)  # muito depois do ciclo
    rec.add(extra)
    assign_cycle_ids(rec.samples, cycles.cycles)
    assert extra.cycle_id is None  # t=10.0, muito depois do fim do ciclo (t=0.6)
    assert rec.samples[0].cycle_id == 1  # t=0.0, inicio exportado (recuperado do pre-buffer)
    assert rec.samples[1].cycle_id == 1  # t=0.1, dentro do ciclo [0.0, 0.6]


# --------------------------------------------------------------------------
# Compatibilidade com sessao antiga (sem lateral_dynamic_filtered)
# --------------------------------------------------------------------------
def test_compare_sessions_handles_old_session_without_dynamic_column():
    import compare_sessions

    old_columns = [
        "frame", "tempo_s", "abertura_relativa", "abertura_filtrada",
        "desvio_lateral_relativo", "desvio_lateral_filtrado", "estado_ciclo", "repeticoes",
    ]
    with tempfile.TemporaryDirectory() as d:
        old_path = os.path.join(d, "antiga", "dados.csv")
        os.makedirs(os.path.dirname(old_path), exist_ok=True)
        with open(old_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(old_columns)
            for i in range(5):
                w.writerow([i, f"{0.1*i:.4f}", "0.3", "0.3", "0.1", "0.1", "fechado", "0"])

        new_path = os.path.join(d, "nova", "dados.csv")
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        with open(new_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(old_columns + ["lateral_dynamic_filtered"])
            for i in range(5):
                w.writerow([i, f"{0.1*i:.4f}", "0.3", "0.3", "0.1", "0.1", "fechado", "0", "0.05"])

        old = compare_sessions.load_session(old_path)
        new = compare_sessions.load_session(new_path)

        assert old.has_dynamic is False
        assert new.has_dynamic is True

        stats_old = compare_sessions.session_stats(old)
        stats_new = compare_sessions.session_stats(new)
        assert stats_old["lateral_dinamico_max_abs"] is None  # nao inventa valor
        assert stats_new["lateral_dinamico_max_abs"] is not None

        # A comparacao mista (uma com, outra sem) deve cair para o absoluto
        # para AMBAS, sem lancar excecao.
        out = compare_sessions.plot_comparison([old, new], os.path.join(d, "cmp.png"), False)
        assert os.path.exists(out)
