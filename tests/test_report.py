"""
Testes da leitura clinica da sessao: analise medida a medida, persistencia da
analise frontal e do biofeedback no CSV, e geracao dos relatorios HTML/PDF.

O foco e nas regras de veracidade do projeto:
    - sem calibracao em mm, NAO se compara com a faixa clinica;
    - onde a literatura nao da corte, o relatorio diz isso explicitamente;
    - medida nao coletada nao vira medida com valor inventado.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mandibular.config import CycleConfig  # noqa: E402
from mandibular.exporter import export_session  # noqa: E402
from mandibular.metrics import CycleDetector  # noqa: E402
from mandibular.recorder import CSV_COLUMNS, Sample, SessionRecorder  # noqa: E402
from mandibular.report import (  # noqa: E402
    SEM_CORTE,
    build_findings,
    build_report_data,
)


def _sessao(ref_mm: float | None = 63.0, com_frontal: bool = True,
            n: int = 300) -> tuple[SessionRecorder, CycleDetector]:
    """Sessao sintetica com ciclos de abertura bem definidos."""
    rec = SessionRecorder()
    cyc = CycleDetector(CycleConfig())
    cyc.calibrate(0.05, 0.50)
    for i in range(n):
        t = i / 30.0
        op = 0.05 + 0.40 * max(0.0, math.sin(2 * math.pi * t / 3.0))
        cyc.update(op, t)
        lat = 0.02
        rec.add(Sample(
            session_id="s", frame=i, timestamp="t", time_s=t,
            face_detected=True, frame_valid=True,
            opening_raw=op * 200, opening_rel=op, opening_filtered=op,
            opening_mm=(op * 120 if ref_mm else None),
            lateral_raw=4.0, lateral_rel=lat, lateral_filtered=lat,
            lateral_mm=(lat * 120 if ref_mm else None),
            direction="direita", cycle_state=cyc.state,
            repetitions=cyc.repetitions, quality_warning=None,
            symmetry_index=(0.92 if com_frontal else None),
            midline_offset_rel=(0.003 if com_frontal else None),
            cant_deg=(1.5 if com_frontal else None),
            mand_deviation_deg=(4.0 if com_frontal else None),
            is_frontal=(True if com_frontal else None),
            biofeedback=("Desvio para a direita" if i % 2 == 0 else None),
        ))
    return rec, cyc


# -- Persistencia no CSV -----------------------------------------------------
def test_csv_tem_colunas_da_analise_frontal_e_biofeedback():
    for coluna in ("simetria_indice", "simetria_linha_media", "cant_graus",
                   "desvio_mandibular_graus", "frontal", "biofeedback"):
        assert coluna in CSV_COLUMNS

    rec, _ = _sessao()
    with tempfile.TemporaryDirectory() as d:
        caminho = os.path.join(d, "dados.csv")
        rec.to_csv(caminho)
        linhas = list(csv.DictReader(open(caminho, encoding="utf-8")))
    assert linhas[0]["simetria_indice"] == "0.9200"
    assert linhas[0]["desvio_mandibular_graus"] == "4.00"
    assert linhas[0]["frontal"] == "1"
    assert linhas[0]["biofeedback"] == "Desvio para a direita"


def test_frame_sem_analise_frontal_grava_vazio_e_nao_zero():
    """Medida ausente nao pode virar 0.0 -- seria fabricar um dado."""
    rec, _ = _sessao(com_frontal=False)
    with tempfile.TemporaryDirectory() as d:
        caminho = os.path.join(d, "dados.csv")
        rec.to_csv(caminho)
        linha = next(iter(csv.DictReader(open(caminho, encoding="utf-8"))))
    assert linha["simetria_indice"] == ""
    assert linha["desvio_mandibular_graus"] == ""
    assert linha["frontal"] == ""


# -- Agregacao ---------------------------------------------------------------
def test_agregacao_da_analise_frontal_e_do_biofeedback():
    rec, cyc = _sessao()
    dados = build_report_data(rec, cyc, 63.0)
    assert abs(dados["simetria_media"] - 0.92) < 1e-6
    assert abs(dados["desvio_mand_medio_deg"] - 4.0) < 1e-6
    assert abs(dados["cant_medio_deg"] - 1.5) < 1e-6
    assert dados["pct_frontal"] == 100.0
    assert dados["biofeedback"]["Desvio para a direita"] > 0


def test_agregacao_sem_analise_frontal_fica_none():
    rec, cyc = _sessao(com_frontal=False)
    dados = build_report_data(rec, cyc, 63.0)
    assert dados["simetria_media"] is None
    assert dados["desvio_mand_medio_deg"] is None
    assert dados["pct_frontal"] is None


# -- Analise medida a medida -------------------------------------------------
def test_findings_com_calibracao_comparam_com_a_faixa_de_referencia():
    rec, cyc = _sessao(ref_mm=63.0)
    medidas = {m.nome: m for m in build_findings(build_report_data(rec, cyc, 63.0))}

    abertura = medidas["Abertura máxima da boca"]
    assert "mm" in abertura.valor
    assert "40–60 mm" in abertura.referencia and "Dufour" in abertura.referencia

    # Toda medida analisada precisa dizer contra o que esta sendo comparada.
    for m in medidas.values():
        assert m.referencia.strip(), f"{m.nome} sem referencia declarada"
        assert m.leitura.strip(), f"{m.nome} sem leitura"
        assert m.nivel in ("ok", "info", "atencao")


def test_findings_sem_calibracao_nao_classificam_em_mm():
    """Sem ref_mm nao pode haver comparacao com a faixa clinica."""
    rec, cyc = _sessao(ref_mm=None)
    medidas = {m.nome: m for m in build_findings(build_report_data(rec, cyc, None))}

    abertura = medidas["Abertura máxima da boca"]
    assert "mm" not in abertura.valor
    assert "Não aplicável sem calibração" in abertura.referencia
    assert abertura.nivel == "atencao"
    assert "40" not in abertura.referencia


def test_simetria_declara_ausencia_de_corte_clinico():
    rec, cyc = _sessao()
    medidas = {m.nome: m for m in build_findings(build_report_data(rec, cyc, 63.0))}
    for nome in ("Simetria facial", "Desvio da linha média mandibular"):
        assert SEM_CORTE in medidas[nome].referencia


def test_grupos_das_medidas():
    rec, cyc = _sessao()
    grupos = {m.grupo for m in build_findings(build_report_data(rec, cyc, 63.0))}
    assert grupos == {"movimento", "frontal", "captura"}


# -- Exportacao completa -----------------------------------------------------
def test_exportacao_gera_html_e_pdf():
    rec, cyc = _sessao()
    with tempfile.TemporaryDirectory() as d:
        paths = export_session(rec, cyc, os.path.join(d, "s"), "s", ref_mm=63.0,
                               paciente="Teste", history_dir=d)
        assert os.path.getsize(paths["relatorio"]) > 2000
        assert os.path.getsize(paths["relatorio_pdf"]) > 5000
        with open(paths["relatorio_pdf"], "rb") as f:
            assert f.read(5) == b"%PDF-"
        html = open(paths["relatorio"], encoding="utf-8").read()
        assert "Análise facial frontal" in html
        assert "Biofeedback durante a sessão" in html
        # O historico do paciente so funciona porque a simetria agora e gravada.
        historico = list(csv.DictReader(open(paths["historico"], encoding="utf-8")))
        assert historico[0]["simetria_media"] != ""
        assert historico[0]["desvio_mandib_medio_deg"] != ""


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    falhas = 0
    for fn in tests:
        try:
            fn()
            print(f"OK  {fn.__name__}")
        except AssertionError as e:
            falhas += 1
            print(f"FALHOU  {fn.__name__}: {e}")
    print(f"\n{len(tests) - falhas}/{len(tests)} testes passaram.")
    sys.exit(1 if falhas else 0)
