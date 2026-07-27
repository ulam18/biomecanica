"""
Leitura da sessao em linguagem clinica: agregacao dos numeros, analise medida a
medida e geracao do relatorio HTML.

Divisao de responsabilidades:
    build_report_data  -> agrega os numeros crus da sessao (sem formatacao);
    build_findings     -> analisa CADA medida: valor, referencia e leitura;
    write_report       -> renderiza o HTML.
    (pdf_report.py     -> renderiza o mesmo `build_findings` em PDF)

O HTML e o PDF consomem a MESMA lista de `Medida`. Sem isso, os dois documentos
sobre o mesmo paciente poderiam divergir silenciosamente -- o mesmo problema que
`pipeline.py` resolve entre o modo ao vivo e a analise offline.

Regras de veracidade herdadas de `classification.py`:
    - so ha comparacao com faixa de referencia quando existe fonte citavel
      (abertura 40-60 mm e diducao 9-12 mm, Dufour & Pillu p. 553);
    - onde a literatura NAO fornece corte (simetria, desvio do mento, angulo
      mandibular), isso e dito explicitamente e a medida e apresentada como
      comparacao do paciente com ele mesmo;
    - sem calibracao em milimetros, o relatorio declara a limitacao em vez de
      estimar um valor clinico.
"""

from __future__ import annotations

import html
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .classification import (
    ABERTURA_NORMAL_MM,
    DIDUCAO_NORMAL_MM,
    classify_chin_deviation,
    classify_diduction,
    classify_opening,
    classify_symmetry,
)
from .metrics import CycleDetector
from .recorder import SessionRecorder

DISCLAIMER = (
    "Este relatório é uma ferramenta de APOIO e documentação. Os valores são "
    "estimativas obtidas por câmera comum e não substituem o exame clínico, "
    "a medição direta nem o diagnóstico profissional."
)

SEM_CORTE = "Sem corte clínico na literatura consultada"
FONTE_DUFOUR = "Dufour & Pillu, Biomecânica Funcional, p. 553"

# Coeficiente de variacao da amplitude entre ciclos -> leitura qualitativa.
# Nao ha corte clinico para isso; as faixas servem para comparacao do paciente
# com ele mesmo entre sessoes.
CV_ALTA_CONSISTENCIA = 0.15
CV_MEDIA_CONSISTENCIA = 0.35

# Percentual minimo de quadros frontais para que a simetria seja interpretavel.
PCT_FRONTAL_CONFIAVEL = 80.0


@dataclass
class Medida:
    """
    Uma medida analisada: o que foi medido, contra o que se compara e o que se
    lê disso. E a unidade compartilhada entre o relatorio HTML e o PDF.
    """
    grupo: str        # "movimento" | "frontal" | "captura"
    nome: str
    valor: str        # ja formatado, com unidade
    referencia: str   # faixa + fonte, ou SEM_CORTE
    leitura: str      # interpretacao em linguagem clinica
    nivel: str        # "ok" | "info" | "atencao"


def _fmt(value: float | None, casas: int = 1, sufixo: str = "") -> str:
    """Formata um numero no padrao brasileiro (virgula decimal)."""
    if value is None:
        return "—"
    return f"{value:.{casas}f}{sufixo}".replace(".", ",")


def _consistency_text(cv: float | None) -> tuple[str, str]:
    """Traduz o coeficiente de variacao da amplitude em texto + nivel."""
    if cv is None:
        return ("não avaliada (menos de 2 repetições completas)", "info")
    if cv <= CV_ALTA_CONSISTENCIA:
        return (f"alta — as repetições variaram {cv * 100:.0f}% entre si", "ok")
    if cv <= CV_MEDIA_CONSISTENCIA:
        return (f"média — as repetições variaram {cv * 100:.0f}% entre si", "info")
    return (f"baixa — as repetições variaram {cv * 100:.0f}% entre si", "atencao")


# ===========================================================================
# 1. Agregacao dos numeros da sessao
# ===========================================================================
def build_report_data(
    recorder: SessionRecorder,
    cycles: CycleDetector,
    ref_mm: float | None,
) -> dict:
    """Agrega os numeros da sessao usados no relatorio (sem formatacao)."""
    samples = recorder.samples
    validos = [s for s in samples if s.frame_valid]

    op_rel = [s.opening_rel for s in validos if s.opening_rel is not None]
    op_mm = [s.opening_mm for s in validos if s.opening_mm is not None]
    lat_rel = [s.lateral_rel for s in validos if s.lateral_rel is not None]
    lat_mm = [s.lateral_mm for s in validos if s.lateral_mm is not None]

    duracao = (samples[-1].time_s - samples[0].time_s) if len(samples) > 1 else 0.0
    rep = cycles.repeatability() or {}

    # -- Analise facial frontal (so quadros validos, so o que foi medido) ----
    def col(attr: str) -> list[float]:
        return [v for v in (getattr(s, attr, None) for s in validos) if v is not None]

    simetria = col("symmetry_index")
    mand = col("mand_deviation_deg")
    cant = col("cant_deg")
    frontais = [s.is_frontal for s in validos if s.is_frontal is not None]

    # Biofeedback: com que frequencia cada mensagem foi mostrada na sessao.
    avisos = Counter(
        msg for s in samples if s.biofeedback for msg in s.biofeedback.split(" | ")
    )

    # Desvio do queixo com a boca aberta vs fechada: a literatura descreve o
    # desvio do CAMINHO de abertura, entao a comparacao entre os dois estados
    # diz mais que a media da sessao inteira.
    def lat_no_estado(*estados: str) -> float | None:
        vals = [
            s.lateral_filtered
            for s in validos
            if s.lateral_filtered is not None and s.cycle_state.value in estados
        ]
        return float(np.mean(vals)) if vals else None

    return {
        "total_frames": len(samples),
        "frames_validos": len(validos),
        "percentual_valido": (100.0 * len(validos) / len(samples)) if samples else 0.0,
        "duracao_s": duracao,
        "calibrado": cycles.is_calibrated,
        "ref_mm": ref_mm,
        "abertura_max_rel": (max(op_rel) if op_rel else None),
        "abertura_max_mm": (max(op_mm) if op_mm else None),
        "desvio_medio_rel": (float(np.mean(lat_rel)) if lat_rel else None),
        "desvio_medio_mm": (float(np.mean(lat_mm)) if lat_mm else None),
        "amplitude_lateral_mm": ((max(lat_mm) - min(lat_mm)) if len(lat_mm) > 1 else None),
        "amplitude_lateral_rel": ((max(lat_rel) - min(lat_rel)) if len(lat_rel) > 1 else None),
        "repeticoes": cycles.repetitions,
        "amplitude_cv": rep.get("amplitude_cv"),
        "duracao_media_ciclo_s": rep.get("duracao_media_s"),
        "simetria_media": (float(np.mean(simetria)) if simetria else None),
        "simetria_min": (float(np.min(simetria)) if simetria else None),
        "desvio_mand_medio_deg": (float(np.mean(mand)) if mand else None),
        "desvio_mand_absmax_deg": (float(np.max(np.abs(mand))) if mand else None),
        "cant_medio_deg": (float(np.mean(cant)) if cant else None),
        "pct_frontal": ((100.0 * sum(frontais) / len(frontais)) if frontais else None),
        "desvio_boca_aberta_rel": lat_no_estado("aberto"),
        "desvio_boca_fechada_rel": lat_no_estado("fechado"),
        "biofeedback": dict(avisos.most_common()),
    }


# ===========================================================================
# 2. Analise medida a medida
# ===========================================================================
def build_findings(dados: dict) -> list[Medida]:
    """
    Analisa cada medida da sessao e devolve a lista usada pelo HTML e pelo PDF.

    Uma medida so aparece se foi efetivamente medida; nada e preenchido por
    suposicao. Quando falta a calibracao em milimetros, a medida entra assim
    mesmo, mas com a limitacao declarada no lugar da faixa de referencia.
    """
    ms: list[Medida] = []
    cal = dados["calibrado"]

    # -- Abertura bucal ------------------------------------------------------
    if dados["abertura_max_mm"] is not None:
        cls = classify_opening(dados["abertura_max_mm"])
        ms.append(Medida(
            "movimento", "Abertura máxima da boca",
            _fmt(dados["abertura_max_mm"], 1, " mm"),
            f"{ABERTURA_NORMAL_MM[0]:.0f}–{ABERTURA_NORMAL_MM[1]:.0f} mm "
            f"({FONTE_DUFOUR})",
            f"Abertura {cls.category}. Valor estimado por câmera, não medido "
            "diretamente com régua.",
            {"ok": "ok", "atencao": "atencao"}.get(cls.level, "info"),
        ))
    elif dados["abertura_max_rel"] is not None:
        ms.append(Medida(
            "movimento", "Abertura máxima da boca",
            _fmt(dados["abertura_max_rel"], 3) + " (proporção do rosto)",
            "Não aplicável sem calibração em milímetros",
            "Não é possível comparar com a faixa de referência: a sessão foi "
            "feita sem informar a distância entre os cantos externos dos olhos. "
            "Meça o paciente com régua e repita para obter o valor em mm.",
            "atencao",
        ))

    # -- Diducao (movimento lateral) ----------------------------------------
    if dados["amplitude_lateral_mm"] is not None:
        cls = classify_diduction(dados["amplitude_lateral_mm"])
        ms.append(Medida(
            "movimento", "Movimento lateral (didução)",
            _fmt(dados["amplitude_lateral_mm"], 1, " mm"),
            f"{DIDUCAO_NORMAL_MM[0]:.0f}–{DIDUCAO_NORMAL_MM[1]:.0f} mm "
            f"({FONTE_DUFOUR})",
            f"Amplitude total de um lado ao outro: {cls.category}. Se o paciente "
            "não foi orientado a lateralizar a mandíbula durante a coleta, este "
            "valor reflete apenas o desvio espontâneo, não a didução máxima.",
            {"ok": "ok", "atencao": "atencao"}.get(cls.level, "info"),
        ))
    elif dados["amplitude_lateral_rel"] is not None:
        ms.append(Medida(
            "movimento", "Movimento lateral (didução)",
            _fmt(dados["amplitude_lateral_rel"], 3) + " (proporção do rosto)",
            "Não aplicável sem calibração em milímetros",
            "Amplitude total registrada de um lado ao outro. Informe a medida "
            "entre os olhos para comparar com a faixa de referência.",
            "info",
        ))

    # -- Desvio do mento -----------------------------------------------------
    if dados["desvio_medio_mm"] is not None:
        cls = classify_chin_deviation(dados["desvio_medio_mm"])
        ms.append(Medida(
            "movimento", "Desvio médio do queixo",
            _fmt(dados["desvio_medio_mm"], 1, " mm"),
            SEM_CORTE + " (6–7 mm relatados em casos cirúrgicos severos, "
            "Carlini & Gomes)",
            f"Deslocamento {cls.category} em relação à linha média facial. "
            "Serve para acompanhar o próprio paciente entre sessões, não para "
            "classificar como normal ou alterado.",
            "info",
        ))
    elif dados["desvio_medio_rel"] is not None:
        lado = "direita" if dados["desvio_medio_rel"] >= 0 else "esquerda"
        ms.append(Medida(
            "movimento", "Desvio médio do queixo",
            _fmt(dados["desvio_medio_rel"], 3) + " (proporção do rosto)",
            SEM_CORTE,
            f"Deslocamento médio para a {lado} em relação à linha média facial. "
            "Serve para acompanhar o próprio paciente entre sessões.",
            "info",
        ))

    # -- Repeticoes e repetibilidade ----------------------------------------
    consist_txt, consist_nivel = _consistency_text(dados["amplitude_cv"])
    ms.append(Medida(
        "movimento", "Repetições e consistência",
        f"{dados['repeticoes']} repetições",
        SEM_CORTE + " (comparação intrassujeito)",
        f"Consistência entre as repetições: {consist_txt}."
        + ("" if cal else " A sessão não foi calibrada, o que torna a contagem "
           "menos confiável."),
        consist_nivel,
    ))

    if dados["duracao_media_ciclo_s"] is not None:
        ms.append(Medida(
            "movimento", "Duração média do ciclo",
            _fmt(dados["duracao_media_ciclo_s"], 2, " s"),
            SEM_CORTE,
            "Tempo médio de um ciclo completo de abertura e fechamento.",
            "info",
        ))

    # -- Analise facial frontal ---------------------------------------------
    if dados["simetria_media"] is not None:
        cls = classify_symmetry(dados["simetria_media"])
        ms.append(Medida(
            "frontal", "Simetria facial",
            f"{dados['simetria_media'] * 100:.0f}%",
            SEM_CORTE + " — faixa didática, comparação intrassujeito",
            f"Índice de simetria {cls.category} entre os pares esquerda/direita "
            "(olhos, nariz, boca, contorno). A literatura consultada afirma que "
            "toda face é assimétrica e não fornece limiar que separe o "
            "fisiológico do patológico.",
            "info",
        ))

    if dados["desvio_mand_medio_deg"] is not None:
        graus = dados["desvio_mand_medio_deg"]
        lado = "direita" if graus >= 0 else "esquerda"
        ms.append(Medida(
            "frontal", "Desvio da linha média mandibular",
            _fmt(graus, 1, "°"),
            SEM_CORTE,
            f"Inclinação média da linha násio→mento para a {lado}, sendo 0° o "
            f"mento alinhado (máximo observado: {_fmt(dados['desvio_mand_absmax_deg'], 1, '°')}). "
            "É a versão angular do desvio do queixo, menos sensível à distância "
            "da câmera que a medida linear.",
            "info",
        ))

    if dados["cant_medio_deg"] is not None:
        ms.append(Medida(
            "frontal", "Inclinação olhos × boca (cant)",
            _fmt(dados["cant_medio_deg"], 1, "°"),
            SEM_CORTE + " — canon estético, não limiar clínico",
            "Ângulo entre a linha dos olhos e a linha dos cantos da boca. Em uma "
            "face frontal simétrica as duas linhas são paralelas (0°).",
            "info",
        ))

    ab, fe = dados["desvio_boca_aberta_rel"], dados["desvio_boca_fechada_rel"]
    if ab is not None and fe is not None:
        sentido = "aumenta" if abs(ab) > abs(fe) else "diminui"
        ms.append(Medida(
            "frontal", "Desvio ao abrir a boca",
            f"{_fmt(fe, 3)} → {_fmt(ab, 3)}",
            SEM_CORTE,
            f"O desvio do queixo {sentido} quando a boca abre (valores em "
            "proporção da largura facial, com a boca fechada e aberta). A "
            "literatura descreve o desvio do caminho de abertura, então esta "
            "comparação é mais informativa que a média da sessão inteira.",
            "info",
        ))

    # -- Condicoes da captura ------------------------------------------------
    pct = dados["percentual_valido"]
    if pct >= 80:
        q_nivel, q_txt = "ok", "boas"
    elif pct >= 50:
        q_nivel, q_txt = "info", "aceitáveis"
    else:
        q_nivel, q_txt = "atencao", "ruins"
    ms.append(Medida(
        "captura", "Qualidade da captura",
        f"{pct:.0f}% dos quadros válidos",
        "Heurísticas de engenharia (não clínicas)",
        f"Condições de captura {q_txt}. Quadros inválidos ocorrem com pouca luz, "
        "rosto muito distante, cabeça inclinada ou movimento brusco."
        + ("" if pct >= 50 else " Recomenda-se repetir a coleta com melhor "
           "iluminação e o rosto centralizado."),
        q_nivel,
    ))

    if dados["pct_frontal"] is not None:
        pf = dados["pct_frontal"]
        confiavel = pf >= PCT_FRONTAL_CONFIAVEL
        ms.append(Medida(
            "captura", "Enquadramento frontal",
            f"{pf:.0f}% do tempo",
            f"Mínimo recomendado: {PCT_FRONTAL_CONFIAVEL:.0f}%",
            "Percentual de quadros em que a cabeça esteve suficientemente de "
            "frente para a câmera. A simetria só é interpretável na vista frontal."
            + ("" if confiavel else " Percentual baixo: leia as medidas de "
               "simetria desta sessão com reserva."),
            "ok" if confiavel else "atencao",
        ))

    return ms


# ===========================================================================
# 3. Renderizacao HTML
# ===========================================================================
def _card(m: Medida) -> str:
    return (
        f'<div class="card {m.nivel}">'
        f'<div class="card-titulo">{html.escape(m.nome)}</div>'
        f'<div class="card-valor">{html.escape(m.valor)}</div>'
        f'<div class="card-detalhe">{html.escape(m.leitura)}</div>'
        f'<div class="card-ref">Referência: {html.escape(m.referencia)}</div>'
        f"</div>"
    )


def _grupo_html(medidas: list[Medida], grupo: str, titulo: str) -> str:
    doc = [m for m in medidas if m.grupo == grupo]
    if not doc:
        return ""
    return f"<h2>{titulo}</h2><div class=\"cards\">{''.join(_card(m) for m in doc)}</div>"


def write_report(
    path: str,
    dados: dict,
    session_id: str,
    paciente: str = "",
    figuras: dict[str, str] | None = None,
    observacoes: str = "",
) -> str:
    """
    Escreve o relatorio HTML da sessao e retorna o caminho do arquivo.

    `figuras` mapeia rotulo -> caminho de imagem (relativo a pasta do relatorio).
    """
    figuras = figuras or {}
    medidas = build_findings(dados)

    figs_html = ""
    for rotulo, arq in figuras.items():
        if arq:
            figs_html += (
                f'<figure><img src="{html.escape(os.path.basename(arq))}" '
                f'alt="{html.escape(rotulo)}"><figcaption>{html.escape(rotulo)}'
                f"</figcaption></figure>"
            )

    # -- Biofeedback: o que o paciente viu na tela durante a sessao ----------
    bio = dados.get("biofeedback") or {}
    total = max(dados["total_frames"], 1)
    if bio:
        linhas = "".join(
            f"<li>{html.escape(msg)} &mdash; <b>{100.0 * n / total:.0f}%</b> do tempo</li>"
            for msg, n in bio.items()
        )
        bio_corpo = (
            '<p class="card-detalhe">Mensagens exibidas ao paciente na tela '
            "enquanto executava o movimento, com a fração do tempo em que cada "
            "uma esteve ativa.</p>"
            f'<ul class="bio">{linhas}</ul>'
        )
    else:
        bio_corpo = (
            '<p class="card-detalhe">Nenhuma mensagem de biofeedback foi '
            "disparada: o movimento permaneceu dentro da faixa treinada, sem "
            "desvio acima da zona morta nem inconsistência entre repetições.</p>"
        )

    obs_html = ""
    if observacoes.strip():
        obs_html = (
            '<h2>Observações</h2><p class="obs">'
            + html.escape(observacoes).replace("\n", "<br>")
            + "</p>"
        )

    tem_mm = dados["ref_mm"] is not None
    calib_txt = (
        f"Calibração de escala: {dados['ref_mm']:.0f} mm entre os cantos externos "
        "dos olhos (medidos pelo profissional)."
        if tem_mm else
        "Sem calibração em milímetros: os valores estão em medida relativa "
        "(proporção da largura do rosto)."
    )
    calib_mov = (
        "Faixa de movimento calibrada no início da sessão (boca fechada &rarr; "
        "boca aberta)."
        if dados["calibrado"] else
        "A faixa de movimento NÃO foi calibrada nesta sessão; a contagem de "
        "repetições usou os valores mínimo e máximo observados."
    )

    doc = f"""<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>Relatório da sessão &mdash; {html.escape(paciente or session_id)}</title>
<style>
  :root {{ --ok:#1b7f3b; --info:#1f4e79; --atencao:#a35a00; --linha:#dcdcdc; }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: "Segoe UI", Arial, sans-serif; color:#222; margin:0;
         background:#f4f5f7; }}
  .pagina {{ max-width: 900px; margin: 0 auto; background:#fff; padding: 32px 40px;
             min-height:100vh; }}
  h1 {{ font-size: 24px; margin: 0 0 4px; }}
  h2 {{ font-size: 18px; margin: 32px 0 12px; border-bottom:2px solid var(--linha);
        padding-bottom:6px; }}
  .sub {{ color:#666; margin: 0 0 24px; font-size: 14px; }}
  .ident td {{ padding: 4px 12px 4px 0; font-size: 14px; }}
  .ident td:first-child {{ color:#666; }}
  .cards {{ display:flex; flex-wrap:wrap; gap:14px; }}
  .card {{ flex:1 1 260px; border:1px solid var(--linha); border-left-width:6px;
           border-radius:6px; padding:14px 16px; background:#fbfbfb; }}
  .card.ok {{ border-left-color: var(--ok); }}
  .card.info {{ border-left-color: var(--info); }}
  .card.atencao {{ border-left-color: var(--atencao); background:#fff8ee; }}
  .card-titulo {{ font-size:13px; color:#555; text-transform:uppercase;
                  letter-spacing:.4px; }}
  .card-valor {{ font-size:28px; font-weight:600; margin:6px 0; }}
  .card-detalhe {{ font-size:13px; color:#444; line-height:1.45; }}
  .card-ref {{ font-size:12px; color:#777; margin-top:8px; padding-top:8px;
               border-top:1px dashed var(--linha); }}
  figure {{ margin: 18px 0; }}
  figure img {{ width:100%; border:1px solid var(--linha); border-radius:4px; }}
  figcaption {{ font-size:13px; color:#666; margin-top:6px; }}
  .aviso {{ margin-top:32px; padding:14px 16px; border:1px solid #e0c9a0;
            background:#fff8ee; border-radius:6px; font-size:14px; line-height:1.5; }}
  .obs {{ font-size:14px; line-height:1.6; white-space:pre-wrap; }}
  ul.bio {{ font-size:14px; line-height:1.7; margin:8px 0 0 18px; padding:0; }}
  .rodape {{ margin-top:28px; font-size:12px; color:#888; }}
  .imprimir {{ margin: 20px 0; }}
  button {{ font-size:15px; padding:10px 18px; border-radius:6px; border:0;
            background:#1f4e79; color:#fff; cursor:pointer; }}
  @media print {{
    body {{ background:#fff; }} .pagina {{ padding:0; }} .imprimir {{ display:none; }}
  }}
</style>
</head>
<body>
<div class="pagina">
  <h1>Relatório da avaliação mandibular</h1>
  <p class="sub">Documento de apoio gerado automaticamente pelo sistema.</p>

  <table class="ident">
    <tr><td>Paciente</td><td><b>{html.escape(paciente) if paciente else "não informado"}</b></td></tr>
    <tr><td>Data da sessão</td><td>{datetime.now().strftime("%d/%m/%Y às %H:%M")}</td></tr>
    <tr><td>Identificador</td><td>{html.escape(session_id)}</td></tr>
    <tr><td>Duração</td><td>{_fmt(dados["duracao_s"], 0, " segundos")}</td></tr>
  </table>

  <div class="imprimir"><button onclick="window.print()">Imprimir / salvar em PDF</button></div>

  {_grupo_html(medidas, "movimento", "Movimento mandibular")}

  {_grupo_html(medidas, "frontal", "Análise facial frontal")}

  <h2>Biofeedback durante a sessão</h2>
  {bio_corpo}

  {_grupo_html(medidas, "captura", "Condições da coleta")}
  <p class="card-detalhe" style="margin-top:12px">{calib_txt}<br>{calib_mov}</p>

  {"<h2>Gráficos</h2>" + figs_html if figs_html else ""}

  {obs_html}

  <div class="aviso"><b>Importante.</b> {DISCLAIMER}</div>
  <p class="rodape">Sistema de reconhecimento mandibular digital &mdash;
     arquivos técnicos completos (dados.csv, resumo.json) na mesma pasta.</p>
</div>
</body>
</html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path
