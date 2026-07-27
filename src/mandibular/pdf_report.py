"""
Relatorio da sessao em PDF (A4), para anexar ao prontuario do paciente.

Renderiza a MESMA lista de `Medida` produzida por `report.build_findings`, de
modo que o PDF e o HTML nunca divirjam sobre o mesmo paciente.

Usa matplotlib (ja exigido pelos graficos) em vez de uma biblioteca de PDF
dedicada: evita mais uma dependencia na instalacao feita pelo profissional, que
e justamente a etapa mais fragil para quem nao tem familiaridade com TI.

O texto flui em paginas A4 com quebra automatica (`_Doc`): a quantidade de
medidas varia conforme a sessao (com ou sem calibracao, com ou sem analise
frontal), entao um layout de posicoes fixas transbordaria o rodape.
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # sem janela: o PDF e gerado em segundo plano

import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from .report import DISCLAIMER, Medida, _fmt, build_findings  # noqa: E402

A4 = (8.27, 11.69)       # polegadas, retrato
FONTE = "DejaVu Sans"    # fonte embutida no matplotlib, com acentuacao completa

MARGEM_X = 0.075         # fracao da largura
TOPO = 0.95              # y inicial do conteudo
BASE = 0.075             # y minimo: abaixo disso, quebra a pagina
LINHA = 0.0145           # avanco vertical de uma linha de texto

COR_TITULO = "#1f4e79"
COR_TEXTO = "#222222"
COR_SUAVE = "#666666"
NIVEL_COR = {"ok": "#1b7f3b", "info": "#1f4e79", "atencao": "#a35a00"}

GRUPOS = [
    ("movimento", "Movimento mandibular"),
    ("frontal", "Análise facial frontal"),
    ("captura", "Condições da coleta"),
]


def _largura(tamanho: float) -> int:
    """
    Quantos caracteres cabem em uma linha, para o corpo de texto util da A4.

    Aproximacao empirica calibrada na DejaVu Sans: passar de ~760/pt faz a
    linha ultrapassar a margem direita.
    """
    return max(40, int(760 / tamanho))


class _Doc:
    """Documento A4 com fluxo de texto e quebra de pagina automatica."""

    def __init__(self, paciente: str) -> None:
        self.paciente = paciente
        self.figs: list[plt.Figure] = []
        self.fig: plt.Figure | None = None
        self.y = TOPO
        self.nova_pagina()

    # -- Paginas -------------------------------------------------------------
    def nova_pagina(self) -> None:
        fig = plt.figure(figsize=A4)
        fig.patch.set_facecolor("white")
        self.figs.append(fig)
        self.fig = fig
        self.y = TOPO

    def espaco(self, altura: float) -> None:
        """Garante `altura` disponivel; quebra a pagina se nao houver."""
        if self.y - altura < BASE:
            self.nova_pagina()

    # -- Elementos -----------------------------------------------------------
    def texto(self, txt: str, *, tamanho=10, cor=COR_TEXTO, negrito=False,
              x=MARGEM_X, avanco=LINHA, ha="left") -> None:
        px = x if ha == "left" else (1 - MARGEM_X)
        self.fig.text(px, self.y, txt, fontsize=tamanho, color=cor, family=FONTE,
                      fontweight="bold" if negrito else "normal",
                      va="top", ha=ha)
        self.y -= avanco

    def paragrafo(self, txt: str, *, tamanho=9, cor=COR_TEXTO, x=MARGEM_X) -> None:
        for linha in textwrap.wrap(txt, width=_largura(tamanho)):
            self.espaco(LINHA)
            self.texto(linha, tamanho=tamanho, cor=cor, x=x)

    def regua(self) -> None:
        self.fig.add_artist(plt.Line2D([MARGEM_X, 1 - MARGEM_X], [self.y, self.y],
                                       color="#dcdcdc", linewidth=0.8))
        self.y -= 0.014

    def titulo_secao(self, txt: str) -> None:
        self.espaco(0.055)          # nao deixa um titulo orfao no rodape
        self.y -= 0.008
        self.texto(txt.upper(), tamanho=10.5, cor=COR_TITULO, negrito=True,
                   avanco=0.020)

    def cabecalho(self, titulo: str, subtitulo: str = "") -> None:
        self.texto(titulo, tamanho=17, cor=COR_TITULO, negrito=True, avanco=0.025)
        if subtitulo:
            self.texto(subtitulo, tamanho=9, cor=COR_SUAVE, avanco=0.020)
        self.regua()

    # -- Finalizacao ---------------------------------------------------------
    def salvar(self, pdf: PdfPages) -> int:
        total = len(self.figs)
        for i, fig in enumerate(self.figs, start=1):
            fig.text(MARGEM_X, 0.030,
                     "Sistema de reconhecimento mandibular digital — "
                     f"{self.paciente or 'paciente não identificado'}",
                     fontsize=7.5, color=COR_SUAVE, ha="left", family=FONTE)
            fig.text(1 - MARGEM_X, 0.030, f"Página {i} de {total}",
                     fontsize=7.5, color=COR_SUAVE, ha="right", family=FONTE)
            pdf.savefig(fig)
            plt.close(fig)
        return total


def _altura_medida(m: Medida) -> float:
    """Altura aproximada do bloco de uma medida, para decidir a quebra."""
    linhas = len(textwrap.wrap(m.leitura, _largura(8.5))) + len(
        textwrap.wrap(f"Referência: {m.referencia}", _largura(7.8))
    )
    return 0.019 + LINHA * linhas + 0.010


def _bloco_medida(doc: _Doc, m: Medida) -> None:
    """Nome + valor destacado + leitura + referencia, com barra de nivel."""
    altura = _altura_medida(m)
    doc.espaco(altura)
    topo = doc.y

    doc.fig.text(MARGEM_X, doc.y, m.nome, fontsize=9.5, color=COR_TEXTO,
                 va="top", fontweight="bold", family=FONTE)
    doc.fig.text(1 - MARGEM_X, doc.y, m.valor, fontsize=11, family=FONTE,
                 color=NIVEL_COR.get(m.nivel, COR_TEXTO), va="top", ha="right",
                 fontweight="bold")
    doc.y -= 0.019

    doc.paragrafo(m.leitura, tamanho=8.5)
    doc.paragrafo(f"Referência: {m.referencia}", tamanho=7.8, cor=COR_SUAVE)
    doc.y -= 0.010

    # Barra colorida do nivel, à esquerda do bloco inteiro.
    doc.fig.add_artist(plt.Rectangle(
        (MARGEM_X - 0.018, doc.y + 0.006), 0.005, topo - doc.y,
        color=NIVEL_COR.get(m.nivel, COR_SUAVE), transform=doc.fig.transFigure,
    ))


def _secao_identificacao(doc: _Doc, dados: dict, session_id: str,
                         paciente: str) -> None:
    linhas = [
        ("Paciente", paciente or "não informado"),
        ("Data da sessão", datetime.now().strftime("%d/%m/%Y às %H:%M")),
        ("Identificador", session_id),
        ("Duração", _fmt(dados["duracao_s"], 0, " segundos")),
        ("Quadros analisados",
         f"{dados['frames_validos']} válidos de {dados['total_frames']}"),
    ]
    for rotulo, valor in linhas:
        doc.fig.text(MARGEM_X, doc.y, f"{rotulo}:", fontsize=9, color=COR_SUAVE,
                     va="top", family=FONTE)
        doc.fig.text(MARGEM_X + 0.17, doc.y, str(valor), fontsize=9,
                     color=COR_TEXTO, va="top", fontweight="bold", family=FONTE)
        doc.y -= 0.0165
    doc.y -= 0.006
    doc.regua()


def _secao_biofeedback(doc: _Doc, dados: dict) -> None:
    bio = dados.get("biofeedback") or {}
    total_frames = max(dados["total_frames"], 1)
    doc.titulo_secao("Biofeedback durante a sessão")
    if bio:
        doc.paragrafo("Mensagens exibidas ao paciente na tela enquanto executava "
                      "o movimento, com a fração do tempo em que cada uma esteve "
                      "ativa:", tamanho=8.5, cor=COR_SUAVE)
        for msg, n in bio.items():
            doc.espaco(LINHA)
            doc.texto(f"•  {msg} — {100.0 * n / total_frames:.0f}% do tempo",
                      tamanho=8.5)
    else:
        doc.paragrafo("Nenhuma mensagem foi disparada: o movimento permaneceu "
                      "dentro da faixa treinada, sem desvio acima da zona morta "
                      "nem inconsistência entre repetições.",
                      tamanho=8.5, cor=COR_SUAVE)


def _secao_graficos(doc: _Doc, figuras: dict[str, str]) -> None:
    existentes = [(r, a) for r, a in figuras.items() if a and os.path.exists(a)]
    if not existentes:
        return
    doc.titulo_secao("Gráficos da sessão")

    altura = 0.235
    for rotulo, arquivo in existentes:
        # Cada grafico e indivisivel: se nao couber inteiro, vai para a proxima
        # pagina em vez de ser cortado ao meio.
        doc.espaco(altura + 0.048)
        doc.fig.text(MARGEM_X, doc.y, rotulo, fontsize=8.5, color=COR_SUAVE,
                     va="top", family=FONTE)
        doc.y -= 0.018
        ax = doc.fig.add_axes([MARGEM_X, doc.y - altura, 1 - 2 * MARGEM_X, altura])
        ax.imshow(mpimg.imread(arquivo))
        ax.axis("off")
        doc.y -= altura + 0.030


def _secao_notas(doc: _Doc, dados: dict) -> None:
    doc.nova_pagina()
    doc.cabecalho("Como ler este relatório",
                  "Método, calibração e limitações da medida")

    tem_mm = dados["ref_mm"] is not None
    blocos = [
        ("Como as medidas são obtidas",
         "Uma câmera comum registra pontos anatômicos da face (olhos, nariz, "
         "lábios e queixo) a cada quadro. A abertura da boca e o desvio do queixo "
         "são medidos em relação à largura facial — a distância entre os cantos "
         "externos dos olhos — o que torna as medidas independentes da distância "
         "do paciente até a câmera."),
        ("Calibração desta sessão",
         f"Escala: {dados['ref_mm']:.0f} mm entre os cantos externos dos olhos, "
         "medidos pelo profissional. Os valores em milímetros são estimativas "
         "derivadas dessa referência."
         if tem_mm else
         "Esta sessão foi feita SEM calibração em milímetros. Os valores estão em "
         "proporção da largura do rosto e não podem ser comparados às faixas de "
         "referência clínicas. Para obter milímetros, meça com régua a distância "
         "entre os cantos externos dos olhos do paciente e informe esse valor "
         "antes de iniciar a avaliação."),
        ("Faixa de movimento",
         "A faixa foi calibrada no início da sessão (boca fechada, depois abertura "
         "máxima), o que define os limiares de contagem das repetições."
         if dados["calibrado"] else
         "A faixa de movimento NÃO foi calibrada nesta sessão. A contagem de "
         "repetições usou os valores mínimo e máximo observados, o que é menos "
         "confiável do que a calibração guiada."),
        ("Sobre as faixas de referência",
         "Abertura de 40 a 60 mm e didução de 9 a 12 mm vêm de Dufour & Pillu, "
         "Biomecânica Funcional, capítulo 16, página 553. Para simetria facial, "
         "desvio do queixo e ângulo mandibular NÃO existe, na literatura "
         "consultada, um limiar que separe o fisiológico do patológico: as fontes "
         "afirmam que toda face é assimétrica. Essas medidas são apresentadas para "
         "comparação do paciente com ele mesmo ao longo do tratamento, e não como "
         "classificação clínica."),
        ("Limitações conhecidas",
         "As medidas dependem de boa iluminação e de cabeça estável e voltada para "
         "a câmera. Rotações da cabeça fora do plano da imagem reduzem a precisão "
         "do desvio lateral e tornam a simetria não interpretável. O sistema "
         "analisa apenas a vista frontal. A conversão para milímetros é uma "
         "estimativa a partir de uma referência facial, não uma medição direta "
         "com instrumento."),
    ]
    for titulo, corpo in blocos:
        doc.titulo_secao(titulo)
        doc.paragrafo(corpo, tamanho=9)

    # -- Ressalva final, em caixa destacada ----------------------------------
    linhas = textwrap.wrap(DISCLAIMER, _largura(8.8))
    altura = 0.030 + LINHA * len(linhas)
    doc.espaco(altura + 0.02)
    doc.y -= 0.012
    doc.fig.add_artist(plt.Rectangle(
        (MARGEM_X - 0.020, doc.y - altura + 0.014), 1 - 2 * MARGEM_X + 0.040, altura,
        facecolor="#fff8ee", edgecolor="#e0c9a0", transform=doc.fig.transFigure,
    ))
    doc.texto("IMPORTANTE", tamanho=9.5, cor="#a35a00", negrito=True, avanco=0.017)
    for linha in linhas:
        doc.texto(linha, tamanho=8.8)


def write_pdf_report(
    path: str,
    dados: dict,
    session_id: str,
    paciente: str = "",
    figuras: dict[str, str] | None = None,
    observacoes: str = "",
) -> str:
    """
    Gera o relatorio PDF da sessao e retorna o caminho do arquivo.

    Recebe os mesmos argumentos de `report.write_report`, para que o exportador
    produza os dois documentos a partir de uma unica agregacao de dados.
    """
    figuras = figuras or {}
    medidas = build_findings(dados)

    doc = _Doc(paciente)
    doc.cabecalho(
        "Relatório da avaliação mandibular",
        "Documento de apoio gerado automaticamente — não substitui o exame clínico",
    )
    _secao_identificacao(doc, dados, session_id, paciente)

    for grupo, titulo in GRUPOS:
        do_grupo = [m for m in medidas if m.grupo == grupo]
        if not do_grupo:
            continue
        doc.titulo_secao(titulo)
        for m in do_grupo:
            _bloco_medida(doc, m)

    _secao_biofeedback(doc, dados)

    if observacoes.strip():
        doc.titulo_secao("Observações")
        for linha in observacoes.strip().splitlines():
            doc.paragrafo(linha, tamanho=8.5)

    _secao_graficos(doc, figuras)
    _secao_notas(doc, dados)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with PdfPages(path) as pdf:
        doc.salvar(pdf)
        info = pdf.infodict()
        info["Title"] = f"Avaliação mandibular — {paciente or session_id}"
        info["Subject"] = "Relatório de apoio; não constitui diagnóstico"
        info["Creator"] = "Sistema de reconhecimento mandibular digital"

    return path
