# Sistema de reconhecimento mandibular digital

Ferramenta de **baixo custo** para reconhecer e acompanhar digitalmente o
movimento mandibular por meio de uma **webcam comum**, como apoio à análise
médica/odontológica e à reabilitação funcional.

> ⚠️ **Uso apenas como apoio.** O sistema não substitui o diagnóstico nem a
> avaliação de um profissional de saúde.

Disciplina: Biomecânica — Engenharia Biomédica
Integrantes: Livia Santelli Pegoraro e Maria Luisa Gonçalves Ferreira

---

## O que o sistema faz

Usando visão computacional (MediaPipe Face Landmarker + OpenCV), o software
identifica pontos anatômicos da face (nariz, cantos da boca, lábios, linha
média facial e queixo) e calcula, **em tempo real**:

| Métrica | Descrição |
|---|---|
| **Abertura bucal relativa** | Distância vertical entre os lábios, adimensional — normalizada pela distância interocular (cantos externos dos olhos), não pela largura total do rosto. Invariante à distância da câmera. |
| **Posição do queixo** (lateral absoluta) | Deslocamento horizontal do queixo em relação à linha média facial (com sinal; mesma normalização acima). |
| **Movimento desde o neutro** (lateral dinâmica) | `posição do queixo − posição neutra calibrada` (baseline da fase de boca fechada da calibração); fica ~0 na posição neutra mesmo que a posição absoluta não seja zero (assimetria facial estática). |
| **Repetibilidade** | Contagem de ciclos de abertura/fechamento (do fechado ao fechado), amplitude, duração e coeficiente de variação entre repetições. |

> Convenção de sinal: **positivo = direita anatômica do paciente, negativo =
> esquerda anatômica** — sempre do ponto de vista do próprio paciente, com a
> correção automática para imagem espelhada (`--no-flip` desliga o espelho e
> os rótulos "direita"/"esquerda" se ajustam sozinhos).

A imagem ao vivo mostra os marcadores faciais, uma barra de biofeedback da
abertura e um painel com os valores instantâneos. Os dados podem ser
exportados em **CSV** e como **gráfico** da evolução temporal.

---

## Instalação

```bash
# 1. Dependências Python
pip install -r requirements.txt

# 2. Modelo de detecção facial (~3,8 MB, servidor oficial do Google)
python download_model.py
```

## Como usar

```bash
python run.py
```

Opções úteis:

```bash
python run.py --camera 1            # escolher outra webcam
python run.py --ref-mm 63           # converter medidas para milímetros
python run.py --width 640 --height 480
python run.py --no-flip             # não espelhar a imagem
```

> `--ref-mm` recebe a distância real, em milímetros, entre os cantos externos
> dos olhos do paciente. Com esse valor, as medidas passam a ser exibidas
> também em mm (calibração de escala).

### Análise offline de vídeos gravados

Para reprocessar uma coleta em vídeo (sem webcam ao vivo) — útil na validação
com vídeos controlados (Semana 5):

```bash
python analyze_video.py coleta.mp4
python analyze_video.py coleta.mp4 --ref-mm 63
```

Gera CSV, gráfico e um **resumo** com repetições, repetibilidade (coeficiente de
variação) e comparação com as faixas de referência clínicas (abertura 40–60 mm,
didução 9–12 mm).

### Controles do teclado

| Tecla | Ação |
|---|---|
| `C` | Calibrar (assistente: boca fechada → boca aberta) |
| `X` | Apagar a calibração atual |
| `V` | Habilitar / desabilitar a gravação de vídeo para a **próxima** sessão |
| `R` | Iniciar / encerrar uma sessão (dados **e** vídeo, sincronizados) |
| `E` | Exportar a sessão encerrada (CSV + resumo + metadados + gráficos + vídeo) |
| `Z` | Zerar a sessão atual (amostras, vídeo e contagem) — **preserva a calibração** |
| `Q` / `ESC` | Sair (finaliza com segurança uma sessão ainda ativa) |

`V` só define se a **próxima** sessão iniciada com `R` vai gravar vídeo; não
afeta uma sessão já em andamento. `R` inicia/encerra os dados e o vídeo no
mesmo instante, para que `dados.csv` e `video.mp4` cubram exatamente o mesmo
intervalo.

### Fluxo recomendado

1. Posicione o rosto bem iluminado e centralizado, cabeça estável.
2. Tecle **`C`** e siga o assistente de calibração (fechado → aberto).
3. (Opcional) Tecle **`V`** para habilitar a gravação de vídeo da sessão.
4. Tecle **`R`** para iniciar a sessão e realize os movimentos de abertura/fechamento.
5. Tecle **`R`** novamente para encerrar a sessão.
6. Tecle **`E`** para exportar a pasta da sessão em `resultados/`.

---

## Estrutura do projeto

```
biomecanica/
├── run.py                    # ponto de entrada (webcam ao vivo)
├── analyze_video.py          # análise offline de vídeos gravados
├── download_model.py         # baixa o modelo face_landmarker.task
├── requirements.txt
├── models/                   # modelo .task (não versionado)
├── resultados/               # CSVs e gráficos exportados
├── docs/
│   └── pesquisa_biomecanica.md  # revisão de ATM e movimento mandibular (Semana 1)
├── tests/
│   └── test_metrics.py       # testes da lógica biomecânica (sem webcam)
└── src/mandibular/
    ├── config.py             # índices de landmarks e parâmetros
    ├── landmarks.py          # wrapper do MediaPipe Face Landmarker
    ├── metrics.py            # métricas biomecânicas + detecção de ciclos
    ├── recorder.py           # gravação e exportação CSV
    ├── plotting.py           # geração de gráficos
    └── app.py                # aplicação em tempo real (interface OpenCV)
```

## Testes

```bash
python tests/test_metrics.py
```

Os testes validam o cálculo das métricas e a detecção de ciclos usando
landmarks sintéticos, sem necessidade de webcam.

---

## Limitações (importantes)

- Requer **boa iluminação** e **posicionamento estável** da cabeça.
- As medidas são **relativas**; a conversão para mm depende de calibração
  (`--ref-mm`) e é uma **estimativa**.
- Rotações acentuadas da cabeça (fora do plano) reduzem a precisão do desvio
  lateral.
- Ferramenta **de apoio e uso didático** — não é um dispositivo médico.
```
