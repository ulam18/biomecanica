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
média facial e queixo) **na vista frontal** e calcula, **em tempo real**:

| Métrica | Descrição |
|---|---|
| **Abertura bucal relativa** | Distância vertical entre os lábios, adimensional — normalizada pela distância interocular (cantos externos dos olhos), não pela largura total do rosto. Invariante à distância da câmera. |
| **Posição do queixo** (lateral absoluta) | Deslocamento horizontal do queixo em relação à linha média facial (com sinal; mesma normalização acima). |
| **Movimento desde o neutro** (lateral dinâmica) | `posição do queixo − posição neutra calibrada` (baseline da fase de boca fechada da calibração); fica ~0 na posição neutra mesmo que a posição absoluta não seja zero (assimetria facial estática). |
| **Repetibilidade** | Contagem de ciclos de abertura/fechamento (do fechado ao fechado), amplitude, duração e coeficiente de variação entre repetições. |
| **Simetria facial** | Índice 0–1 por pares homólogos (olhos, nariz, boca, contorno) + ângulo `cant` (olhos × boca). |
| **Desvio da linha média mandibular** | Inclinação, em graus, da linha násio→mento — versão angular do desvio, menos sensível à distância da câmera. |

> Convenção de sinal: **positivo = direita anatômica do paciente, negativo =
> esquerda anatômica** — sempre do ponto de vista do próprio paciente, com a
> correção automática para imagem espelhada (`--no-flip` desliga o espelho e
> os rótulos "direita"/"esquerda" se ajustam sozinhos).

A imagem ao vivo mostra os marcadores faciais, uma barra de biofeedback da
abertura e um painel com os valores instantâneos. Ao final, a sessão é
exportada em **CSV**, **gráficos**, **JSON** e um **relatório em PDF e HTML**
que analisa **cada medida** — valor, faixa de referência com fonte e leitura em
linguagem clínica —, pronto para anexar ao prontuário.

> O sistema trabalha **apenas com a vista frontal**. A análise de perfil
> (plano sagital) foi removida: em 2D, sem foto padronizada nem controle de
> distância e rotação da cabeça, as projeções e ângulos sagitais não se
> mostraram confiáveis.

---

## Dois modos de uso

| Modo | Para quem | Como abrir |
|---|---|---|
| **Menu com botões** | uso clínico, sem conhecimento de TI | `INSTALAR.bat` → ícone na Área de Trabalho |
| **Linha de comando** | uso acadêmico/desenvolvimento | `python run.py` |

### Uso clínico (sem terminal)

1. Clique duas vezes em **`INSTALAR.bat`**. Ele localiza o Python (e orienta a
   instalação, se faltar), cria o ambiente `.venv`, instala as dependências,
   baixa o modelo e cria o atalho **Análise Mandibular** na Área de Trabalho.
2. Abra o programa pelo ícone. O menu ([launcher.py](launcher.py)) tem campo de
   paciente, calibração em mm, escolha da câmera e botões grandes.
3. Na janela da câmera, a faixa inferior traz os botões
   **1. CALIBRAR → 2. GRAVAR → 3. SALVAR**, além de RECOMEÇAR e SAIR. O teclado
   continua funcionando em paralelo.
4. Ao salvar, o **relatório abre sozinho no navegador**.

O passo a passo em linguagem não técnica está em [LEIA-ME.txt](LEIA-ME.txt) e
dentro do programa, no botão "Como usar".

---

## Instalação manual (desenvolvimento)

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
python run.py --paciente "Ana"      # nomeia o relatório e agrupa o histórico
python run.py --simples             # painel em linguagem clínica + relatório automático
python run.py --sem-botoes          # oculta a faixa de botões (só teclado)
```

> `--ref-mm` recebe a distância real, em milímetros, entre os cantos externos
> dos olhos do paciente. Com esse valor, as medidas passam a ser exibidas
> também em mm (calibração de escala).

### Análise offline de vídeos gravados

Para reprocessar uma coleta em vídeo (sem webcam ao vivo) — útil na validação
com vídeos controlados (Semana 5):

```bash
python analyze_video.py coleta.mp4
python analyze_video.py coleta.mp4 --ref-mm 63 --paciente "Ana" --abrir-relatorio
```

Gera CSV, gráfico e um **resumo** com repetições, repetibilidade (coeficiente de
variação) e comparação com as faixas de referência clínicas (abertura 40–60 mm,
didução 9–12 mm).

### Controles

Cada ação tem um botão na faixa inferior da janela (modo simples) **e** um
atalho de teclado; os dois disparam exatamente a mesma lógica.

| Botão | Tecla | Ação |
|---|---|---|
| `1. CALIBRAR` | `C` | Calibrar (assistente: boca fechada → boca aberta) |
| — | `X` | Apagar a calibração atual |
| `2. GRAVAR` | `R` | Iniciar / encerrar uma sessão (dados **e** vídeo, sincronizados) |
| — | `V` | Habilitar / desabilitar a gravação de vídeo para a **próxima** sessão |
| `3. SALVAR` | `E` | Exportar a sessão encerrada (CSV, resumo, metadados, gráficos, vídeo, relatório) |
| `RECOMEÇAR` | `Z` | Zerar a sessão atual (amostras, vídeo e contagem) — **preserva a calibração** |
| `SAIR` | `Q` / `ESC` | Sair (finaliza com segurança uma sessão ainda ativa) |

`V` só define se a **próxima** sessão iniciada com `R`/`GRAVAR` vai gravar
vídeo; não afeta uma sessão já em andamento. `R`/`GRAVAR` inicia e encerra os
dados e o vídeo no mesmo instante, para que `dados.csv` e `video.mp4` cubram
exatamente o mesmo intervalo.

### Fluxo recomendado

1. Posicione o rosto bem iluminado e centralizado, cabeça estável.
2. **Calibrar** e seguir o assistente (fechado → aberto).
3. (Opcional) **Vídeo** para habilitar a gravação de vídeo da sessão.
4. **Gravar** para iniciar a sessão e realizar os movimentos de abertura/fechamento.
5. **Gravar** novamente para encerrar a sessão.
6. **Salvar**: gera a pasta da sessão em `resultados/`.

### O que é gerado por sessão

```
resultados/sessao_AAAA-MM-DD_HH-MM-SS/
├── relatorio.pdf        # documento do prontuário: medida a medida, imprimível
├── relatorio.html       # mesma leitura, em versão navegável
├── dados.csv            # série temporal, uma linha por quadro
├── resumo.json          # estatísticas + avisos de qualidade
├── metadados.json       # câmera, resolução, calibração, paciente
├── abertura_tempo.png
├── lateralidade_tempo.png
├── lateralidade_dinamica_tempo.png
├── trajetoria_abertura_lateralidade.png
├── trajetoria_abertura_lateralidade_dinamica.png
├── ciclos_individuais.png     # se houve ciclos completos
├── ciclos_curva_media.png     # se houve ciclos completos
└── video.mp4             # opcional (tecla V / botão vídeo)
```

Com `--paciente` preenchido, cada sessão também acrescenta uma linha-resumo em
`resultados/evolucao_<paciente>.csv`, base do gráfico de evolução entre
atendimentos (botão "Ver a evolução do paciente" no menu).

---

## Estrutura do projeto

```
biomecanica/
├── LEIA-ME.txt               # instruções para o usuário final (não técnico)
├── INSTALAR.bat              # instalação em um duplo clique (Windows)
├── criar_atalho.py           # gera o ícone e o atalho na Área de Trabalho
├── launcher.py               # menu com botões (ponto de entrada clínico)
├── run.py                    # ponto de entrada por linha de comando
├── analyze_video.py          # análise offline de vídeos gravados
├── compare_sessions.py       # comparação entre sessões
├── download_model.py         # baixa o modelo face_landmarker.task
├── requirements.txt
├── assets/                   # ícone gerado na instalação
├── models/                   # modelo .task (não versionado)
├── resultados/               # sessões exportadas (não versionado)
├── docs/
│   ├── pesquisa_biomecanica.md  # revisão de ATM e movimento mandibular
│   └── artigo/                  # artigo em LaTeX
├── tests/                    # testes sem webcam (landmarks sintéticos)
└── src/mandibular/
    ├── config.py             # índices de landmarks e parâmetros
    ├── landmarks.py          # wrapper do MediaPipe Face Landmarker
    ├── pipeline.py           # processamento de um quadro
    ├── metrics.py            # métricas biomecânicas + detecção de ciclos
    ├── quality.py            # controle de qualidade do quadro
    ├── calibration.py        # assistente de calibração
    ├── filters.py            # suavização EMA
    ├── classification.py     # faixas de referência (com fonte)
    ├── feedback.py           # mensagens de biofeedback
    ├── overlay.py            # desenho: marcadores, painel e botões clicáveis
    ├── recorder.py           # gravação e exportação CSV
    ├── plotting.py           # geração de gráficos
    ├── report.py             # agregação + análise medida a medida + HTML
    ├── pdf_report.py         # o mesmo conteúdo em PDF (A4, paginado)
    ├── evolution.py          # histórico do paciente entre sessões
    ├── exporter.py           # empacota a sessão exportada
    ├── video_recorder.py     # gravação do vídeo anotado
    └── app.py                # aplicação em tempo real (interface OpenCV)
```

## Testes

```bash
python -m pytest tests -q
```

55 testes validam métricas, detecção de ciclos, qualidade, análise frontal,
exportação, análise medida a medida, classificação e histórico usando landmarks
sintéticos, sem necessidade de webcam.

---

## Limitações (importantes)

- Requer **boa iluminação** e **posicionamento estável** da cabeça.
- As medidas são **relativas**; a conversão para mm depende de calibração
  (`--ref-mm`) e é uma **estimativa**.
- Rotações acentuadas da cabeça (fora do plano) reduzem a precisão do desvio
  lateral.
- Apenas **vista frontal**: não há análise de perfil (ver nota acima).
- Ferramenta **de apoio e uso didático** — não é um dispositivo médico.
```
