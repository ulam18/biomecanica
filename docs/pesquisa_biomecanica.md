# Pesquisa — Biomecânica da ATM e do movimento mandibular

**Projeto:** Sistema de reconhecimento mandibular digital para apoio à análise médica
**Disciplina:** Biomecânica — Engenharia Biomédica
**Integrantes:** Livia Santelli Pegoraro e Maria Luisa Gonçalves Ferreira
**Entregável:** Semana 1 — Revisão de ATM, movimento mandibular e ferramentas de visão computacional

> **Nota de fundamentação.** Os valores anatômicos e as amplitudes de referência
> deste documento foram extraídos de *Biomecânica Funcional* (Dufour & Pillu),
> Capítulo 16 — "Cabeça (crânio e face)", nas páginas indicadas entre parênteses
> (numeração impressa do livro). Onde há estimativa, derivação ou aproximação, isso
> está sinalizado no texto. Este material é de **apoio e uso didático**; não
> constitui protocolo diagnóstico.

---

## 1. Objetivo da revisão

Fundamentar biomecanicamente as métricas que o software calcula a partir de
vídeo — **abertura bucal**, **desvio lateral (didução)** e **repetibilidade** —,
relacionando cada uma ao movimento fisiológico da articulação temporomandibular
(ATM) e às faixas de amplitude consideradas normais na literatura da disciplina.

---

## 2. Anatomia funcional da ATM

A ATM é **a única articulação da cabeça com mobilidade visível e importante**
(p. 546). Anatomicamente é classificada como **bicondilar**: existem duas
articulações, uma direita e uma esquerda, fisicamente separadas mas
funcionalmente acopladas (p. 546). Cada côndilo mandibular é um ovoide cujo
grande eixo forma um ângulo de cerca de **160°** com o contralateral (p. 546).

Componentes relevantes para o movimento:

- **Superfícies articulares** — no crânio, o tubérculo articular do temporal
  (convexo) e a fossa mandibular (côncava); na mandíbula, o côndilo mandibular
  (p. 545–546).
- **Disco articular** — menisco móvel que recobre o côndilo "como uma boina",
  dividindo a cavidade em compartimentos superior e inferior; é tracionado para
  a frente pelo pterigóideo lateral durante a abertura da boca (p. 548).
- **Cápsula e ligamentos** — cápsula frouxa com sinovial fibrosa; ligamentos
  colaterais medial e lateral de cada lado (p. 548).

A posição estável passiva é o **fechamento da boca com os dentes engrenados**;
a abertura máxima é estabilizada ativamente pela musculatura. As **posições
intermediárias** são as de maior risco mecânico, quando a propulsão se soma ao
abaixamento (p. 554).

---

## 3. Movimentos mandibulares e amplitudes de referência

O livro organiza a mobilidade da ATM em movimentos **analíticos** (elementares)
e a mobilidade **funcional** (a abertura da boca real, que combina os
analíticos). Os movimentos da ATM servem a três funções: **mastigação, fonação
e deglutição** (p. 551).

### 3.1 Abaixamento–elevação
Movimento **sagital**, em torno de um eixo transversal que passa pelo centro das
cabeças condilares (p. 551). É a "abertura angular" pura (visível em um esqueleto
montado), que difere da abertura da boca real do ser humano.

### 3.2 Propulsão–retropulsão
Deslizamento da mandíbula para a frente e para trás, em **plano horizontal**.
Amplitude normal da propulsão: **6 a 8 mm** (medida entre os incisivos superiores
e inferiores); a retropulsão tem amplitude semelhante, em sentido inverso (p. 551).

### 3.3 Didução (lateralização) — base do "desvio lateral"
Movimento de **lateralização da ponta do queixo** para a direita ou esquerda,
produzido pelo avanço unilateral de um côndilo enquanto o outro permanece na
fossa. Amplitude média: **9 a 12 mm** (p. 553). É um teste clínico importante da
propulsão unilateral.

### 3.4 Abertura da boca (mobilidade funcional) — base da "abertura"
É um movimento **combinado**: primeiro um abaixamento angular (até ~20 mm de
afastamento dos dentes), ao qual se associa, em seguida, uma propulsão (p. 553).
Amplitude normal: **em média 40 a 60 mm** — equivalente a intercalar três dedos
sobrepostos entre os dentes superiores e inferiores (p. 553).

### 3.5 Caminho de abertura e simetria — base da "assimetria/desvio"
A estabilidade dinâmica se traduz pelo **domínio da simetria na abertura**,
formando um **"caminho de abertura sagital"** (p. 554). A simetria posicional é
avaliada pelo **alinhamento entre a junção dos incisivos superiores e inferiores**
(p. 554). Perturbações do caminho de abertura (mau alinhamento no início ou ao
longo do movimento) são medidas **em milímetros com paquímetro**, entre a linha
de separação dos incisivos (p. 553).

### Tabela-resumo (valores de referência)

| Movimento | Amplitude normal | Fonte (pág.) |
|---|---|---|
| Abertura da boca | **40–60 mm** | p. 553 |
| Didução (lateral) | **9–12 mm** | p. 553 |
| Propulsão / retropulsão | **6–8 mm** | p. 551 |
| Início da propulsão na abertura | a partir de ~20 mm | p. 553 |

---

## 3.6 Simetria e assimetria facial (fontes complementares)

Esta seção fundamenta as métricas de **simetria** e proporções da **vista
frontal** com literatura de análise facial e de assimetrias dentofaciais. As afirmações
abaixo estão amarradas às fontes; onde a literatura **não** fornece um limiar,
isso é declarado — o software **não** cria cortes clínicos inexistentes.

### 3.6.1 Premissa: toda face é assimétrica
Carlini & Gomes (2005) afirmam que *"todas as faces são assimétricas"*; o que
motiva conduta clínica é a **queixa estética + estabilidade oclusal + etiologia**,
não um número. **Consequência para o software:** o índice de simetria é tratado
como medida **relativa/intrasujeito** e didática, sem rótulo de "patológico".

### 3.6.2 Eixos e planos de referência (implementáveis em 2D frontal)
Do material de análise facial:
- **Plano sagital médio** é o eixo da simetria; os demais planos devem ser-lhe
  perpendiculares.
- **Plano bipupilar** (linha dos olhos) e **plano intercomissural** (cantos da
  boca) devem ser **paralelos entre si** e ⊥ ao plano sagital médio. → o software
  mede o **ângulo entre a linha dos olhos e a linha da boca** (`cant`), que é ~0
  numa face simétrica.
- **Linha interpupilar** como referência horizontal também aparece no método de
  Benson & Laskin (mordida em espátula), citado por Carlini & Gomes (2005). O
  software usa o eixo inter-ocular como horizontal — mesma lógica.

### 3.6.3 Proporções faciais (cânones estéticos — não limiares clínicos)
Do material de análise facial:
- **Terços verticais** iguais: Trichion→Glabela, Glabela→Subnasal, Subnasal→Mento;
  em perfil, terço médio (G–Sn) : terço inferior (Sn–Me) = **1:1**.
- **Quintos horizontais**: a largura facial equivale a **cinco larguras de olho**;
  distância intercantal medial ≈ largura de uma fenda ocular ≈ largura interalar.
- **Boca**: distância intercomissural deve ser **menor que a interpupilar e maior
  que a interalar**.
- A **razão áurea (1,618)** aparece nas fontes apenas de forma qualitativa e é
  **explicitamente desqualificada** como modelo científico (Camargos et al., 2009)
  — por isso **não** é usada como critério no software.

### 3.6.4 Ângulo de perfil (revisão bibliográfica — não implementado)

> **Nota (revisão do escopo):** a análise de **vista de perfil** foi **removida do software**. Em 2D, sem foto padronizada nem controle de distância e rotação da cabeça, as projeções e ângulos sagitais não se mostraram confiáveis. O sistema opera **apenas na vista frontal**. A revisão abaixo permanece como registro bibliográfico.

Do material de análise facial, o **ângulo Glabela–Subnasal–Pogônio mole**:
- **< 165°** → perfil **convexo** (padrão Classe II);
- **> 175°** → perfil **côncavo** (padrão Classe III);
- entre eles → perfil **reto** (Classe I).

O ângulo clínico é medido em **foto de perfil padronizada** — condição que a
captura por webcam não garante; por isso essa classificação não foi mantida.

### 3.6.5 Desvio mandibular / do mento
Carlini & Gomes (2005) descrevem o desvio como deslocamento da **linha média
dentária inferior / do mento** em relação à **linha média facial** — o que o
software estima pelo desvio horizontal do queixo (152) frente ao nasion (168).
**Não há limiar de referência** nas fontes; os únicos números são magnitudes de
**casos cirúrgicos severos** (desvio de mento de **6–7 mm**), usados apenas como
ordem de grandeza, com ressalva.

### 3.6.7 Angulações de perfil de tecido mole (revisão bibliográfica — não implementado)

> **Nota (revisão do escopo):** a análise de **vista de perfil** foi **removida do software**. Em 2D, sem foto padronizada nem controle de distância e rotação da cabeça, as projeções e ângulos sagitais não se mostraram confiáveis. O sistema opera **apenas na vista frontal**. A revisão abaixo permanece como registro bibliográfico.

Normas levantadas na revisão para eventual triagem de proeminência (fora de
**média ± 2 DP**). São específicas de população/idade.

| Ângulo | Pontos | Norma (média ± DP) | Fonte |
|---|---|---|---|
| Convexidade facial | G–Sn–Pg | **168° ± 5** (faixa clínica 165–175°) | EJO 2008; Ngeow & Aljunid (fotogrametria) |
| Convexidade total | G–Prn–Pg | **147° ± 5** | Ngeow & Aljunid |
| Nasofrontal | G–N–Prn | **144° ± 5** | Ngeow & Aljunid |
| Nasolabial | ~Cm–Sn–Ls | **~100° ± 11** (alta variabilidade) | Ngeow & Aljunid |
| Labiomental | Li–Sm–Pg | **~134° ± 12** (alta variabilidade) | Ngeow & Aljunid |
| Linha E (Ricketts) | Prn–Pg; lábios | lábio sup. **−4 ± 2 mm**, inf. **−2 ± 2 mm** | Ricketts (amostra caucasiana) |

> A convexidade facial aparece na literatura em duas convenções equivalentes: o
> ângulo **interno** G-Sn-Pg (~168°) e o **"ângulo de convexidade"** como desvio
> (~12°), sendo um o suplemento do outro (180° − interno). A linha E dependeria de
> calibração em mm (`--ref-mm`).

### 3.6.6 Por que 2D é uma aproximação (limitação honesta)
As fontes de diagnóstico tridimensional (Carvalho et al., 2025; Alencar et al.)
indicam que o **3D (TCFC/CBCT, estereofotogrametria)** supera o 2D por capturar
**volume e informação espacial** que a imagem 2D não fornece, permitindo
identificar desvios sutis. Nos métodos de avaliação de assimetria levantados por
Alencar et al., a **TCFC responde por ~40%** dos estudos e as **fotografias
digitais 2D por ~10%** — ou seja, o 2D é reconhecido, porém minoritário. O
software é, portanto, uma ferramenta 2D de **triagem/acompanhamento**, não de
diagnóstico de assimetria.

---

## 4. Musculatura

Os músculos mastigadores são classificados pela função — levantadores,
abaixadores, propulsores ou retropropulsores (p. 548, Quadro 16.1):

| Músculo | Abaix. | Elev. | Prop. | Retrop. |
|---|:--:|:--:|:--:|:--:|
| Masseter | – | +++ | P | – |
| Temporal | – | +++ | – | R |
| Pterigóideo lateral | – | – | +++ | – |
| Pterigóideo medial | – | E | P | – |
| Milo-hióideo | A | – | – | R |
| Digástrico | A | – | – | R |
| Gênio-hióideo | A | – | – | R |

Todos os mastigadores são inervados pelo **trigêmeo (nervo mandibular, V3)**; os
músculos da face (inervados pelo nervo facial, VII) funcionam como músculos de
substituição (p. 548).

---

## 5. Disfunções temporomandibulares (contexto clínico)

Alterações do movimento se manifestam como perturbações do caminho de abertura,
estalidos/ressaltos na propulsão, assimetrias e dor. O livro agrupa essas
condições sob **"síndromes algo-disfuncionais do aparelho mandibular" (SADAM)**
— na literatura mais recente, **disfunção temporomandibular (DTM)** (p. 553). A
reeducação recorre à cinesioterapia, ortofonia e ortodontia (p. 551).

Isso justifica o valor de um **registro objetivo e repetível** entre sessões —
exatamente o que o software oferece como complemento à observação clínica.

---

## 6. Ferramentas de visão computacional

### 6.1 MediaPipe Face Landmarker (Tasks API)
Modelo de malha facial do Google que estima **478 landmarks 3D** da face a partir
de imagem 2D (468 do modelo base + 10 refinamentos de íris). Roda em CPU em tempo
real, sem hardware especializado — adequado ao requisito de **baixo custo** do
projeto. Fornece coordenadas normalizadas `(x, y, z)` por landmark, com índices
estáveis correspondentes a pontos anatômicos conhecidos.

> **Observação técnica de ambiente.** A versão instalada expõe apenas a *Tasks
> API* (`FaceLandmarker`), não a antiga `mediapipe.solutions.face_mesh`. O modelo
> `.task` é obtido via `download_model.py`. Os índices de landmark permanecem os
> mesmos do Face Mesh canônico.

### 6.2 OpenCV
Captura de vídeo (webcam ou arquivo), conversão de espaço de cor, desenho da
sobreposição (landmarks, linhas, painel, barra de biofeedback) e a janela
interativa.

### 6.3 Pontos anatômicos utilizados (índices Face Mesh)

| Ponto | Índice | Uso na métrica |
|---|---|---|
| Canto externo do olho esq./dir. | 33 / 263 | Referência facial (escala) + eixo horizontal (linha bipupilar) |
| Canto interno do olho esq./dir. | 133 / 362 | Simetria (par) + quintos/intercantal |
| Raiz do nariz (nasion) | 168 | Referência da linha média |
| Ponta do nariz | 1 | Linha média |
| Asa do nariz esq./dir. | 129 / 358 | Simetria (par) + largura interalar |
| Bochecha esq./dir. | 234 / 454 | Simetria (par) + largura facial (quintos) |
| Lábio interno superior/inferior | 13 / 14 | Abertura bucal |
| Cantos da boca | 61 / 291 | Extremidades + linha intercomissural (`cant`) |
| Glabela / Subnasal | 9 / 2 | Terços faciais |
| Queixo (menton) | 152 | Desvio lateral + linha média mandibular |

---

## 7. Mapeamento biomecânica → métricas do software

Esta é a ponte entre a revisão e o código (`src/mandibular/metrics.py`).

### 7.1 Abertura bucal
- **Biomecânica:** abertura funcional da boca (abaixamento + propulsão), normal 40–60 mm (p. 553).
- **No software:** componente vertical da distância entre os lábios internos (13–14), **normalizada pela largura facial** (distância entre os cantos externos dos olhos, 33–263). A normalização torna a medida invariante à distância da câmera. Com calibração `--ref-mm`, converte-se para milímetros e pode-se comparar com a faixa 40–60 mm.

### 7.2 Desvio lateral (didução)
- **Biomecânica:** lateralização da ponta do queixo, normal 9–12 mm; e a avaliação de simetria pelo alinhamento dos incisivos / caminho de abertura sagital (p. 553–554).
- **No software:** componente **horizontal** da posição do queixo (152) em relação à raiz do nariz (168), projetada sobre o eixo inter-ocular e normalizada pela largura facial. O sinal indica o lado do desvio; a média ao longo da abertura estima a **assimetria do caminho**.

### 7.3 Repetibilidade
- **Biomecânica:** o movimento fisiológico requer harmonia e simetria repetíveis; disfunções aparecem como inconsistência entre repetições (p. 553).
- **No software:** máquina de estados com histerese detecta ciclos abre/fecha; calcula-se amplitude e duração por ciclo e o **coeficiente de variação (CV)** entre ciclos. CV baixo = movimento mais consistente.

### 7.4 Robustez (escolhas de projeto)
Como as medidas são projetadas no referencial definido pelo **eixo inter-ocular**,
elas são aproximadamente invariantes à **inclinação da cabeça no plano (roll)** e
à **distância da câmera** — propriedades verificadas por testes automatizados
(`tests/test_metrics.py`: `test_roll_invariance`, `test_scale_invariance`).

### 7.5 Simetria, proporções e angulações (frontal)
- **Biomecânica/análise facial:** planos bipupilar e intercomissural paralelos e ⊥
  ao plano sagital médio; terços e quintos faciais (§3.6).
- **No software** (`compute_symmetry`, `compute_proportions`): índice de simetria
  por pares homólogos esquerda/direita; ângulo `cant` (olhos × boca); razão dos
  terços (G–Sn : Sn–Me), quintos (largura facial : olho) e cânone da boca. A
  simetria e os ângulos são calculados **dentro de `pipeline.process_frame`**, ou
  seja, no mesmo quadro e sob a mesma regra dos demais sinais: quadro inválido
  não gera medida (grava vazio, nunca zero).
- **Angulações frontais** (`compute_frontal_angles`): **ângulo de desvio mandibular**
  — inclinação (com sinal) da linha média mandibular (Násio→Mento) em relação à
  vertical da linha média facial — e o `cant`. São medidas **relativas/intrasujeito**
  (sem corte clínico; "toda face é assimétrica", §3.6.1), gravadas por frame no CSV
  (`simetria_indice`, `cant_graus`, `desvio_mandibular_graus`, `frontal`) e
  agregadas por sessão no histórico `evolucao_<paciente>.csv`. O desvio **linear**
  do mento (mm/relativo) continua em `compute_frame_metrics` + `classify_chin_deviation`.

### 7.6 Biofeedback e leitura da sessão
- **Ao vivo** (`feedback.py`): mensagens curtas e não diagnósticas sobre faixa de
  abertura treinada, direção do desvio e repetibilidade, desenhadas no painel a
  cada quadro.
- **Persistido** (coluna `biofeedback` do CSV): as mesmas mensagens são gravadas
  por quadro, o que permite ao relatório informar **por qual fração do tempo**
  cada aviso esteve ativo — sem isso, o biofeedback desapareceria ao fim da sessão.
- **Leitura medida a medida** (`report.build_findings`): cada medida vira um
  registro com valor, **referência declarada** (faixa + fonte, ou a afirmação
  explícita de que não há corte clínico) e a leitura em linguagem clínica. O
  relatório HTML (`report.py`) e o PDF (`pdf_report.py`) renderizam essa **mesma**
  lista, de modo que os dois documentos não possam divergir sobre o paciente.

### 7.7 Classificação e qualidade
- **Classificação simples** (`classification.py`, Semana 3): abertura (40–60 mm),
  didução (9–12 mm), desvio do mento e simetria (faixa didática), cada saída
  com **fonte** e ressalva.
- **Controle de qualidade** (`quality.py`, Semana 5): iluminação, contraste,
  distância (tamanho da face), frontalidade (yaw) e estabilidade (jitter) —
  heurísticas de engenharia que avisam quando a captura sai das condições válidas.

---

## 8. Protocolo de captura proposto (rascunho — Semana 1/3)

1. Iluminação frontal difusa; fundo neutro; rosto totalmente enquadrado.
2. Cabeça estável e frontal à câmera (limitação assumida do método 2D).
3. Calibração por etapas acionadas pela tecla `C` (boca fechada → confirma →
   abertura máxima → conclui); cada fase dura o tempo que o usuário precisar.
4. (Opcional) informar `--ref-mm` com a distância real entre os cantos externos
   dos olhos, para leitura em milímetros.
5. Movimentos padronizados: N aberturas/fechamentos completos; N lateralizações
   para cada lado.
6. Exportar CSV + gráfico (tecla `E`) e registrar por sessão para comparação.

---

## 9. Limitações

- **Método 2D:** rotações da cabeça fora do plano (yaw/pitch) reduzem a precisão,
  sobretudo do desvio lateral. A literatura mede a didução em 3D (avanço condilar);
  a estimativa por câmera única é uma **aproximação projetiva**. Para assimetria,
  o 3D (CBCT/estereofotogrametria) é superior por capturar **volume** que o 2D
  não fornece (Carvalho et al., 2025; Alencar et al.). O sistema detecta e sinaliza
  a rotação da cabeça (`yaw`) para reduzir leituras de simetria pouco confiáveis.
- **Simetria sem corte clínico:** a literatura consultada não define limiar mm/%
  de assimetria; "todas as faces são assimétricas" (Carlini & Gomes, 2005). O
  índice serve para **comparação intrasujeito** e biofeedback, não para diagnóstico.
- **Calibração mm:** depende de uma medida real informada; sem ela, as medidas são
  relativas (adimensionais) e servem para comparação intrasujeito.
- **Referência facial:** assume que a distância inter-ocular não varia — válido
  para o mesmo sujeito, mas não comparável entre sujeitos diferentes sem calibração.
- **Não é dispositivo médico:** ferramenta de **apoio e ensino**; não substitui
  exame clínico, palpação do trago, paquímetro nem imagem (RM/TC) da ATM.

---

## 10. Referências

- **Dufour, M.; Pillu, M.** *Biomecânica Funcional.* Cap. 16 — "Cabeça (crânio e
  face)": pp. 545–557 (material da disciplina). Fonte das amplitudes e da anatomia
  funcional da ATM citadas acima.
  - Referências internas do capítulo para claims específicos: Catic & Naeije
    (1999) — eixo do abaixamento; Naeije & Hofman (2003), Rantala et al. (2003) —
    abertura da boca; Hiraba et al. (2000) — didução; Itoh et al. (1996), Javaux
    et al. (1999) — geometria condilar; Gillies et al. (2003) — simetria dos
    incisivos.
- **Google MediaPipe** — Face Landmarker (Tasks API), 478 landmarks faciais.
  Documentação técnica do modelo `face_landmarker.task`.
- **OpenCV** — biblioteca de visão computacional (captura, processamento e
  exibição de vídeo).

### Fontes complementares — simetria e assimetria facial (Semana 3/5)

- **Carlini, J. L.; Gomes, K. U.** "Diagnóstico e tratamento das assimetrias
  dentofaciais." *R. Dental Press Ortodon. Ortop. Facial*, v.10, n.1, 2005.
  — premissa "todas as faces são assimétricas"; desvio de linha média/mento;
  linha interpupilar (Benson & Laskin); magnitudes de casos severos (6–7 mm).
- **Camargos, Mendonça, Duarte.** "Da imagem visual do rosto humano: simetria,
  textura e padrão." *Saúde Soc.*, São Paulo, v.18, n.3, 2009. — método de
  espelhamento; cânone vitruviano; razão áurea qualificada como insustentável.
- **Material de análise facial** (Estética Facial — análise frontal e de perfil).
  — planos bipupilar/intercomissural, terços e quintos, ângulo de perfil (165°/175°).
- **Carvalho et al.** "Assimetrias faciais pós-trauma: desafios no diagnóstico
  tridimensional…" *Interference Journal*, v.11, n.2, 2025. — superioridade do
  3D (CBCT) na avaliação volumétrica.
- **Alencar et al.** "Métodos de avaliação das assimetrias faciais" (revisão de
  escopo, UFPE). — frequência dos métodos (TCFC ~40%, fotografia digital ~10%).
- **Figueiredo, S. D. F. et al.** "Bruxismo: uma revisão de literatura."
  *Research, Society and Development*, v.13, n.9, 2024. — relação bruxismo/DTM.

### Fontes web — angulações de perfil (tecido mole), consultadas jul/2026

- **Análise angular do perfil de tecido mole (fotogrametria)** — normas de
  convexidade facial/total, nasofrontal, nasolabial, labiomental (média ± DP):
  Ngeow & Aljunid, *"Angular photogrammetric analysis of the soft tissue
  profile"*, PMC4298958 — https://pmc.ncbi.nlm.nih.gov/articles/PMC4298958/
- **Análise angular de perfil de tecido mole** — *European Journal of
  Orthodontics*, 30(2):135 (2008) — https://academic.oup.com/ejo/article/30/2/135/419126
- **Linha E de Ricketts (posição labial, −4/−2 mm)** — https://www.bceph.com/e-line-analysis
- **Ângulo de convexidade facial (definição/faixas)** — Qoves —
  https://www.qoves.com/insights/measurements/facial-convexity-angle
- **Guia clínico de avaliação do perfil facial** — Spear Education —
  https://www.speareducation.com/resources/spear-digest/evaluating-facial-esthetics-facial-profile/
- **Análise Facial (perfil de tecido mole, PT-BR)** — SciELO / Dental Press —
  https://www.scielo.br/j/dpress/a/vcNJFKLcHG8ZsCh747q9Hxy/?format=pdf&lang=pt

> As normas de perfil são **estéticas/populacionais**, não limiares diagnósticos,
> e derivam de amostras específicas (etnia/idade). A triagem por ±2 DP é **apoio**;
> a decisão é do profissional. Confirmar valores na fonte antes de uso clínico.

> Os valores clínicos (40–60 mm, 9–12 mm, 6–8 mm) provêm da fonte impressa citada
> e devem ser confirmados com a bibliografia clínica de referência antes de uso
> fora do contexto didático. Para **simetria/assimetria**, as fontes acima **não**
> fornecem limiar mm/% que separe fisiológico de patológico — o software não
> inventa cortes: usa faixas didáticas e comparação intrasujeito.

---

## 11. Cobertura do cronograma (rastreabilidade)

| Semana | Entregável do plano | Estado no software |
|---|---|---|
| 1 | Escopo e revisão (ATM, visão comp.) | ✅ este documento + `README` |
| 2 | Detecção facial (MediaPipe/OpenCV), pontos | ✅ `landmarks.py`, `config.py` |
| 3 | Métricas + **critérios de classificação** + CSV | ✅ `metrics.py`, **`classification.py`**, `recorder.py` |
| 4 | Interface, biofeedback, gráficos de evolução | ✅ `app.py` (biofeedback c/ meta e cores, simetria), `plotting.py` |
| 5 | Testes e **validação** (iluminação/distância/movimento) | ✅ `analyze_video.py`, **`quality.py`**, `tests/` |
| 6 | Documentação e finalização | ✅ docs, limitações, este mapeamento |

**Documentação da evolução do paciente** (proposta: "exportar dados para
documentação da evolução do paciente"): `evolution.py` agrega cada sessão em uma
linha-resumo por paciente (`resultados/evolucao_<paciente>.csv`) e `plot_evolution`
gera o gráfico da tendência entre sessões — desvio mandibular (lateral em mm/rel e
angular) e simetria. Para desvio mandibular lateral, a melhora do tratamento
aparece como o desvio caminhando para 0. Em tempo real, a barra de centralização
do queixo (`_draw_centering_bar`) dá o biofeedback durante a coleta.

**Extensões além do escopo mínimo, guiadas pelas novas fontes:** índice de
simetria facial + `cant` bipupilar/intercomissural, proporções faciais (terços/
quintos), modo de perfil com ângulo de convexidade, detecção de rotação da
cabeça (yaw) e controle de qualidade da captura.
