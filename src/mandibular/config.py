"""
Parametros de configuracao e indices dos pontos anatomicos.

Os indices seguem o modelo Face Mesh do MediaPipe (468 landmarks). Os pontos
foram escolhidos por serem estaveis e clinicamente interpretaveis para a
analise do movimento mandibular (ATM).
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Indices de landmarks do MediaPipe Face Mesh
# ---------------------------------------------------------------------------
# Referencia: pontos padronizados do modelo canonico de 468 vertices.
class Landmark:
    # Nariz
    NOSE_TIP = 1          # ponta do nariz (proxima a linha media)
    NASION = 168          # raiz do nariz, entre os olhos (referencia da linha media)

    # Boca / labios (parte interna, central) -> abertura bucal
    UPPER_LIP_INNER = 13  # centro do labio superior (borda interna)
    LOWER_LIP_INNER = 14  # centro do labio inferior (borda interna)

    # Cantos da boca -> largura bucal / extremidades
    MOUTH_LEFT = 61       # canto esquerdo da boca (lado direito da imagem)
    MOUTH_RIGHT = 291     # canto direito da boca (lado esquerdo da imagem)

    # Queixo
    CHIN = 152            # ponto mais inferior do queixo (menton)

    # Olhos -> referencia facial estavel para normalizacao e eixo horizontal
    EYE_OUTER_LEFT = 33   # canto externo do olho esquerdo
    EYE_OUTER_RIGHT = 263 # canto externo do olho direito
    EYE_INNER_LEFT = 133  # canto interno do olho esquerdo
    EYE_INNER_RIGHT = 362 # canto interno do olho direito

    # Pontos adicionais para simetria (pares esquerda/direita) e perfil
    NOSE_ALA_LEFT = 129   # asa do nariz, lado esquerdo
    NOSE_ALA_RIGHT = 358  # asa do nariz, lado direito
    CHEEK_LEFT = 234      # contorno da bochecha esquerda
    CHEEK_RIGHT = 454     # contorno da bochecha direita
    FOREHEAD = 10         # topo da testa (proximo ao trichion), linha media
    GLABELA = 9           # glabela (entre as sobrancelhas), linha media
    SUBNASALE = 2         # base do nariz / columela (subnasal)
    LABIALE_SUP = 0       # labiale superius (borda do labio superior)
    LABIALE_INF = 17      # labiale inferius (borda do labio inferior)
    SUBLABIALE = 200      # sulco mentolabial (entre labio inferior e mento)


# Conjunto de pontos desenhados/destacados na interface.
HIGHLIGHT_POINTS = [
    Landmark.NOSE_TIP,
    Landmark.NASION,
    Landmark.UPPER_LIP_INNER,
    Landmark.LOWER_LIP_INNER,
    Landmark.MOUTH_LEFT,
    Landmark.MOUTH_RIGHT,
    Landmark.CHIN,
    Landmark.EYE_OUTER_LEFT,
    Landmark.EYE_OUTER_RIGHT,
]


# Pares de landmarks simetricos (regiao, esquerda, direita) para o indice de
# simetria facial (vista frontal). Numa face frontal simetrica, cada par e o
# espelho do outro em relacao a linha media.
SYMMETRIC_PAIRS = [
    ("olhos", Landmark.EYE_OUTER_LEFT, Landmark.EYE_OUTER_RIGHT),
    ("olhos", Landmark.EYE_INNER_LEFT, Landmark.EYE_INNER_RIGHT),
    ("nariz", Landmark.NOSE_ALA_LEFT, Landmark.NOSE_ALA_RIGHT),
    ("boca", Landmark.MOUTH_LEFT, Landmark.MOUTH_RIGHT),
    ("face", Landmark.CHEEK_LEFT, Landmark.CHEEK_RIGHT),
]


@dataclass
class DetectionConfig:
    """Parametros do detector de face (MediaPipe Face Landmarker / Tasks API)."""
    max_num_faces: int = 2
    # 2 (nao 1): permite ao MediaPipe reportar quando ha MAIS DE UMA face no
    # quadro, para o controle de qualidade rejeitar o frame nesse caso (ver
    # QualityConfig.reject_multiple_faces). So o landmark[0] (a face de maior
    # confianca) e usado para as metricas, igual a antes -- pedir ate 2 faces
    # nao muda qual face e escolhida como principal numa cena com uma so
    # pessoa. Se isso causar instabilidade de rastreamento em algum ambiente,
    # reduza para 1 (desliga a deteccao de multiplas faces) ou desative via
    # QualityConfig.reject_multiple_faces=False sem tocar aqui.
    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # Caminho do modelo .task. Se None, usa models/face_landmarker.task.
    model_path: str | None = None


@dataclass
class FilterConfig:
    """
    Parametros do filtro EMA (media movel exponencial) aplicado aos sinais.

    Alpha alto = segue o sinal mais de perto (menos atraso, menos suavizacao);
    alpha baixo = mais suave, mais atraso. Valores por volta de 0.4-0.6 reduzem
    ruido de deteccao sem introduzir atraso perceptivel a ~30 fps.
    """
    alpha_opening: float = 0.5
    alpha_lateral: float = 0.5
    alpha_face_width: float = 0.3


@dataclass
class QualityConfig:
    """
    Limiares para classificar a qualidade do frame e orientar o usuario.

    IMPORTANTE: as razoes de tamanho facial sao normalizadas pela LARGURA do
    frame (face_width_px / frame_width_px), nao pela diagonal. A distancia
    interocular e uma medida essencialmente horizontal; normalizar pela
    diagonal faz o limiar variar com a proporcao da imagem (16:9 vs 4:3) sem
    motivo e, na pratica, exige que o rosto fique perto demais da camera
    (bug observado: 0.15*diagonal em 1280x720 exigia ~220px de distancia
    interocular, so atingivel a menos de ~30cm - por isso toda a sessao era
    marcada como invalida mesmo com o rosto claramente visivel no video).

    Como a razao e uma fracao da largura do frame, os mesmos valores
    funcionam em qualquer resolucao (640x480, 1280x720, etc.).
    """

    min_face_width_ratio: float = 0.06
    # face_width_px / frame_width_px minimo aceitavel. 0.06 corresponde a um
    # rosto a webcam a ~90-100cm de distancia (uso tipico de mesa); abaixo
    # disso ha poucos pixels entre os labios para medir a abertura com
    # confianca. Deliberadamente permissivo: uma face "um pouco distante"
    # mas ainda claramente utilizavel nao deve ser marcada invalida.
    max_face_width_ratio: float = 0.65
    # face_width_px / frame_width_px maximo aceitavel. 0.65 so e excedido com
    # o rosto extremamente proximo da camera (poucos cm), quando pequenos
    # movimentos ja tendem a levar os landmarks para fora da imagem.
    max_roll_deg: float = 30.0
    # inclinacao maxima da cabeca no plano da imagem (roll), em graus.
    max_yaw_deg: float = 35.0
    # rotacao horizontal maxima da cabeca (yaw; "rosto virado"), em graus
    # estimados por proxy de profundidade (sem estereo, ver metrics.HeadPose).
    # Permissivo de proposito: o desvio lateral mandibular por si so ja causa
    # uma leve rotacao aparente da face; um limiar apertado demais invalidaria
    # frames normais do proprio movimento sendo medido.
    max_pitch_deg: float = 25.0
    # inclinacao maxima da cabeca para cima/baixo (pitch), em graus estimados
    # por proxy de profundidade. A abertura bucal move o queixo e pode alterar
    # levemente o proxy; por isso tambem permissivo.
    max_global_jump_fraction: float = 0.25
    # deslocamento do nasion entre frames consecutivos, como fracao da
    # distancia interocular atual (cantos externos dos olhos, o denominador
    # usado em todas as metricas relativas); acima disso considera-se
    # movimento brusco.
    reject_multiple_faces: bool = True
    # invalida o frame quando o detector reporta mais de uma face (ver
    # DetectionConfig.max_num_faces). Isolado/configuravel: desligue aqui se
    # precisar manter max_num_faces=2 sem a rejeicao (ex.: depuracao).


@dataclass
class CycleConfig:
    """
    Parametros da deteccao de ciclos de abertura/fechamento.

    A contagem usa histerese sobre a abertura relativa: uma repeticao e
    contada quando o sinal ultrapassa o limiar de abertura e depois retorna
    abaixo do limiar de fechamento. Os limiares sao expressos como fracao da
    faixa (min..max) observada durante a calibracao.
    """
    open_fraction: float = 0.60    # fracao da faixa para considerar "aberto"
    close_fraction: float = 0.25   # fracao da faixa para considerar "fechado"
    min_cycle_seconds: float = 0.25  # ignora oscilacoes mais rapidas que isso (ruido)

    # -- Limites EXPORTADOS do ciclo (distintos dos limiares de histerese
    # acima, que continuam controlando so a maquina de estados/robustez a
    # ruido) -----------------------------------------------------------
    boundary_closed_fraction: float = 0.05
    # banda de "boca REALMENTE fechada" (closed_baseline_limit = baseline +
    # boundary_closed_fraction*span), usada para validar o inicio/fim
    # exportados de cada ciclo. Bem mais estreita que close_fraction (25%):
    # aquele so evita oscilacao da maquina de estados por ruido; este define
    # o que conta como "voltou ao fechado de verdade" para fins de relatorio.
    close_stability_seconds: float = 0.15
    # duracao MINIMA (segundos, nao numero de frames) que o sinal precisa
    # permanecer dentro de closed_baseline_limit, de forma continua, para
    # confirmar o fim do ciclo. Em segundos (nao frames) para o
    # comportamento ficar equivalente a 10, 15 ou 30 fps.
    prebuffer_seconds: float = 0.4
    # janela deslizante de tempo auxiliar (ver `last_stable_closed_sample` em
    # CycleDetector para a fonte PRINCIPAL do inicio exportado); usada so
    # como fallback quando nao ha nenhuma amostra fechada conhecida (ex.:
    # inicio de sessao ja em movimento).

    # -- Classificacao direita/esquerda/centro por ciclo -----------------
    direction_deadzone_min: float = 0.03
    direction_noise_multiplier: float = 3.0
    # effective_direction_deadzone = max(direction_deadzone_min,
    # direction_noise_multiplier * lateral_baseline_std) -- ver
    # CycleDetector.effective_direction_deadzone. Adapta o limiar de
    # classificacao ao ruido REAL medido na fase de boca fechada da
    # calibracao, em vez de um valor fixo (0.02 fixo se mostrou sensivel
    # demais a ruido de deteccao numa coleta real).
    direction_plateau_fraction: float = 0.95
    # amostras com abertura_filtrada >= direction_plateau_fraction*pico do
    # ciclo formam o "plato de abertura maxima"; a MEDIANA de
    # lateral_dynamic_filtered nesse plato (nao o valor de um unico frame no
    # pico, sensivel a ruido) e o criterio de classificacao da direcao.
    direction_min_plateau_samples: int = 3
    # abaixo disso, o plato e considerado curto demais para uma mediana
    # confiavel; cai no fallback de janela temporal ao redor do pico.
    direction_plateau_fallback_window_s: float = 0.1
    # meia-largura (segundos) da janela ao redor de peak_time usada quando o
    # plato tem poucas amostras (secao 3 do refinamento de precisao).


@dataclass
class AppConfig:
    """Configuracao geral da aplicacao."""
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    flip_horizontal: bool = True   # espelha a imagem (mais intuitivo p/ o usuario)
    draw_full_mesh: bool = False   # desenhar toda a malha (mais pesado)
    output_dir: str = "resultados"

    video_output_fps: float = 30.0
    # FPS de CODIFICACAO do video anotado (nao o FPS de processamento do
    # pipeline, que costuma ser MENOR). O video e escrito por "pacing": a
    # cada frame processado, escreve-se (repetindo se preciso) ate o indice
    # de frame correspondente ao tempo real decorrido da sessao, nesta taxa.
    # Isso mantem a duracao do MP4 igual a duracao real da sessao mesmo que
    # o pipeline nao consiga processar 30 fps (ver VideoRecorder.write_paced).

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    cycle: CycleConfig = field(default_factory=CycleConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

    # Calibracao opcional para converter unidades relativas em milimetros.
    # Se informado, e a distancia real (mm) entre os cantos externos dos olhos.
    reference_distance_mm: float | None = None
