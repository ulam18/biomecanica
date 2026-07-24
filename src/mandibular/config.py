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


@dataclass
class DetectionConfig:
    """Parametros do detector de face (MediaPipe Face Landmarker / Tasks API)."""
    max_num_faces: int = 1
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
    max_global_jump_fraction: float = 0.25
    # deslocamento do nasion entre frames consecutivos, como fracao da
    # largura facial atual; acima disso considera-se movimento brusco.


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


@dataclass
class AppConfig:
    """Configuracao geral da aplicacao."""
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    flip_horizontal: bool = True   # espelha a imagem (mais intuitivo p/ o usuario)
    draw_full_mesh: bool = False   # desenhar toda a malha (mais pesado)
    output_dir: str = "resultados"

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    cycle: CycleConfig = field(default_factory=CycleConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

    # Calibracao opcional para converter unidades relativas em milimetros.
    # Se informado, e a distancia real (mm) entre os cantos externos dos olhos.
    reference_distance_mm: float | None = None
