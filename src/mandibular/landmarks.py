"""
Wrapper do MediaPipe Face Landmarker (Tasks API).

Encapsula a deteccao dos pontos anatomicos da face e devolve as coordenadas
em pixels, isolando o restante do sistema dos detalhes da biblioteca.

O modelo Face Landmarker produz 478 landmarks, com os mesmos indices do
modelo canonico Face Mesh usado em `config.Landmark`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .config import DetectionConfig

# Local padrao do modelo .task, relativo a raiz do projeto.
DEFAULT_MODEL_PATH = os.path.join("models", "face_landmarker.task")


@dataclass
class FaceLandmarks:
    """Landmarks de uma face detectada, em coordenadas de pixel (x, y).

    Opcionalmente carrega a profundidade relativa `depth` (coordenada z do
    modelo, reescalada para pixels), util para estimar a rotacao horizontal da
    cabeca (yaw). Pode ser None quando a fonte nao fornece z.

    `num_faces` e o numero TOTAL de faces detectadas no frame (nao so a
    principal): usado pelo controle de qualidade para rejeitar frames com
    mais de uma pessoa no quadro (ver DetectionConfig.max_num_faces e
    QualityConfig.reject_multiple_faces). Os pontos/depth aqui sempre se
    referem a face de maior confianca (indice 0); as demais nao sao usadas
    para metricas, apenas contadas.
    """
    points: np.ndarray          # shape (N, 2), float32, em pixels
    image_width: int
    image_height: int
    depth: np.ndarray | None = None  # shape (N,), profundidade relativa (px)
    num_faces: int = 1

    def point(self, index: int) -> np.ndarray:
        """Retorna o ponto (x, y) em pixels para o indice informado."""
        return self.points[index]

    @property
    def has_depth(self) -> bool:
        return self.depth is not None

    def z(self, index: int) -> float:
        """Profundidade relativa do ponto (0.0 se indisponivel)."""
        return 0.0 if self.depth is None else float(self.depth[index])


class FaceMeshDetector:
    """
    Detector de malha facial baseado no MediaPipe Face Landmarker (Tasks API).

    Uso:
        with FaceMeshDetector() as det:
            face = det.process(frame_bgr, timestamp_ms)
            if face is not None:
                p = face.point(Landmark.CHIN)
    """

    def __init__(self, config: DetectionConfig | None = None):
        self.config = config or DetectionConfig()
        model_path = self.config.model_path or DEFAULT_MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Modelo Face Landmarker nao encontrado em '{model_path}'.\n"
                "Baixe o modelo com:  python download_model.py\n"
                "ou informe o caminho via DetectionConfig.model_path."
            )

        # Importacao adiada: o mediapipe so e necessario em tempo de execucao.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self.config.max_num_faces,
            min_face_detection_confidence=self.config.min_detection_confidence,
            min_face_presence_confidence=self.config.min_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceLandmarks | None:
        """
        Processa um frame BGR (padrao do OpenCV) e retorna os landmarks da
        primeira face detectada, ou None se nenhuma face for encontrada.

        `timestamp_ms` deve ser monotonicamente crescente (exigencia do modo
        VIDEO da Tasks API).
        """
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb
        )
        result = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))

        if not result.face_landmarks:
            return None

        face = result.face_landmarks[0]
        pts = np.array(
            [(lm.x * w, lm.y * h) for lm in face],
            dtype=np.float32,
        )
        # z do modelo tem escala aproximada da largura normalizada; reescala
        # para pixels (x w) para ficar comparavel as coordenadas (x, y).
        depth = np.array([lm.z * w for lm in face], dtype=np.float32)
        return FaceLandmarks(
            points=pts, image_width=w, image_height=h, depth=depth,
            num_faces=len(result.face_landmarks),
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceMeshDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
