"""
Ponto de entrada do sistema de reconhecimento mandibular.

Exemplos:
    python run.py
    python run.py --camera 1 --ref-mm 63
    python run.py --width 640 --height 480 --no-flip
"""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from mandibular.app import MandibularApp
from mandibular.config import AppConfig


def parse_args() -> AppConfig:
    p = argparse.ArgumentParser(
        description="Sistema de reconhecimento mandibular digital (webcam)."
    )
    p.add_argument("--camera", type=int, default=0, help="indice da webcam (padrao: 0)")
    p.add_argument("--width", type=int, default=1280, help="largura de captura")
    p.add_argument("--height", type=int, default=720, help="altura de captura")
    p.add_argument("--no-flip", action="store_true", help="nao espelhar a imagem")
    p.add_argument(
        "--ref-mm",
        type=float,
        default=None,
        help="distancia real (mm) entre os cantos externos dos olhos, para "
        "converter as medidas em milimetros (opcional).",
    )
    p.add_argument("--output", default="resultados", help="pasta de saida")
    p.add_argument(
        "--video-fps", type=float, default=30.0,
        help="fps de codificacao do video anotado (o video e escrito por "
        "pacing pelo tempo real da sessao; nao depende da velocidade do "
        "pipeline). Padrao: 30.",
    )
    a = p.parse_args()

    cfg = AppConfig()
    cfg.camera_index = a.camera
    cfg.frame_width = a.width
    cfg.frame_height = a.height
    cfg.flip_horizontal = not a.no_flip
    cfg.reference_distance_mm = a.ref_mm
    cfg.output_dir = a.output
    cfg.video_output_fps = a.video_fps
    return cfg


def main() -> None:
    cfg = parse_args()
    app = MandibularApp(cfg)
    app.run()


if __name__ == "__main__":
    main()
