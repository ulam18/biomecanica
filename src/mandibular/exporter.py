"""
Exportacao da sessao: cria uma pasta propria com CSV, resumo, metadados,
graficos e (opcionalmente) o video anotado.

Usa um identificador anonimo por padrao (sessao_AAAA-MM-DD_HH-MM-SS), sem
nome de paciente.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime

import numpy as np

from .metrics import (
    REL_UNIT_DESCRIPTION,
    CycleDetector,
    cycle_predominant_direction,
    lateral_direction,
)
from .plotting import (
    plot_cycles_mean_band,
    plot_cycles_normalized,
    plot_lateral_dynamic_time,
    plot_lateral_time,
    plot_opening_time,
    plot_trajectory,
    plot_trajectory_dynamic,
)
from .recorder import SessionRecorder, assign_cycle_ids

DISCLAIMER = (
    "Ferramenta de apoio funcional/didatico; nao substitui avaliacao "
    "profissional e nao constitui diagnostico de DTM."
)

# Convencao anatomica FIXA (independente de `mirrored`): os ROTULOS
# direita/esquerda (anatomical_direction, direcao_predominante etc.) e os
# campos "movimento_*"/"magnitude_*" do resumo ja sao calculados levando
# `mirrored` em conta (ver lateral_direction), entao por definicao positivo
# sempre corresponde a "direita" nesses campos derivados. As colunas BRUTAS
# (lateral_absolute_raw/filtered, lateral_dynamic_raw/filtered) mantêm o
# sinal de camera (dependente de mirrored) -- use os rotulos/campos
# derivados, nao o sinal bruto isolado, para uma leitura independente de
# mirrored.
LATERAL_SIGN_CONVENTION = "positivo=direita anatomica; negativo=esquerda anatomica"


def make_session_id(now: datetime | None = None) -> str:
    now = now or datetime.now()
    return "sessao_" + now.strftime("%Y-%m-%d_%H-%M-%S")


def _cycle_summaries(cycles: CycleDetector, mirrored: bool) -> list[dict]:
    deadzone = cycles.effective_direction_deadzone
    out = []
    for c in cycles.cycles:
        direction, value_used = cycle_predominant_direction(c, mirrored, deadzone=deadzone)
        out.append({
            "cycle_id": c.cycle_id,
            "inicio_s": round(c.start_time, 4),
            "fim_s": round(c.end_time, 4),
            "duracao_s": round(c.duration, 4),
            "abertura_maxima": round(c.peak_opening, 6),
            "amplitude": round(c.amplitude, 6),
            # limites exportados (secao 4 do refinamento): abertura na
            # amostra de inicio/fim, a banda "realmente fechada" usada para
            # valida-los, e se cada ponta de fato caiu dentro dela.
            "abertura_inicio": round(c.start_opening, 6),
            "abertura_fim": round(c.end_opening, 6),
            "limite_baseline_fechado": round(c.closed_baseline_limit, 6),
            "inicio_dentro_baseline": c.start_within_baseline,
            "fim_dentro_baseline": c.end_within_baseline,
            # origem/motivo do inicio exportado (secao 1/2 do refinamento):
            # "last_stable_closed_sample" no caso normal, "fallback" so
            # quando nao havia nenhuma amostra fechada conhecida antes do
            # movimento (ver metrics.CycleDetector._begin_cycle).
            "origem_inicio_ciclo": c.start_origin,
            "tempo_entre_anchor_e_confirmacao_s": round(c.anchor_confirm_gap_s, 4),
            "motivo_inicio_fora_baseline": (
                c.start_fallback_reason if not c.start_within_baseline else None
            ),
            "lateral_absolute_no_pico": c.lateral_absolute_at_peak,
            "lateral_dynamic_no_pico": c.lateral_dynamic_at_peak,  # preservado, nao substituido
            "lateral_dynamic_max_positivo": c.lateral_dynamic_max,
            "lateral_dynamic_min_negativo": c.lateral_dynamic_min,
            "lateral_dynamic_max_absoluto": c.lateral_dynamic_abs_max,
            # direcao pelo plato de abertura maxima (secao 3 do refinamento):
            # mediana de lateral_dynamic nas amostras >=95% do pico, mais
            # robusta a ruido isolado que o valor de um unico frame.
            "criterio_direcao_predominante": "mediana_lateral_dynamic_plato_95pct",
            "numero_amostras_plato": c.plateau_sample_count,
            "plato_usou_fallback_temporal": c.plateau_used_fallback,
            "lateral_dynamic_mediana_plato": c.lateral_dynamic_median_plateau,
            "effective_direction_deadzone": deadzone,
            "direcao_predominante": direction,
            "valor_usado_na_classificacao": value_used,
            "velocidade_media_abertura": round(c.opening_velocity, 6),
            "velocidade_media_fechamento": round(c.closing_velocity, 6),
        })
    return out


def _signed_extremes(lateral_dyn: np.ndarray, mirrored: bool) -> dict:
    """
    Extremos do movimento dinamico com sinal NORMALIZADO (positivo=direita
    anatomica, sempre, independente de mirrored) e magnitudes (>=0), para
    nao misturar valor assinado com magnitude na mesma leitura (secao 7).
    """
    if len(lateral_dyn) == 0:
        return {
            "movimento_max_direita": None,
            "movimento_min_esquerda": None,
            "magnitude_max_direita": None,
            "magnitude_max_esquerda": None,
        }
    positive_is_right = lateral_direction(1.0, mirrored) == "direita"
    anatomical = lateral_dyn if positive_is_right else -lateral_dyn
    movimento_max_direita = float(anatomical.max()) if np.any(anatomical > 0) else 0.0
    movimento_min_esquerda = float(anatomical.min()) if np.any(anatomical < 0) else 0.0
    return {
        "movimento_max_direita": movimento_max_direita,
        "movimento_min_esquerda": movimento_min_esquerda,
        "magnitude_max_direita": max(movimento_max_direita, 0.0),
        "magnitude_max_esquerda": abs(min(movimento_min_esquerda, 0.0)),
    }


def export_session(
    recorder: SessionRecorder,
    cycles: CycleDetector,
    session_dir: str,
    session_id: str,
    ref_mm: float | None = None,
    video_path: str | None = None,
    extra_metadata: dict | None = None,
    mirrored: bool = True,
) -> dict[str, str]:
    """Exporta a sessao na pasta `session_dir`; retorna os caminhos gerados."""
    if recorder.is_empty:
        raise ValueError("Nao ha amostras para exportar.")

    os.makedirs(session_dir, exist_ok=True)
    paths: dict[str, str] = {}

    # cycle_id so e conhecido apos o ciclo completar; precisa rodar ANTES do
    # CSV e dos graficos por ciclo.
    assign_cycle_ids(recorder.samples, cycles.cycles)

    csv_path = os.path.join(session_dir, "dados.csv")
    recorder.to_csv(csv_path)
    paths["csv"] = csv_path

    use_mm = ref_mm is not None
    paths["abertura_png"] = plot_opening_time(
        recorder, os.path.join(session_dir, "abertura_tempo.png"), use_mm=use_mm
    )
    paths["lateral_png"] = plot_lateral_time(
        recorder, os.path.join(session_dir, "lateralidade_tempo.png"),
        use_mm=use_mm, mirrored=mirrored,
    )
    paths["lateral_dinamica_png"] = plot_lateral_dynamic_time(
        recorder, os.path.join(session_dir, "lateralidade_dinamica_tempo.png"), mirrored=mirrored
    )
    paths["trajetoria_png"] = plot_trajectory(
        recorder,
        os.path.join(session_dir, "trajetoria_abertura_lateralidade.png"),
        use_mm=use_mm, mirrored=mirrored,
    )
    paths["trajetoria_dinamica_png"] = plot_trajectory_dynamic(
        recorder,
        os.path.join(session_dir, "trajetoria_abertura_lateralidade_dinamica.png"),
        mirrored=mirrored,
    )
    if cycles.cycles:
        paths["ciclos_png"] = plot_cycles_normalized(
            recorder, cycles, os.path.join(session_dir, "ciclos_individuais.png")
        )
        paths["ciclos_media_png"] = plot_cycles_mean_band(
            recorder, cycles, os.path.join(session_dir, "ciclos_curva_media.png")
        )

    # So os frames validos entram nas estatisticas de abertura/desvio: "*_filtered"
    # e None para frames invalidos (nao inventamos uma medicao 0.0 para eles).
    opening = np.array(
        [s.opening_filtered for s in recorder.samples if s.opening_filtered is not None]
    )
    lateral_abs = np.array(
        [s.lateral_filtered for s in recorder.samples if s.lateral_filtered is not None]
    )
    lateral_dyn = np.array(
        [s.lateral_dynamic_filtered for s in recorder.samples
         if s.lateral_dynamic_filtered is not None]
    )
    rep = cycles.repeatability()

    total_frames = len(recorder.samples)
    frames_com_face = sum(1 for s in recorder.samples if s.face_detected)
    frames_validos = sum(1 for s in recorder.samples if s.frame_valid)
    percentual_valido = (100.0 * frames_validos / total_frames) if total_frames else 0.0
    avisos_qualidade = dict(
        Counter(s.quality_warning for s in recorder.samples if s.quality_warning)
    )

    resumo = {
        "session_id": session_id,
        "total_frames": total_frames,
        "frames_com_face": frames_com_face,
        "frames_validos": frames_validos,
        "percentual_valido": round(percentual_valido, 1),
        "avisos_qualidade": avisos_qualidade,
        "calibrado": cycles.is_calibrated,
        "repeticoes": cycles.repetitions,
        "abertura_minima": float(opening.min()) if len(opening) else None,
        "abertura_maxima": float(opening.max()) if len(opening) else None,
        # "posicao do queixo" (lateral_absolute) e "movimento desde o neutro"
        # (lateral_dynamic) -- nomes apresentados ao usuario (secao 1); os
        # nomes de campo do JSON sao preservados por compatibilidade.
        "desvio_lateral_maximo_abs": float(np.abs(lateral_abs).max()) if len(lateral_abs) else None,
        "desvio_lateral_medio_abs": float(np.abs(lateral_abs).mean()) if len(lateral_abs) else None,
        "desvio_dinamico_medio_abs": float(np.abs(lateral_dyn).mean()) if len(lateral_dyn) else None,
        "repetibilidade": rep,
        "criterio_direcao_predominante": "mediana_lateral_dynamic_plato_95pct",
        "descricao_criterio_direcao": (
            "Mediana do movimento lateral desde o neutro nas amostras com abertura "
            "filtrada maior ou igual a 95% da abertura maxima do ciclo."
        ),
        "direction_deadzone_min": cycles.config.direction_deadzone_min,
        "direction_noise_multiplier": cycles.config.direction_noise_multiplier,
        "lateral_baseline_std": cycles.lateral_baseline_std,
        "effective_direction_deadzone": cycles.effective_direction_deadzone,
        "limiar_direcao": cycles.effective_direction_deadzone,  # alias (compat)
        "ciclos": _cycle_summaries(cycles, mirrored),
        "aviso": DISCLAIMER,
    }
    resumo.update(_signed_extremes(lateral_dyn, mirrored))
    # Campos antigos preservados (mesmo valor, ja eram magnitude >=0).
    resumo["desvio_dinamico_max_direita"] = resumo["magnitude_max_direita"]
    resumo["desvio_dinamico_max_esquerda"] = resumo["magnitude_max_esquerda"]

    if not cycles.is_calibrated:
        resumo["aviso_calibracao"] = (
            "Sessao NAO calibrada. A contagem de repeticoes (se houver) usa uma "
            "faixa dinamica de fallback (min/max observados), menos confiavel "
            "que os limiares de uma calibracao guiada (tecle C)."
        )
    if cycles.lateral_baseline is None:
        resumo["aviso_lateral_baseline"] = (
            "Sem lateral_neutral_baseline / posicao neutra calibrada (calibracao "
            "nao concluida ou sem amostras laterais na fase de boca fechada): "
            "'movimento desde o neutro' (lateral_dynamic) indisponivel nesta sessao."
        )
    resumo_path = os.path.join(session_dir, "resumo.json")
    with open(resumo_path, "w", encoding="utf-8") as f:
        json.dump(resumo, f, ensure_ascii=False, indent=2)
    paths["resumo"] = resumo_path

    if resumo["magnitude_max_direita"] is not None:
        print(f"[resumo] Maior movimento para a direita: {resumo['magnitude_max_direita']:.4f}")
        print(f"[resumo] Maior movimento para a esquerda: {resumo['magnitude_max_esquerda']:.4f}")

    metadados = {
        "session_id": session_id,
        "criado_em": datetime.now().isoformat(timespec="seconds"),
        "ref_mm": ref_mm,
        "calibrado": cycles.is_calibrated,
        "mirrored": mirrored,
        "lateral_neutral_baseline": cycles.lateral_baseline,
        "lateral_sign_convention": LATERAL_SIGN_CONVENTION,
        "unidade_metrica_relativa": REL_UNIT_DESCRIPTION,
        "direction_deadzone_min": cycles.config.direction_deadzone_min,
        "lateral_baseline_std": cycles.lateral_baseline_std,
        "direction_noise_multiplier": cycles.config.direction_noise_multiplier,
        "effective_direction_deadzone": cycles.effective_direction_deadzone,
    }
    if extra_metadata:
        metadados.update(extra_metadata)
    metadados_path = os.path.join(session_dir, "metadados.json")
    with open(metadados_path, "w", encoding="utf-8") as f:
        json.dump(metadados, f, ensure_ascii=False, indent=2)
    paths["metadados"] = metadados_path

    if video_path and os.path.exists(video_path):
        paths["video"] = video_path

    return paths
