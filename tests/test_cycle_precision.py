"""
Testes da recuperacao do inicio realmente fechado do ciclo
(last_stable_closed_sample / candidate_closed_anchor) e da classificacao de
direcao pelo plato de abertura maxima (mediana, nao o frame exato do pico).

Sinteticos, sem webcam. Usam CycleDetector diretamente.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mandibular.config import CycleConfig  # noqa: E402
from mandibular.metrics import CycleDetector, cycle_predominant_direction  # noqa: E402


def _close(det: CycleDetector, t0: float, value: float = 0.02, lateral_dynamic=None) -> None:
    """Fecha o ciclo em andamento: 2 frames na banda fechada, 0.2s de intervalo."""
    det.update(value, t0, lateral_dynamic=lateral_dynamic)
    det.update(value, t0 + 0.2, lateral_dynamic=lateral_dynamic)


# --------------------------------------------------------------------------
# last_stable_closed_sample / candidate_closed_anchor
# --------------------------------------------------------------------------
def test_slow_rise_exceeding_prebuffer_window_still_recovers_correct_start():
    """
    Abertura lenta levando bem mais que prebuffer_seconds (0.4s, padrao) ate
    confirmar: com o pre-buffer temporal isolado, a amostra fechada teria
    sido descartada da janela; com last_stable_closed_sample (congelado no
    candidato), o inicio correto e recuperado independente do tempo gasto.
    """
    det = CycleDetector(CycleConfig(min_cycle_seconds=0.0))
    det.calibrate(0.0, 1.0)  # open_th=0.6, closed_baseline_limit=0.05
    det.update(0.0, 0.0)
    det.update(0.0, 0.1)
    det.update(0.0, 0.2)   # ultima amostra fechada antes do movimento
    # rampa lenta: 1.3s ate confirmar (>> prebuffer_seconds=0.4s)
    det.update(0.10, 0.3)
    det.update(0.20, 0.5)
    det.update(0.30, 0.7)
    det.update(0.40, 0.9)
    det.update(0.50, 1.1)
    det.update(0.55, 1.3)
    det.update(0.90, 1.5)  # confirma (0.9 >= 0.6)
    _close(det, 1.6)

    assert det.repetitions == 1
    c = det.cycles[0]
    assert abs(c.start_time - 0.2) < 1e-9
    assert c.start_opening <= 0.05
    assert c.start_within_baseline is True
    assert c.start_origin == "last_stable_closed_sample"
    assert abs(c.anchor_confirm_gap_s - 1.3) < 1e-6


def test_last_stable_closed_sample_preserved_during_candidate_rise():
    """last_stable_closed_sample nao e sobrescrita enquanto o candidato esta em curso."""
    det = CycleDetector(CycleConfig(min_cycle_seconds=0.0))
    det.calibrate(0.0, 1.0)
    det.update(0.0, 0.0)
    det.update(0.0, 0.1)   # last_stable_closed_sample = t=0.1
    assert abs(det._last_stable_closed_sample[0] - 0.1) < 1e-9

    det.update(0.20, 0.2)  # sai da banda fechada -> congela o anchor em t=0.1
    assert det._candidate_anchor is not None
    assert abs(det._candidate_anchor[0] - 0.1) < 1e-9
    assert abs(det._last_stable_closed_sample[0] - 0.1) < 1e-9  # ainda a mesma

    det.update(0.30, 0.3)
    det.update(0.40, 0.4)
    # nem o anchor nem last_stable_closed_sample mudaram durante o candidato
    assert abs(det._candidate_anchor[0] - 0.1) < 1e-9
    assert abs(det._last_stable_closed_sample[0] - 0.1) < 1e-9


def test_candidate_anchor_frozen_not_overwritten_until_resolved():
    det = CycleDetector(CycleConfig(min_cycle_seconds=0.0))
    det.calibrate(0.0, 1.0)
    det.update(0.0, 0.0)
    det.update(0.0, 0.1)
    det.update(0.20, 0.2)
    anchor_1 = det._candidate_anchor
    det.update(0.90, 0.3)  # confirma -- so agora o anchor e consumido/limpo
    assert det._candidate_anchor is None  # consumido apos _begin_cycle
    c_pending_start = det._cycle_start_t
    assert abs(c_pending_start - anchor_1[0]) < 1e-9


def test_unconfirmed_candidate_discards_anchor_and_resumes_tracking():
    """Um candidato que nunca confirma descarta o anchor; last_stable volta a ser atualizada."""
    det = CycleDetector(CycleConfig(min_cycle_seconds=0.0))
    det.calibrate(0.0, 1.0)
    det.update(0.0, 0.0)
    det.update(0.0, 0.1)          # last_stable = t=0.1
    det.update(0.20, 0.2)         # candidato: congela anchor em t=0.1
    assert det._candidate_anchor is not None
    det.update(0.0, 0.3)          # volta ao fechado sem confirmar -> descarta o candidato
    assert det._candidate_anchor is None
    assert abs(det._last_stable_closed_sample[0] - 0.3) < 1e-9

    det.update(0.90, 0.4)         # movimento real subsequente
    _close(det, 0.5)

    assert det.repetitions == 1
    c = det.cycles[0]
    # usa a amostra fechada MAIS RECENTE (t=0.3), nao a antiga (t=0.1) nem
    # o pico do movimento descartado.
    assert abs(c.start_time - 0.3) < 1e-9
    assert c.start_within_baseline is True


def test_first_movement_without_prior_closed_sample_uses_documented_fallback():
    """Sem nenhuma amostra fechada observada (1o frame ja em movimento): fallback explicito."""
    det = CycleDetector(CycleConfig(min_cycle_seconds=0.0))
    det.calibrate(0.0, 1.0)
    det.update(0.90, 0.0)  # primeiro frame da sessao, ja confirmando abertura
    _close(det, 0.2)

    assert det.repetitions == 1
    c = det.cycles[0]
    assert c.start_origin == "fallback"
    assert c.start_fallback_reason is not None and len(c.start_fallback_reason) > 0
    assert c.start_within_baseline is False  # nao inventa um valor dentro da banda


def test_all_normal_cycles_start_within_baseline():
    """5 ciclos consecutivos (como a coleta real) devem comecar dentro da banda fechada."""
    det = CycleDetector(CycleConfig(min_cycle_seconds=0.0))
    det.calibrate(0.0, 1.0)
    det.update(0.0, 0.0)
    t = 0.1
    for _ in range(5):
        det.update(0.0, t); t += 0.1
        det.update(0.7, t); t += 0.1   # confirma
        det.update(1.0, t); t += 0.1   # pico
        det.update(0.02, t); t += 0.1  # 1o frame fechado
        det.update(0.02, t); t += 0.2  # confirma fechamento
        det.update(0.0, t); t += 0.1   # respiro fechado entre ciclos

    assert det.repetitions == 5
    assert all(c.start_within_baseline for c in det.cycles)
    assert all(c.end_within_baseline for c in det.cycles)


# --------------------------------------------------------------------------
# Direcao pelo plato de abertura maxima (mediana, nao o frame exato do pico)
# --------------------------------------------------------------------------
def _build_cycle_with_plateau(peak_values_and_lateral, deadzone=0.03):
    """
    peak_values_and_lateral: lista de (t, opening, lateral_dynamic) para o
    trecho de abertura (apos a confirmacao); fecha automaticamente no final.
    """
    det = CycleDetector(CycleConfig(min_cycle_seconds=0.0))
    det.calibrate(0.0, 1.0)
    det.update(0.0, 0.0)
    det.update(0.0, 0.1)
    for t, opening, lat in peak_values_and_lateral:
        det.update(opening, t, lateral_dynamic=lat)
    last_t = peak_values_and_lateral[-1][0]
    _close(det, last_t + 0.2, lateral_dynamic=peak_values_and_lateral[-1][2])
    assert det.repetitions == 1
    return det.cycles[0]


def test_direction_uses_plateau_median_not_single_noisy_peak_frame():
    """
    Reproduz o caso real: pico exato com lateral_dynamic ruidoso (+0.031,
    pouco acima da deadzone 0.03), mas a mediana do plato (~+0.0255) fica
    claramente dentro da zona "centro".
    """
    c = _build_cycle_with_plateau([
        (0.2, 0.70, 0.024),   # confirma abertura
        (0.3, 0.96, 0.025),   # plato
        (0.4, 1.00, 0.031),   # PICO exato -- valor ruidoso
        (0.5, 0.97, 0.026),   # plato
        (0.6, 0.96, 0.024),   # plato
    ])

    assert c.lateral_dynamic_at_peak == 0.031  # preservado, NAO substituido
    assert c.plateau_sample_count == 4
    assert abs(c.lateral_dynamic_median_plateau - 0.0255) < 1e-6
    assert c.plateau_used_fallback is False

    direction, value = cycle_predominant_direction(c, mirrored=True, deadzone=0.03)
    assert direction == "centro"
    assert abs(value - c.lateral_dynamic_median_plateau) < 1e-9


def test_plateau_median_positive_above_deadzone_is_direita():
    c = _build_cycle_with_plateau([
        (0.2, 0.70, 0.080),
        (0.3, 0.96, 0.080),
        (0.4, 1.00, 0.080),
        (0.5, 0.97, 0.080),
        (0.6, 0.96, 0.080),
    ])
    direction, value = cycle_predominant_direction(c, mirrored=True, deadzone=0.03)
    assert direction == "direita"
    assert abs(value - 0.080) < 1e-6


def test_plateau_median_negative_below_deadzone_is_esquerda():
    c = _build_cycle_with_plateau([
        (0.2, 0.70, -0.063),
        (0.3, 0.96, -0.063),
        (0.4, 1.00, -0.063),
        (0.5, 0.97, -0.063),
        (0.6, 0.96, -0.063),
    ])
    direction, value = cycle_predominant_direction(c, mirrored=True, deadzone=0.03)
    assert direction == "esquerda"
    assert abs(value - (-0.063)) < 1e-6


def test_plateau_fallback_to_time_window_when_few_plateau_samples():
    """Plato com poucas amostras (<direction_min_plateau_samples) cai no fallback temporal."""
    c = _build_cycle_with_plateau([
        (0.2, 0.70, 0.080),
        (0.3, 1.00, 0.080),   # unico frame realmente no plato (>=95% do pico)
        (0.4, 0.50, 0.080),   # cai rapido, fora do plato
    ])
    assert c.plateau_sample_count < 3
    assert c.plateau_used_fallback is True
    assert c.lateral_dynamic_median_plateau is not None  # fallback preencheu, nao ficou None


def test_lateral_dynamic_no_pico_preserved_alongside_plateau_median():
    c = _build_cycle_with_plateau([
        (0.2, 0.70, 0.010),
        (0.3, 0.96, 0.012),
        (0.4, 1.00, 0.050),   # pico com valor bem diferente da mediana do plato
        (0.5, 0.97, 0.011),
        (0.6, 0.96, 0.013),
    ])
    assert c.lateral_dynamic_at_peak == 0.050
    assert c.lateral_dynamic_median_plateau is not None
    assert c.lateral_dynamic_at_peak != c.lateral_dynamic_median_plateau
