"""
Testes da delimitacao COMPLETA dos ciclos (pre-buffer para o inicio real /
fechamento estavel por tempo, nao por contagem de frames) e da normalizacao
por fase dos graficos de ciclo (secoes 1-4 do refinamento de precisao).

Sinteticos, sem webcam. Usam CycleDetector diretamente (maquina de estados)
e plotting._cycle_progress_series (normalizacao 0-50-100%).
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mandibular.config import CycleConfig  # noqa: E402
from mandibular.metrics import CycleDetector, MovementState  # noqa: E402
from mandibular.plotting import _cycle_progress_series, plot_cycles_normalized  # noqa: E402
from mandibular.recorder import Sample, SessionRecorder, assign_cycle_ids  # noqa: E402


def _sample_at(i: int, t: float, opening: float) -> Sample:
    return Sample(
        session_id="s", frame=i, timestamp=f"{t:.3f}s", time_s=t,
        face_detected=True, frame_valid=True,
        opening_raw=opening * 200, opening_rel=opening, opening_filtered=opening, opening_mm=None,
        lateral_raw=0.0, lateral_rel=0.0, lateral_filtered=0.0, lateral_mm=None,
        direction="centro", cycle_state=MovementState.FECHADO, repetitions=0,
        quality_warning=None,
    )


# Sinal gradual com "hold" perto do baseline antes/depois do movimento
# (realista: o paciente fica parado de boca fechada um instante antes e
# depois de cada repeticao). calibrate(0.0, 1.0) -> open_th=0.6, close_th=
# 0.25, closed_baseline_limit=0.05 (boundary_closed_fraction padrao).
_GRADUAL_SCHEDULE = (
    [(round(0.1 * i, 4), 0.0) for i in range(4)]                                    # t=0.0..0.3, fechado
    + [(round(0.1 * i, 4), v) for i, v in zip(range(4, 9), [0.2, 0.4, 0.6, 0.8, 1.0])]   # abre
    + [(round(0.1 * i, 4), v) for i, v in zip(range(9, 14), [0.8, 0.6, 0.4, 0.2, 0.0])]  # fecha
    + [(round(0.1 * i, 4), 0.0) for i in range(14, 18)]                             # t=1.4..1.7, fechado
)


def _run_gradual_schedule(det: CycleDetector) -> None:
    for t, v in _GRADUAL_SCHEDULE:
        det.update(v, t)


def _build_gradual_cycle_recorder():
    det = CycleDetector(CycleConfig(min_cycle_seconds=0.0))
    det.calibrate(0.0, 1.0)
    rec = SessionRecorder()
    for i, (t, v) in enumerate(_GRADUAL_SCHEDULE):
        det.update(v, t)
        rec.add(_sample_at(i, t, v))
    assign_cycle_ids(rec.samples, det.cycles)
    return rec, det


# --------------------------------------------------------------------------
# Pre-buffer: recupera a ultima amostra fechada ANTES do cruzamento tardio
# --------------------------------------------------------------------------
def test_prebuffer_recovers_sample_before_confirmed_rise():
    """
    O ciclo deve comecar na ULTIMA amostra de boca fechada ANTES da subida
    (recuperada do pre-buffer), nao no frame tardio em que o limiar de
    abertura (60%) foi cruzado.
    """
    det = CycleDetector(CycleConfig(min_cycle_seconds=0.0))
    det.calibrate(0.0, 1.0)  # open_th=0.6, closed_baseline_limit=0.05
    det.update(0.0, 0.0)
    det.update(0.0, 0.1)
    det.update(0.0, 0.2)   # ultima amostra fechada antes do movimento
    det.update(0.9, 0.3)   # confirma imediatamente (degrau) -- mas o inicio
    #                        exportado deve ser t=0.2, nao t=0.3.
    det.update(0.0, 0.4)
    det.update(0.0, 0.5)
    det.update(0.0, 0.6)

    assert det.repetitions == 1
    c = det.cycles[0]
    assert abs(c.start_time - 0.2) < 1e-9
    assert c.start_opening <= 0.05
    assert c.start_within_baseline is True
    assert c.start_time < 0.3  # bem antes do frame de cruzamento do limiar


def test_small_movement_never_reaching_threshold_is_discarded():
    """
    Um movimento que sobe mas nunca atinge o limiar de abertura nao pode
    virar ciclo -- mesmo que passe perto do limiar de fechamento em algum
    ponto (a "descarta" e implicita: o pre-buffer so continua rolando).
    """
    det = CycleDetector(CycleConfig(min_cycle_seconds=0.0))
    det.calibrate(0.0, 1.0)  # open_th=0.6
    det.update(0.0, 0.0)
    det.update(0.30, 0.1)   # sobe um pouco...
    det.update(0.20, 0.2)   # ...mas nunca atinge 0.6
    det.update(0.0, 0.3)
    assert det.repetitions == 0
    assert det.cycles == []


def test_confirmed_movement_after_discarded_small_rise_uses_correct_start():
    """
    Depois de um movimento pequeno (nunca confirmado), um movimento real
    subsequente deve comecar na amostra fechada mais recente, nao arrastar
    o pico do movimento pequeno anterior para dentro do ciclo.
    """
    det = CycleDetector(CycleConfig(min_cycle_seconds=0.0))
    det.calibrate(0.0, 1.0)
    det.update(0.0, 0.0)
    det.update(0.30, 0.1)   # pequeno movimento, nunca confirmado
    det.update(0.0, 0.2)    # volta ao fechado -- amostra fechada mais recente
    det.update(0.9, 0.3)    # movimento real, confirma
    det.update(0.0, 0.4)
    det.update(0.0, 0.5)
    det.update(0.0, 0.6)

    assert det.repetitions == 1
    c = det.cycles[0]
    assert abs(c.start_time - 0.2) < 1e-9
    assert c.peak_opening <= 0.9 + 1e-9  # nao herda um pico espurio do movimento descartado


# --------------------------------------------------------------------------
# Delimitacao completa: inicio e fim proximos do baseline de boca fechada
# --------------------------------------------------------------------------
def test_cycle_boundaries_close_to_closed_baseline_with_gradual_signal():
    """
    Sinal gradual (rampa) com "hold" fechado antes/depois: verifica que o
    ciclo gravado comeca ANTES do limiar de abertura (60%) e termina dentro
    da banda "realmente fechada" (5%), preservando a mesma contagem (1 ciclo).
    """
    det = CycleDetector(CycleConfig(min_cycle_seconds=0.0))
    det.calibrate(0.0, 1.0)
    _run_gradual_schedule(det)

    assert det.repetitions == 1
    c = det.cycles[0]

    assert abs(c.peak_opening - 1.0) < 1e-6
    assert abs(c.peak_time - 0.8) < 1e-6
    assert c.baseline_opening == 0.0

    # Inicio: bem antes do limiar de abertura (0.6), dentro da banda fechada.
    assert c.start_time < 0.6
    assert abs(c.start_time - 0.3) < 1e-6
    assert c.start_opening <= 0.05
    assert c.start_within_baseline is True

    # Fim: dentro da banda fechada (nao so do limiar de fechamento de 25%).
    assert c.end_opening <= 0.05
    assert abs(c.end_time - 1.3) < 1e-6
    assert c.end_within_baseline is True

    # Duracao/velocidades usam os limites COMPLETOS (nao so o trecho entre limiares).
    assert abs(c.duration - 1.0) < 1e-6
    assert abs(c.opening_velocity - 2.0) < 1e-4
    assert abs(c.closing_velocity - 2.0) < 1e-4


def test_incomplete_cycle_still_discarded_with_new_state_machine():
    """Ciclo que abre mas a sessao termina antes do fechamento nunca e contado."""
    det = CycleDetector()
    det.calibrate(0.0, 1.0)
    det.update(0.0, 0.0)
    det.update(0.7, 0.1)
    det.update(0.9, 0.2)  # ainda ABERTO -- sessao "acaba" aqui, sem fechar
    assert det.repetitions == 0
    assert det.cycles == []


def test_cycle_behaviour_equivalent_across_fps():
    """
    A mesma sequencia de aberturas, reproduzida a 10, 15 e 30 fps (so muda o
    dt entre amostras), deve produzir 1 ciclo com inicio/fim dentro da banda
    fechada em todos os casos -- close_stability_seconds/prebuffer_seconds
    sao em tempo, nao em numero de frames.
    """
    shape = (
        [0.0, 0.0, 0.0, 0.0]
        + [0.2, 0.4, 0.6, 0.8, 1.0]
        + [0.8, 0.6, 0.4, 0.2, 0.0]
        + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # hold mais longo p/ garantir 0.15s mesmo a 10fps
    )
    for fps in (10, 15, 30):
        dt = 1.0 / fps
        det = CycleDetector(CycleConfig(min_cycle_seconds=0.0))
        det.calibrate(0.0, 1.0)
        t = 0.0
        for v in shape:
            det.update(v, t)
            t = round(t + dt, 6)
        assert det.repetitions == 1, fps
        c = det.cycles[0]
        assert c.start_within_baseline is True, fps
        assert c.end_within_baseline is True, fps
        assert abs(c.peak_opening - 1.0) < 1e-6, fps


# --------------------------------------------------------------------------
# Normalizacao por fase dos graficos: pico em 50%, extremos proximos do baseline
# --------------------------------------------------------------------------
def test_cycle_progress_peak_aligned_at_50_percent():
    rec, det = _build_gradual_cycle_recorder()
    assert det.repetitions == 1
    c = det.cycles[0]

    progress, openings = _cycle_progress_series(rec, c)
    assert len(progress) >= 2
    peak_idx = int(np.argmax(openings))
    assert abs(progress[peak_idx] - 0.5) < 1e-6


def test_cycle_progress_starts_and_ends_near_closed_baseline():
    rec, det = _build_gradual_cycle_recorder()
    c = det.cycles[0]
    progress, openings = _cycle_progress_series(rec, c)

    assert abs(progress[0] - 0.0) < 1e-6
    assert abs(progress[-1] - 1.0) < 1e-6
    # amostras em 0%/100% devem estar dentro da banda fechada (0.05), nao no
    # meio da abertura.
    assert openings[0] <= 0.05
    assert openings[-1] <= 0.05


def test_plot_cycles_normalized_generates_file():
    rec, det = _build_gradual_cycle_recorder()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ciclos.png")
        out = plot_cycles_normalized(rec, det, path)
        assert os.path.exists(out)
        assert os.path.getsize(out) > 0
