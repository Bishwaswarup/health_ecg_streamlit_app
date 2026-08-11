"""
rhythms.py
==========

Rhythm-specific beat planners for the synthetic ECG generator.

Every function in this module takes a :class:`~parameters.SimulationConfig`
and a NumPy random Generator, and returns a :class:`RhythmResult`
describing:

* a list of :class:`BeatEvent` objects (each an R-peak time plus the
  :class:`~parameters.BeatMorphology` to draw there), and
* an optional *overlay* callable for continuous, non-beat-locked
  activity (fibrillatory f-waves, flutter waves, or fully chaotic
  ventricular fibrillation) that :mod:`generator` adds on top of the
  beat-based signal.

All rhythms are built from the same five-Gaussian beat model; only the
*timing* (when beats occur, and whether P/QRS are linked) and the
*morphology* (wave amplitudes/widths/offsets) differ between rhythms.
Some rhythms (especially AFib, AFlutter and VFib) are, by nature,
irregular/chaotic and are only ever *approximations* suitable for
educational and testing purposes — they are not meant to be
diagnostic-quality simulations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Optional

import numpy as np

from .parameters import BeatMorphology, GaussianWaveParams, SimulationConfig

OverlayFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]


@dataclass
class BeatEvent:
    """A single beat to be rendered: where (R-peak time) and what shape."""

    r_peak_time: float
    morphology: BeatMorphology
    label: str = "N"


@dataclass
class RhythmResult:
    """Full plan for rendering a rhythm: discrete beats + optional overlay."""

    beats: List[BeatEvent]
    overlay: Optional[OverlayFn] = None
    description: str = ""


# --------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------

_SILENT = GaussianWaveParams(amplitude=0.0, center=0.0, width=0.01)


def jitter_morphology(
    morphology: BeatMorphology, rng: np.random.Generator, factor: float
) -> BeatMorphology:
    """Return a copy of ``morphology`` with small random amplitude/width
    jitter applied to every wave, controlled by ``factor`` (0 disables).
    """
    if factor <= 0:
        return morphology

    def jitter_wave(w: GaussianWaveParams) -> GaussianWaveParams:
        amp_factor = 1.0 + rng.normal(0, factor)
        width_factor = 1.0 + rng.normal(0, factor * 0.5)
        width_factor = max(width_factor, 0.2)
        return w.scaled(amplitude_factor=amp_factor, width_factor=width_factor)

    return replace(
        morphology,
        p_wave=jitter_wave(morphology.p_wave),
        q_wave=jitter_wave(morphology.q_wave),
        r_wave=jitter_wave(morphology.r_wave),
        s_wave=jitter_wave(morphology.s_wave),
        t_wave=jitter_wave(morphology.t_wave),
    )


def _without_p_wave(morphology: BeatMorphology) -> BeatMorphology:
    """Return a copy of ``morphology`` with the P wave suppressed
    (used for ventricular ectopy / escape beats / AFib, which lack an
    organized preceding P wave).
    """
    return replace(morphology, p_wave=_SILENT)


def _p_wave_only(morphology: BeatMorphology) -> BeatMorphology:
    """Return a copy of ``morphology`` containing only the P wave
    (used for non-conducted / dropped beats and for AV-dissociated P
    waves in third-degree block).
    """
    return replace(
        morphology,
        q_wave=_SILENT,
        r_wave=_SILENT,
        s_wave=_SILENT,
        t_wave=_SILENT,
    )


def _widened(morphology: BeatMorphology, factor: float) -> BeatMorphology:
    """Return a copy with the QRS complex widened by ``factor`` (bundle
    branch blocks, ventricular ectopy/escape all produce wide QRS).
    """
    return replace(morphology, qrs_widen_factor=morphology.qrs_widen_factor * factor)


def _regular_times(duration: float, bpm: float, hr_std: float, rng: np.random.Generator) -> List[float]:
    """Generate R-peak times for a (quasi-)regular rhythm with mild HRV."""
    times: List[float] = []
    t = 0.0
    while t < duration:
        times.append(t)
        beat_bpm = bpm
        if hr_std > 0:
            beat_bpm = max(15.0, rng.normal(bpm, hr_std))
        t += 60.0 / beat_bpm
    return times


def _irregular_times(
    duration: float, mean_bpm: float, irregularity: float, rng: np.random.Generator
) -> List[float]:
    """Generate R-peak times for an 'irregularly irregular' rhythm
    (used for atrial fibrillation), by drawing RR intervals from a
    gamma distribution whose coefficient of variation is controlled by
    ``irregularity``.
    """
    mean_rr = 60.0 / mean_bpm
    times: List[float] = []
    t = 0.0
    shape = max(1.0 / max(irregularity, 1e-3) ** 2, 1.5)
    scale = mean_rr / shape
    while t < duration:
        times.append(t)
        rr = rng.gamma(shape, scale)
        rr = float(np.clip(rr, 0.25, 2.0))
        t += rr
    return times


def _clip_bpm(config_bpm: float, low: float, high: float) -> float:
    """Clamp the user-provided heart rate into a physiologically
    plausible range for a given rhythm category.
    """
    return float(np.clip(config_bpm, low, high))


# Use non-overlapping representative rate ranges when synthesising labelled
# training/demo cases. The rhythm functions still clamp their own inputs, but
# these ranges avoid creating identical 100 bpm normal/tachycardia examples.
RHYTHM_HEART_RATE_RANGES: dict[str, tuple[int, int]] = {
    "normal": (60, 95),
    "sinus_bradycardia": (32, 48),
    "sinus_tachycardia": (105, 150),
    "atrial_fibrillation": (95, 145),
    "atrial_flutter": (60, 120),
    "premature_atrial_contraction": (60, 90),
    "premature_ventricular_contraction": (60, 90),
    "ventricular_tachycardia": (160, 220),
    "ventricular_fibrillation": (150, 220),
    "first_degree_av_block": (60, 95),
    "second_degree_av_block_mobitz1": (60, 90),
    "second_degree_av_block_mobitz2": (60, 90),
    "third_degree_av_block": (70, 95),
    "left_bundle_branch_block": (60, 95),
    "right_bundle_branch_block": (60, 95),
    "st_elevation": (60, 95),
    "st_depression": (60, 95),
}


def recommended_heart_rate_range(rhythm: str) -> tuple[int, int]:
    """Return a representative synthetic heart-rate range for a rhythm."""
    key = rhythm.strip().lower().replace(" ", "_").replace("-", "_")
    if key not in RHYTHM_HEART_RATE_RANGES:
        raise ValueError(f"Unknown rhythm: {rhythm}")
    return RHYTHM_HEART_RATE_RANGES[key]


def _base_morphology(config: SimulationConfig, rng: np.random.Generator) -> BeatMorphology:
    return jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability)


# --------------------------------------------------------------------------
# 1. Normal sinus rhythm
# --------------------------------------------------------------------------

def normal_sinus_rhythm(config: SimulationConfig, rng: np.random.Generator) -> RhythmResult:
    bpm = _clip_bpm(config.heart_rate, 50, 100)
    times = _regular_times(config.duration, bpm, config.heart_rate_std, rng)
    beats = [
        BeatEvent(t, jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability))
        for t in times
    ]
    return RhythmResult(beats, description="Normal sinus rhythm (50-100 bpm, regular).")


# --------------------------------------------------------------------------
# 2 & 3. Bradycardia / Tachycardia
# --------------------------------------------------------------------------

def sinus_bradycardia(config: SimulationConfig, rng: np.random.Generator) -> RhythmResult:
    bpm = _clip_bpm(config.heart_rate, 30, 50)
    times = _regular_times(config.duration, bpm, config.heart_rate_std, rng)
    beats = [
        BeatEvent(t, jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability))
        for t in times
    ]
    return RhythmResult(beats, description="Sinus bradycardia (<50 bpm, regular).")


def sinus_tachycardia(config: SimulationConfig, rng: np.random.Generator) -> RhythmResult:
    bpm = _clip_bpm(config.heart_rate, 100, 180)
    times = _regular_times(config.duration, bpm, config.heart_rate_std, rng)
    beats = [
        BeatEvent(t, jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability))
        for t in times
    ]
    return RhythmResult(beats, description="Sinus tachycardia (>100 bpm, regular).")


# --------------------------------------------------------------------------
# 4. Atrial fibrillation
# --------------------------------------------------------------------------

def atrial_fibrillation(config: SimulationConfig, rng: np.random.Generator) -> RhythmResult:
    bpm = _clip_bpm(config.heart_rate, 90, 170)
    times = _irregular_times(config.duration, bpm, irregularity=0.35, rng=rng)
    beats = []
    for t in times:
        m = jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability)
        m = _without_p_wave(m)  # no organized atrial depolarization
        beats.append(BeatEvent(t, m, label="AFib"))

    def overlay(t_arr: np.ndarray, orng: np.random.Generator) -> np.ndarray:
        # Fine, chaotic fibrillatory (f) waves at 350-600 "atrial events"/min.
        f_wave = np.zeros_like(t_arr)
        n_components = 6
        for _ in range(n_components):
            freq = orng.uniform(350, 600) / 60.0
            phase = orng.uniform(0, 2 * np.pi)
            amp = orng.uniform(0.01, 0.04)
            f_wave += amp * np.sin(2 * np.pi * freq * t_arr + phase)
        return f_wave / n_components * n_components  # keep amplitude scale

    return RhythmResult(
        beats,
        overlay=overlay,
        description="Atrial fibrillation: irregularly-irregular RR, no discrete P waves, fine f-wave baseline (synthetic approximation).",
    )


# --------------------------------------------------------------------------
# 5. Atrial flutter
# --------------------------------------------------------------------------

def atrial_flutter(config: SimulationConfig, rng: np.random.Generator) -> RhythmResult:
    atrial_rate = rng.uniform(250, 320)  # flutter (F) wave rate, bpm
    conduction_ratio = int(rng.choice([2, 3, 4]))
    ventricular_bpm = atrial_rate / conduction_ratio
    times = _regular_times(config.duration, ventricular_bpm, config.heart_rate_std * 0.5, rng)
    beats = []
    for t in times:
        m = jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability)
        m = _without_p_wave(m)  # flutter waves replace discrete P waves
        beats.append(BeatEvent(t, m, label="AFL"))

    def overlay(t_arr: np.ndarray, orng: np.random.Generator) -> np.ndarray:
        # Sawtooth 'flutter' (F) waves at the atrial rate.
        freq = atrial_rate / 60.0
        phase = orng.uniform(0, 2 * np.pi)
        # Sawtooth built from a Fourier series to avoid extra dependencies.
        wave = np.zeros_like(t_arr)
        for k in range(1, 6):
            wave += ((-1) ** (k + 1)) * np.sin(2 * np.pi * k * freq * t_arr + phase) / k
        wave *= 0.12  # scale to a realistic flutter-wave amplitude (mV)
        return wave

    return RhythmResult(
        beats,
        overlay=overlay,
        description=(
            f"Atrial flutter: sawtooth flutter waves at ~{atrial_rate:.0f} bpm with "
            f"{conduction_ratio}:1 AV conduction (synthetic approximation)."
        ),
    )


# --------------------------------------------------------------------------
# 6. Premature Atrial Contraction (PAC)
# --------------------------------------------------------------------------

def premature_atrial_contraction(config: SimulationConfig, rng: np.random.Generator) -> RhythmResult:
    bpm = _clip_bpm(config.heart_rate, 55, 95)
    base_times = _regular_times(config.duration, bpm, config.heart_rate_std, rng)
    beats: List[BeatEvent] = []
    # A labelled PAC window must contain at least one PAC. Otherwise a large
    # fraction of short training windows are indistinguishable from normal.
    forced_pac_index = int(rng.integers(1, len(base_times) - 1)) if len(base_times) > 2 else -1
    i = 0
    while i < len(base_times):
        t = base_times[i]
        m = jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability)
        if i > 0 and i < len(base_times) - 1 and (i == forced_pac_index or rng.random() < 0.12):
            # Fire early (premature) with a slightly different P-wave morphology,
            # followed by a short (incomplete) compensatory pause.
            prev_t = beats[-1].r_peak_time
            next_expected = base_times[i + 1]
            earliest = prev_t + 0.35
            pac_time = rng.uniform(earliest, max(earliest + 0.01, t))
            pac_m = replace(
                m,
                p_wave=GaussianWaveParams(
                    amplitude=m.p_wave.amplitude * rng.uniform(0.6, 1.3),
                    center=m.p_wave.center * rng.uniform(0.7, 1.2),
                    width=m.p_wave.width,
                ),
            )
            beats.append(BeatEvent(pac_time, pac_m, label="PAC"))
            # Incomplete compensatory pause: push the next sinus beat out slightly.
            base_times[i + 1] = pac_time + (next_expected - t) * rng.uniform(0.85, 1.0)
            i += 1
            continue
        beats.append(BeatEvent(t, m, label="N"))
        i += 1
    return RhythmResult(
        beats, description="Sinus rhythm with intermittent premature atrial contractions (PACs)."
    )


# --------------------------------------------------------------------------
# 7. Premature Ventricular Contraction (PVC)
# --------------------------------------------------------------------------

def premature_ventricular_contraction(config: SimulationConfig, rng: np.random.Generator) -> RhythmResult:
    bpm = _clip_bpm(config.heart_rate, 55, 95)
    base_times = _regular_times(config.duration, bpm, config.heart_rate_std, rng)
    beats: List[BeatEvent] = []
    # As with PAC, guarantee an observable PVC in every labelled PVC window.
    forced_pvc_index = int(rng.integers(1, len(base_times) - 1)) if len(base_times) > 2 else -1
    i = 0
    while i < len(base_times):
        t = base_times[i]
        m = jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability)
        if i > 0 and i < len(base_times) - 1 and (i == forced_pvc_index or rng.random() < 0.10):
            prev_t = beats[-1].r_peak_time
            next_expected = base_times[i + 1]
            earliest = prev_t + 0.30
            pvc_time = rng.uniform(earliest, max(earliest + 0.01, t))
            pvc_m = _without_p_wave(m)
            pvc_m = _widened(pvc_m, factor=2.2)
            pvc_m = replace(
                pvc_m,
                r_wave=GaussianWaveParams(
                    amplitude=m.r_wave.amplitude * rng.uniform(1.3, 1.8) * rng.choice([-1, 1]),
                    center=0.0,
                    width=m.r_wave.width * 2.2,
                ),
                t_wave=GaussianWaveParams(
                    amplitude=-abs(m.t_wave.amplitude) * 1.4,  # T wave typically opposite the QRS
                    center=m.t_wave.center * 1.4,
                    width=m.t_wave.width * 1.5,
                ),
            )
            beats.append(BeatEvent(pvc_time, pvc_m, label="PVC"))
            # Full compensatory pause: the following sinus beat lands on schedule
            # relative to the beat *before* the PVC (i.e. RR ~ 2x normal around it).
            base_times[i + 1] = prev_t + 2 * (next_expected - t)
            i += 1
            continue
        beats.append(BeatEvent(t, m, label="N"))
        i += 1
    return RhythmResult(
        beats,
        description=(
            "Sinus rhythm with intermittent premature ventricular contractions (PVCs): "
            "wide bizarre QRS, no preceding P wave, full compensatory pause."
        ),
    )


# --------------------------------------------------------------------------
# 8. Ventricular tachycardia
# --------------------------------------------------------------------------

def ventricular_tachycardia(config: SimulationConfig, rng: np.random.Generator) -> RhythmResult:
    bpm = _clip_bpm(max(config.heart_rate, 160), 150, 250)
    times = _regular_times(config.duration, bpm, config.heart_rate_std * 1.5, rng)
    beats = []
    for t in times:
        m = jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability)
        m = _without_p_wave(m)
        m = _widened(m, factor=2.0)
        m = replace(
            m,
            r_wave=GaussianWaveParams(
                amplitude=m.r_wave.amplitude * 1.6, center=0.0, width=m.r_wave.width * 2.0
            ),
            t_wave=GaussianWaveParams(
                amplitude=-abs(m.t_wave.amplitude) * 1.2,
                center=m.t_wave.center * 1.3,
                width=m.t_wave.width * 1.4,
            ),
        )
        beats.append(BeatEvent(t, m, label="VT"))
    return RhythmResult(
        beats,
        description="Ventricular tachycardia: regular wide-complex rhythm at 150-250 bpm, no P waves.",
    )


# --------------------------------------------------------------------------
# 9. Ventricular fibrillation (chaotic overlay only — no organized beats)
# --------------------------------------------------------------------------

def ventricular_fibrillation(config: SimulationConfig, rng: np.random.Generator) -> RhythmResult:
    def overlay(t_arr: np.ndarray, orng: np.random.Generator) -> np.ndarray:
        # Sum of many random-frequency, random-phase, slowly-modulated
        # sinusoids: a coarse but pedagogically useful stand-in for the
        # totally disorganized electrical activity of VFib. This is a
        # purely illustrative approximation, not a physiological model.
        chaos = np.zeros_like(t_arr)
        n_components = 25
        for _ in range(n_components):
            freq = orng.uniform(3, 9)  # Hz, coarse VFib undulation range
            phase = orng.uniform(0, 2 * np.pi)
            amp = orng.uniform(0.1, 0.6)
            # Slow amplitude modulation so the chaos isn't perfectly stationary.
            mod_freq = orng.uniform(0.05, 0.3)
            mod_phase = orng.uniform(0, 2 * np.pi)
            envelope = 0.6 + 0.4 * np.sin(2 * np.pi * mod_freq * t_arr + mod_phase)
            chaos += amp * envelope * np.sin(2 * np.pi * freq * t_arr + phase)
        chaos *= 0.5 / np.sqrt(n_components)
        return chaos

    return RhythmResult(
        beats=[],
        overlay=overlay,
        description=(
            "Ventricular fibrillation: chaotic, disorganized waveform with no identifiable "
            "P-QRS-T structure (educational synthetic approximation only)."
        ),
    )


# --------------------------------------------------------------------------
# 10. First-degree AV block
# --------------------------------------------------------------------------

def first_degree_av_block(config: SimulationConfig, rng: np.random.Generator) -> RhythmResult:
    bpm = _clip_bpm(config.heart_rate, 50, 100)
    times = _regular_times(config.duration, bpm, config.heart_rate_std, rng)
    prolonged_pr = rng.uniform(0.22, 0.32)  # seconds, > normal 0.12-0.20s
    beats = []
    for t in times:
        m = jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability)
        m = replace(
            m,
            p_wave=replace(m.p_wave, center=-prolonged_pr),
            pr_interval=prolonged_pr,
        )
        beats.append(BeatEvent(t, m))
    return RhythmResult(
        beats, description=f"First-degree AV block: constant prolonged PR interval (~{prolonged_pr:.2f}s)."
    )


# --------------------------------------------------------------------------
# 11. Second-degree AV block, Mobitz I (Wenckebach)
# --------------------------------------------------------------------------

def second_degree_av_block_mobitz1(config: SimulationConfig, rng: np.random.Generator) -> RhythmResult:
    atrial_bpm = _clip_bpm(config.heart_rate, 60, 90)
    atrial_rr = 60.0 / atrial_bpm
    group_size = int(rng.choice([3, 4, 5]))  # cycle of (group_size - 1) conducted beats + 1 dropped
    base_pr = 0.16
    pr_increment = 0.05

    beats: List[BeatEvent] = []
    t = 0.0
    while t < config.duration:
        for i in range(group_size - 1):
            if t >= config.duration:
                break
            pr = base_pr + i * pr_increment
            m = jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability)
            m = replace(m, p_wave=replace(m.p_wave, center=-pr), pr_interval=pr)
            beats.append(BeatEvent(t, m, label="N"))
            t += atrial_rr
        # Dropped beat: a lone P wave with no following QRS.
        if t < config.duration:
            m = jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability)
            beats.append(BeatEvent(t, _p_wave_only(m), label="P-only"))
            t += atrial_rr
    return RhythmResult(
        beats,
        description=(
            f"Second-degree AV block, Mobitz I (Wenckebach): PR interval progressively "
            f"lengthens over a {group_size - 1}-beat group until a P wave is not conducted."
        ),
    )


# --------------------------------------------------------------------------
# 12. Second-degree AV block, Mobitz II
# --------------------------------------------------------------------------

def second_degree_av_block_mobitz2(config: SimulationConfig, rng: np.random.Generator) -> RhythmResult:
    atrial_bpm = _clip_bpm(config.heart_rate, 60, 90)
    atrial_rr = 60.0 / atrial_bpm
    drop_every = int(rng.choice([3, 4, 5]))  # constant PR, fixed-ratio dropped beats
    pr = 0.18

    beats: List[BeatEvent] = []
    t = 0.0
    i = 0
    while t < config.duration:
        m = jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability)
        m = replace(m, p_wave=replace(m.p_wave, center=-pr), pr_interval=pr)
        if (i + 1) % drop_every == 0:
            beats.append(BeatEvent(t, _p_wave_only(m), label="P-only"))
        else:
            beats.append(BeatEvent(t, m, label="N"))
        t += atrial_rr
        i += 1
    return RhythmResult(
        beats,
        description=(
            f"Second-degree AV block, Mobitz II: constant PR interval with an unconducted "
            f"P wave every {drop_every} beats (fixed {drop_every}:{drop_every - 1} ratio)."
        ),
    )


# --------------------------------------------------------------------------
# 13. Third-degree (complete) AV block
# --------------------------------------------------------------------------

def third_degree_av_block(config: SimulationConfig, rng: np.random.Generator) -> RhythmResult:
    atrial_bpm = _clip_bpm(config.heart_rate, 70, 100)
    escape_bpm = rng.uniform(30, 45)  # ventricular escape rhythm, independent of atrial rate

    p_times = _regular_times(config.duration, atrial_bpm, hr_std=1.0, rng=rng)
    qrs_times = _regular_times(config.duration, escape_bpm, hr_std=0.5, rng=rng)

    beats: List[BeatEvent] = []
    for t in p_times:
        m = jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability)
        beats.append(BeatEvent(t, _p_wave_only(m), label="P-diss"))
    for t in qrs_times:
        m = jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability)
        m = _without_p_wave(m)
        m = _widened(m, factor=1.6)  # ventricular escape complexes are typically wide
        beats.append(BeatEvent(t, m, label="Escape"))

    beats.sort(key=lambda b: b.r_peak_time)
    return RhythmResult(
        beats,
        description=(
            f"Third-degree (complete) AV block: dissociated atrial activity (~{atrial_bpm:.0f} bpm) "
            f"and a slower, independent ventricular escape rhythm (~{escape_bpm:.0f} bpm)."
        ),
    )


# --------------------------------------------------------------------------
# 14 & 15. Bundle branch blocks
# --------------------------------------------------------------------------

def left_bundle_branch_block(config: SimulationConfig, rng: np.random.Generator) -> RhythmResult:
    bpm = _clip_bpm(config.heart_rate, 50, 100)
    times = _regular_times(config.duration, bpm, config.heart_rate_std, rng)
    beats = []
    for t in times:
        m = jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability)
        m = _widened(m, factor=1.8)  # QRS >= 120ms
        m = replace(
            m,
            q_wave=_SILENT,  # septal Q waves are typically absent in LBBB
            r_wave=replace(m.r_wave, amplitude=m.r_wave.amplitude * 1.15, width=m.r_wave.width * 1.6),
            t_wave=replace(m.t_wave, amplitude=-abs(m.t_wave.amplitude)),  # discordant T wave
        )
        beats.append(BeatEvent(t, m))
    return RhythmResult(
        beats,
        description="Left bundle branch block: widened QRS, absent septal Q wave, discordant T wave.",
    )


def right_bundle_branch_block(config: SimulationConfig, rng: np.random.Generator) -> RhythmResult:
    bpm = _clip_bpm(config.heart_rate, 50, 100)
    times = _regular_times(config.duration, bpm, config.heart_rate_std, rng)
    beats = []
    for t in times:
        m = jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability)
        m = _widened(m, factor=1.7)
        # Approximate the rSR' pattern by broadening/boosting the terminal S wave.
        m = replace(
            m,
            s_wave=replace(m.s_wave, amplitude=m.s_wave.amplitude * 1.8, width=m.s_wave.width * 1.9),
            t_wave=replace(m.t_wave, amplitude=-abs(m.t_wave.amplitude) * 0.8),
        )
        beats.append(BeatEvent(t, m))
    return RhythmResult(
        beats,
        description=(
            "Right bundle branch block: widened QRS with a broadened terminal S wave "
            "(simplified stand-in for the classic rSR' pattern)."
        ),
    )


# --------------------------------------------------------------------------
# 16 & 17. ST elevation / depression
# --------------------------------------------------------------------------

def st_elevation(config: SimulationConfig, rng: np.random.Generator) -> RhythmResult:
    bpm = _clip_bpm(config.heart_rate, 50, 110)
    times = _regular_times(config.duration, bpm, config.heart_rate_std, rng)
    offset = rng.uniform(0.15, 0.35)
    beats = [
        BeatEvent(
            t,
            replace(
                jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability),
                st_offset=offset,
            ),
        )
        for t in times
    ]
    return RhythmResult(beats, description=f"ST elevation: ST segment shifted +{offset:.2f} mV.")


def st_depression(config: SimulationConfig, rng: np.random.Generator) -> RhythmResult:
    bpm = _clip_bpm(config.heart_rate, 50, 110)
    times = _regular_times(config.duration, bpm, config.heart_rate_std, rng)
    offset = -rng.uniform(0.10, 0.25)
    beats = [
        BeatEvent(
            t,
            replace(
                jitter_morphology(BeatMorphology.default_normal(), rng, config.beat_variability),
                st_offset=offset,
            ),
        )
        for t in times
    ]
    return RhythmResult(beats, description=f"ST depression: ST segment shifted {offset:.2f} mV.")


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

RHYTHM_REGISTRY: dict[str, Callable[[SimulationConfig, np.random.Generator], RhythmResult]] = {
    "normal": normal_sinus_rhythm,
    "normal_sinus_rhythm": normal_sinus_rhythm,
    "sinus_bradycardia": sinus_bradycardia,
    "bradycardia": sinus_bradycardia,
    "sinus_tachycardia": sinus_tachycardia,
    "tachycardia": sinus_tachycardia,
    "atrial_fibrillation": atrial_fibrillation,
    "afib": atrial_fibrillation,
    "atrial_flutter": atrial_flutter,
    "aflutter": atrial_flutter,
    "premature_atrial_contraction": premature_atrial_contraction,
    "pac": premature_atrial_contraction,
    "premature_ventricular_contraction": premature_ventricular_contraction,
    "pvc": premature_ventricular_contraction,
    "ventricular_tachycardia": ventricular_tachycardia,
    "vtach": ventricular_tachycardia,
    "vt": ventricular_tachycardia,
    "ventricular_fibrillation": ventricular_fibrillation,
    "vfib": ventricular_fibrillation,
    "vf": ventricular_fibrillation,
    "first_degree_av_block": first_degree_av_block,
    "first_degree_block": first_degree_av_block,
    "second_degree_av_block_mobitz1": second_degree_av_block_mobitz1,
    "mobitz1": second_degree_av_block_mobitz1,
    "wenckebach": second_degree_av_block_mobitz1,
    "second_degree_av_block_mobitz2": second_degree_av_block_mobitz2,
    "mobitz2": second_degree_av_block_mobitz2,
    "third_degree_av_block": third_degree_av_block,
    "complete_heart_block": third_degree_av_block,
    "left_bundle_branch_block": left_bundle_branch_block,
    "lbbb": left_bundle_branch_block,
    "right_bundle_branch_block": right_bundle_branch_block,
    "rbbb": right_bundle_branch_block,
    "st_elevation": st_elevation,
    "stemi": st_elevation,
    "st_depression": st_depression,
}


def available_rhythms() -> List[str]:
    """Return the list of canonical rhythm keys (no aliases)."""
    canonical = [
        "normal",
        "sinus_bradycardia",
        "sinus_tachycardia",
        "atrial_fibrillation",
        "atrial_flutter",
        "premature_atrial_contraction",
        "premature_ventricular_contraction",
        "ventricular_tachycardia",
        "ventricular_fibrillation",
        "first_degree_av_block",
        "second_degree_av_block_mobitz1",
        "second_degree_av_block_mobitz2",
        "third_degree_av_block",
        "left_bundle_branch_block",
        "right_bundle_branch_block",
        "st_elevation",
        "st_depression",
    ]
    return canonical
