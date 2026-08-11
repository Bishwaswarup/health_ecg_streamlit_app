"""
generator.py
============

Core ECG signal synthesis engine.

This module implements :class:`ECGGenerator`, the main entry point of
the package. It is responsible for:

1. Building a time axis from the configured sampling rate/duration.
2. Asking a rhythm-specific planner (see :mod:`rhythms`) for a list of
   beats (R-peak time + waveform shape) and an optional continuous
   overlay (flutter/fibrillation waves).
3. Rendering each beat as the sum of five Gaussian functions
   (P, Q, R, S, T) placed on the time axis, plus a smooth ST-segment
   offset for ST elevation/depression.
4. Adding the overlay (if any) and all requested noise sources
   (baseline wander, white noise, EMG noise, powerline interference).
5. Optionally smoothing the result and returning it as a
   ``pandas.DataFrame`` with ``Time`` and ``Voltage`` columns, ready
   for CSV export or plotting.

All generated data is entirely synthetic — no real patient recordings
are used or required anywhere in this package.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from . import noise as noise_mod
from .parameters import BeatMorphology, GaussianWaveParams, SimulationConfig
from .rhythms import RHYTHM_REGISTRY, RhythmResult, available_rhythms
from .utils import clip_signal, gaussian, get_rng, moving_average

# How many standard deviations of a Gaussian wave to render around its
# center before treating its contribution as negligible. This keeps
# rendering fast (only a small window around each wave needs updating)
# without changing the visible shape of the signal.
_SUPPORT_SIGMAS = 6.0


class ECGGenerator:
    """Generate synthetic ECG signals for a variety of rhythms.

    Example:
        >>> generator = ECGGenerator(sampling_rate=500, duration=10)
        >>> normal = generator.generate("normal")
        >>> normal.to_csv("normal.csv", index=False)

    Args:
        sampling_rate: Samples per second (Hz).
        duration: Length of the generated signal in seconds.
        heart_rate: Nominal heart rate in bpm (rhythm generators may
            clamp this into a physiologically appropriate range).
        config: A fully-specified :class:`SimulationConfig`. If given,
            it takes precedence over the individual keyword arguments
            above (which are provided for convenience).
        **config_overrides: Any other :class:`SimulationConfig` field
            (e.g. ``white_noise_std``, ``random_seed``, ``smoothing``)
            can be passed directly as a keyword argument.
    """

    def __init__(
        self,
        sampling_rate: float = 500.0,
        duration: float = 10.0,
        heart_rate: float = 60.0,
        config: Optional[SimulationConfig] = None,
        **config_overrides,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = SimulationConfig(
                sampling_rate=sampling_rate,
                duration=duration,
                heart_rate=heart_rate,
                **config_overrides,
            )
        self._rng = get_rng(self.config.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, rhythm: str, seed: Optional[int] = None) -> pd.DataFrame:
        """Generate a synthetic ECG for the given rhythm.

        Args:
            rhythm: Name of the rhythm to generate. See
                :func:`rhythms.available_rhythms` for the canonical
                list; common aliases (e.g. ``"afib"``, ``"pvc"``,
                ``"stemi"``) are also accepted.
            seed: Optional per-call random seed, overriding
                ``config.random_seed`` for just this call (useful for
                generating several independent examples from one
                generator instance).

        Returns:
            A ``pandas.DataFrame`` with two columns, ``Time`` (seconds)
            and ``Voltage`` (mV).

        Raises:
            ValueError: If ``rhythm`` is not a recognized rhythm name.
        """
        key = rhythm.strip().lower().replace(" ", "_").replace("-", "_")
        if key not in RHYTHM_REGISTRY:
            valid = ", ".join(available_rhythms())
            raise ValueError(f"Unknown rhythm '{rhythm}'. Available rhythms: {valid}")

        rng = get_rng(seed) if seed is not None else self._rng
        rhythm_fn = RHYTHM_REGISTRY[key]

        t = self._time_axis()
        plan: RhythmResult = rhythm_fn(self.config, rng)

        signal = self._render_beats(t, plan)
        if plan.overlay is not None:
            signal = signal + plan.overlay(t, rng)

        signal = noise_mod.apply_all_noise(
            t,
            signal,
            baseline_wander_amplitude=self.config.baseline_wander_amplitude,
            baseline_wander_frequency=self.config.baseline_wander_frequency,
            white_noise_std=self.config.white_noise_std,
            muscle_noise_std=self.config.muscle_noise_std,
            powerline_amplitude=self.config.powerline_amplitude,
            powerline_frequency=self.config.powerline_frequency,
            sampling_rate=self.config.sampling_rate,
            rng=rng,
        )

        if self.config.smoothing:
            signal = moving_average(signal, self.config.smoothing_window)

        signal = clip_signal(signal)

        return pd.DataFrame({"Time": t, "Voltage": signal})

    def generate_many(self, rhythms: list[str]) -> dict[str, pd.DataFrame]:
        """Convenience helper to generate several rhythms at once.

        Args:
            rhythms: List of rhythm names (see :meth:`generate`).

        Returns:
            Dict mapping each requested rhythm name to its DataFrame.
        """
        return {name: self.generate(name) for name in rhythms}

    @staticmethod
    def list_rhythms() -> list[str]:
        """Return the canonical list of supported rhythm names."""
        return available_rhythms()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _time_axis(self) -> np.ndarray:
        n = self.config.num_samples
        return np.arange(n) / self.config.sampling_rate

    def _render_beats(self, t: np.ndarray, plan: RhythmResult) -> np.ndarray:
        """Render all beats in ``plan`` onto the time axis ``t`` as the
        sum of Gaussian waves, using localized windows for efficiency.
        """
        signal = np.zeros_like(t)
        fs = self.config.sampling_rate
        n = len(t)

        for beat in plan.beats:
            morphology = beat.morphology
            waves = morphology.waves()

            # Include a synthetic "ST segment" Gaussian bump for ST
            # elevation/depression: a broad, flat-ish wave spanning
            # from the S wave to the T wave.
            if morphology.st_offset != 0.0:
                s_center = morphology.s_wave.center * morphology.qrs_widen_factor
                seg_center = (s_center + morphology.t_wave.center) / 2.0
                seg_width = max(abs(morphology.t_wave.center - s_center) / 2.2, 0.02)
                waves = waves + [
                    GaussianWaveParams(
                        amplitude=morphology.st_offset, center=seg_center, width=seg_width
                    )
                ]

            for wave in waves:
                if wave.amplitude == 0.0:
                    continue
                center_time = beat.r_peak_time + wave.center
                half_window = _SUPPORT_SIGMAS * wave.width
                start_idx = max(0, int((center_time - half_window) * fs))
                end_idx = min(n, int((center_time + half_window) * fs) + 1)
                if start_idx >= end_idx:
                    continue
                local_t = t[start_idx:end_idx] - beat.r_peak_time
                local_wave = GaussianWaveParams(
                    amplitude=wave.amplitude, center=wave.center, width=wave.width
                )
                signal[start_idx:end_idx] += gaussian(local_t, local_wave)

        return signal
