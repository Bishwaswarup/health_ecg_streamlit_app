"""
parameters.py
=============

Configuration dataclasses used across the synthetic ECG generator.

This module defines:

* ``GaussianWaveParams`` — parameters (amplitude, center, width) for a
  single Gaussian "bump" used to model one deflection of the ECG
  (P, Q, R, S or T wave).
* ``BeatMorphology`` — a full beat template: five ``GaussianWaveParams``
  plus a nominal RR interval and ST-segment offset.
* ``SimulationConfig`` — global simulation settings (sampling rate,
  duration, heart rate, noise levels, variability, random seed, etc.)

All parameters are validated in ``__post_init__`` so that obviously
invalid configurations (negative sampling rate, zero duration, etc.)
fail fast with a clear error message instead of silently producing
garbage signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass
class GaussianWaveParams:
    """Parameters of a single Gaussian deflection.

    The deflection is evaluated as::

        G(t) = amplitude * exp(-(t - center) ** 2 / (2 * width ** 2))

    Attributes:
        amplitude: Peak height of the wave in millivolts (mV). May be
            negative to represent a downward deflection (e.g. Q, S).
        center: Offset in seconds from the fiducial R-peak of the beat
            at which the wave is centered. Negative values occur
            *before* the R peak, positive values *after*.
        width: Standard deviation (sigma) of the Gaussian in seconds.
            Controls how narrow/wide the deflection is. Must be > 0.
    """

    amplitude: float
    center: float
    width: float

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(
                f"GaussianWaveParams.width must be > 0, got {self.width!r}"
            )

    def scaled(self, amplitude_factor: float = 1.0, width_factor: float = 1.0) -> "GaussianWaveParams":
        """Return a copy scaled by the given amplitude/width factors."""
        return GaussianWaveParams(
            amplitude=self.amplitude * amplitude_factor,
            center=self.center,
            width=self.width * width_factor,
        )


@dataclass
class BeatMorphology:
    """Full template describing the shape of a single heartbeat.

    Attributes:
        p_wave: Parameters of the P wave (atrial depolarization).
        q_wave: Parameters of the Q wave.
        r_wave: Parameters of the R wave (main ventricular spike).
        s_wave: Parameters of the S wave.
        t_wave: Parameters of the T wave (ventricular repolarization).
        st_offset: Constant vertical offset (mV) applied to the segment
            between the S wave and T wave, used to simulate ST
            elevation/depression.
        pr_interval: Nominal PR interval in seconds (P onset to R peak),
            informational / used by some rhythm generators.
        qrs_widen_factor: Multiplicative factor applied to the width of
            Q, R and S waves, used to simulate bundle branch blocks
            (wide QRS complexes).
    """

    p_wave: GaussianWaveParams
    q_wave: GaussianWaveParams
    r_wave: GaussianWaveParams
    s_wave: GaussianWaveParams
    t_wave: GaussianWaveParams
    st_offset: float = 0.0
    pr_interval: float = 0.16
    qrs_widen_factor: float = 1.0

    def waves(self) -> list[GaussianWaveParams]:
        """Return all five waves as a list, honoring ``qrs_widen_factor``
        and ``st_offset``.
        """
        q = self.q_wave.scaled(width_factor=self.qrs_widen_factor)
        r = self.r_wave.scaled(width_factor=self.qrs_widen_factor)
        s = self.s_wave.scaled(width_factor=self.qrs_widen_factor)
        t = replace(self.t_wave, amplitude=self.t_wave.amplitude)
        return [self.p_wave, q, r, s, t]

    @staticmethod
    def default_normal() -> "BeatMorphology":
        """A physiologically plausible default beat template (in mV / s),
        loosely based on commonly cited ECG dynamical-model parameters
        (McSharry et al., 2003) but simplified to plain Gaussians.
        """
        return BeatMorphology(
            p_wave=GaussianWaveParams(amplitude=0.15, center=-0.20, width=0.025),
            q_wave=GaussianWaveParams(amplitude=-0.10, center=-0.05, width=0.010),
            r_wave=GaussianWaveParams(amplitude=1.20, center=0.00, width=0.010),
            s_wave=GaussianWaveParams(amplitude=-0.25, center=0.05, width=0.012),
            t_wave=GaussianWaveParams(amplitude=0.30, center=0.30, width=0.045),
            st_offset=0.0,
            pr_interval=0.16,
            qrs_widen_factor=1.0,
        )


@dataclass
class SimulationConfig:
    """Global configuration for an ECG simulation run.

    Attributes:
        sampling_rate: Samples per second (Hz). Must be > 0.
        duration: Total length of the generated signal in seconds.
            Must be > 0.
        heart_rate: Nominal heart rate in beats per minute (bpm).
            Must be > 0. Individual rhythm generators may override or
            modulate this per-beat.
        heart_rate_std: Standard deviation (bpm) of beat-to-beat heart
            rate variability (HRV). 0 disables HRV.
        beat_variability: Fractional (0-1) random jitter applied to
            each wave's amplitude and width every beat, to avoid an
            unnaturally perfect, repeating waveform.
        baseline_wander_amplitude: Peak amplitude (mV) of the simulated
            low-frequency baseline wander (respiration artifact). 0
            disables it.
        baseline_wander_frequency: Frequency (Hz) of baseline wander,
            typically ~0.15-0.3 Hz (respiration rate).
        white_noise_std: Standard deviation (mV) of additive white
            Gaussian noise. 0 disables it.
        muscle_noise_std: Standard deviation (mV) of high-frequency EMG
            (muscle) noise. 0 disables it.
        powerline_amplitude: Amplitude (mV) of simulated power-line
            interference. 0 disables it.
        powerline_frequency: Frequency (Hz) of power-line interference,
            50 Hz (Europe/Asia) or 60 Hz (US).
        smoothing: If True, apply a light moving-average smoothing
            filter to the final signal.
        smoothing_window: Window length (samples) for smoothing.
        random_seed: Optional seed for NumPy's random generator, for
            reproducible output.
    """

    sampling_rate: float = 500.0
    duration: float = 10.0
    heart_rate: float = 60.0
    heart_rate_std: float = 1.5
    beat_variability: float = 0.03
    baseline_wander_amplitude: float = 0.05
    baseline_wander_frequency: float = 0.2
    white_noise_std: float = 0.01
    muscle_noise_std: float = 0.0
    powerline_amplitude: float = 0.0
    powerline_frequency: float = 60.0
    smoothing: bool = False
    smoothing_window: int = 3
    random_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be > 0")
        if self.duration <= 0:
            raise ValueError("duration must be > 0")
        if self.heart_rate <= 0:
            raise ValueError("heart_rate must be > 0")
        if self.heart_rate_std < 0:
            raise ValueError("heart_rate_std must be >= 0")
        if not (0.0 <= self.beat_variability < 1.0):
            raise ValueError("beat_variability must be in [0, 1)")
        if self.baseline_wander_amplitude < 0:
            raise ValueError("baseline_wander_amplitude must be >= 0")
        if self.white_noise_std < 0:
            raise ValueError("white_noise_std must be >= 0")
        if self.muscle_noise_std < 0:
            raise ValueError("muscle_noise_std must be >= 0")
        if self.powerline_amplitude < 0:
            raise ValueError("powerline_amplitude must be >= 0")
        if self.powerline_frequency <= 0:
            raise ValueError("powerline_frequency must be > 0")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1")

    @property
    def num_samples(self) -> int:
        """Total number of samples in the generated signal."""
        return int(round(self.duration * self.sampling_rate))
