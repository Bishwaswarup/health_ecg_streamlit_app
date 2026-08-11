"""
noise.py
========

Artifact and noise generators used to make synthetic ECG signals look
more realistic: baseline wander, additive white Gaussian noise,
high-frequency muscle (EMG) noise, and power-line interference.

Each function is a pure function of a time array (and an RNG for the
stochastic components) so they can be unit tested independently of the
main ECG generator.
"""

from __future__ import annotations

import numpy as np


def baseline_wander(
    t: np.ndarray,
    amplitude: float = 0.05,
    frequency: float = 0.2,
    rng: np.random.Generator | None = None,
    phase_jitter: bool = True,
) -> np.ndarray:
    """Simulate low-frequency baseline wander caused by respiration.

    Modeled as a slow sinusoid with a small amount of random phase
    drift so it does not look perfectly periodic.

    Args:
        t: Time array (seconds).
        amplitude: Peak amplitude in mV.
        frequency: Oscillation frequency in Hz (typically 0.15-0.3 Hz).
        rng: Random generator used for phase jitter. If None and
            ``phase_jitter`` is True, a fresh unseeded generator is
            used.
        phase_jitter: Whether to add a slowly varying random phase
            offset for extra realism.

    Returns:
        Array of the same shape as ``t`` with the wander signal.
    """
    if amplitude == 0:
        return np.zeros_like(t)
    if rng is None:
        rng = np.random.default_rng()
    phase = 0.0
    if phase_jitter:
        phase = rng.uniform(0, 2 * np.pi)
    wander = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    if phase_jitter:
        # Add a second, slower component for a more organic drift.
        slow_freq = frequency * 0.37
        slow_phase = rng.uniform(0, 2 * np.pi)
        wander += 0.3 * amplitude * np.sin(2 * np.pi * slow_freq * t + slow_phase)
    return wander


def white_noise(t: np.ndarray, std: float = 0.01, rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate additive white Gaussian noise.

    Args:
        t: Time array, used only for its length.
        std: Standard deviation of the noise in mV.
        rng: Random generator to draw from.

    Returns:
        Array of the same shape as ``t``.
    """
    if std == 0:
        return np.zeros_like(t)
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(loc=0.0, scale=std, size=t.shape)


def muscle_noise(
    t: np.ndarray,
    std: float = 0.02,
    rng: np.random.Generator | None = None,
    sampling_rate: float = 500.0,
) -> np.ndarray:
    """Simulate high-frequency EMG (muscle) noise.

    Implemented as white noise passed through a simple high-pass
    (difference) filter, which pushes energy toward higher frequencies
    similar to real muscle-artifact noise.

    Args:
        t: Time array.
        std: Target standard deviation of the resulting noise in mV.
        rng: Random generator to draw from.
        sampling_rate: Sampling rate in Hz (kept for API symmetry /
            potential future frequency-domain shaping).

    Returns:
        Array of the same shape as ``t``.
    """
    if std == 0:
        return np.zeros_like(t)
    if rng is None:
        rng = np.random.default_rng()
    raw = rng.normal(loc=0.0, scale=1.0, size=t.shape)
    # First-difference high-pass filter emphasizes high-frequency content.
    hp = np.diff(raw, prepend=raw[0])
    current_std = np.std(hp) or 1.0
    return hp * (std / current_std)


def powerline_interference(
    t: np.ndarray,
    amplitude: float = 0.0,
    frequency: float = 60.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate power-line (mains) interference.

    Args:
        t: Time array (seconds).
        amplitude: Peak amplitude in mV.
        frequency: Mains frequency, 50 Hz or 60 Hz.
        rng: Random generator used for a small random phase offset.

    Returns:
        Array of the same shape as ``t``.
    """
    if amplitude == 0:
        return np.zeros_like(t)
    if rng is None:
        rng = np.random.default_rng()
    phase = rng.uniform(0, 2 * np.pi)
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


def apply_all_noise(
    t: np.ndarray,
    clean_signal: np.ndarray,
    *,
    baseline_wander_amplitude: float = 0.0,
    baseline_wander_frequency: float = 0.2,
    white_noise_std: float = 0.0,
    muscle_noise_std: float = 0.0,
    powerline_amplitude: float = 0.0,
    powerline_frequency: float = 60.0,
    sampling_rate: float = 500.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compose all enabled noise/artifact sources and add them to a
    clean ECG signal.

    Args:
        t: Time array (seconds).
        clean_signal: The noise-free ECG signal.
        baseline_wander_amplitude: See :func:`baseline_wander`.
        baseline_wander_frequency: See :func:`baseline_wander`.
        white_noise_std: See :func:`white_noise`.
        muscle_noise_std: See :func:`muscle_noise`.
        powerline_amplitude: See :func:`powerline_interference`.
        powerline_frequency: See :func:`powerline_interference`.
        sampling_rate: Sampling rate in Hz.
        rng: Shared random generator for reproducibility.

    Returns:
        The signal with all requested noise sources added.
    """
    if rng is None:
        rng = np.random.default_rng()

    noisy = clean_signal.copy()
    noisy += baseline_wander(
        t, amplitude=baseline_wander_amplitude, frequency=baseline_wander_frequency, rng=rng
    )
    noisy += white_noise(t, std=white_noise_std, rng=rng)
    noisy += muscle_noise(t, std=muscle_noise_std, rng=rng, sampling_rate=sampling_rate)
    noisy += powerline_interference(
        t, amplitude=powerline_amplitude, frequency=powerline_frequency, rng=rng
    )
    return noisy
