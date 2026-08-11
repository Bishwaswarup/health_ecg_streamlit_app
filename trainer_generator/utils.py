"""
utils.py
========

Small, dependency-light helper functions shared across the ECG
generator package: Gaussian wave evaluation, RNG handling, and moving
average smoothing.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .parameters import GaussianWaveParams


def gaussian(t: np.ndarray, params: GaussianWaveParams) -> np.ndarray:
    """Evaluate a single Gaussian wave over a time array.

    G(t) = A * exp(-(t - mu) ** 2 / (2 * sigma ** 2))

    Args:
        t: Array of time values (seconds), typically relative to the
            beat's fiducial R-peak.
        params: Amplitude, center (mu) and width (sigma) of the wave.

    Returns:
        Array of the same shape as ``t`` with the Gaussian evaluated
        at each point.
    """
    return params.amplitude * np.exp(
        -((t - params.center) ** 2) / (2.0 * params.width ** 2)
    )


def get_rng(seed: Optional[int]) -> np.random.Generator:
    """Return a NumPy random Generator, seeded if ``seed`` is given."""
    return np.random.default_rng(seed)


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Apply a simple centered moving-average smoothing filter.

    Edges are handled with 'same' convolution mode so the output array
    has the same length as the input.

    Args:
        signal: 1-D array to smooth.
        window: Window length in samples. Values <= 1 return the
            signal unchanged.

    Returns:
        Smoothed array, same shape as ``signal``.
    """
    if window <= 1:
        return signal
    kernel = np.ones(window) / window
    return np.convolve(signal, kernel, mode="same")


def bpm_to_rr_seconds(bpm: float) -> float:
    """Convert a heart rate in beats-per-minute to an RR interval in seconds."""
    if bpm <= 0:
        raise ValueError("bpm must be > 0")
    return 60.0 / bpm


def clip_signal(signal: np.ndarray, min_val: float = -5.0, max_val: float = 5.0) -> np.ndarray:
    """Clip a signal to a physiologically sane voltage range (mV).

    This is a safety net against runaway values from extreme parameter
    combinations; normal synthetic ECGs will never approach these
    bounds.
    """
    return np.clip(signal, min_val, max_val)
