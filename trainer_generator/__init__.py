"""
ecg_generator
=============

A synthetic ECG signal generator based on a five-Gaussian-wave beat
model (P, Q, R, S, T). Generates entirely synthetic data — no real
patient recordings are used — for testing, visualization, machine
learning, and educational purposes.

Quick start:

    >>> from trainer_generator import ECGGenerator
    >>> gen = ECGGenerator(sampling_rate=500, duration=10)
    >>> ecg = gen.generate("normal")
    >>> ecg.to_csv("normal.csv", index=False)
"""

from .generator import ECGGenerator
from .parameters import BeatMorphology, GaussianWaveParams, SimulationConfig

__all__ = [
    "ECGGenerator",
    "BeatMorphology",
    "GaussianWaveParams",
    "SimulationConfig",
]

__version__ = "1.0.0"
