"""Generate, classify, and visualize a random synthetic ECG case.

This is an educational demonstration only. It generates artificial ECG data,
then asks a CNN trained on the same synthetic generator to classify it. A
displayed percentage is the model's score for a synthetic class, not clinical
certainty and must never be used for diagnosis or patient care.
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from trainer_generator.generator import ECGGenerator
from trainer_generator.rhythms import available_rhythms, recommended_heart_rate_range

# Keep Matplotlib's cache in a writable temporary directory. This must be set
# before TensorFlow or Matplotlib imports it, otherwise startup becomes slower.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ecg_ai_matplotlib"))


SAMPLING_RATE = 100
DURATION_SECONDS = 5
# Prefer a locally retrained model. Fall back to the compatible bundled model.
MODEL_DIRECTORY = Path(__file__).resolve().parent / "trainer_generator"
MODEL_PATH = (
    MODEL_DIRECTORY / "synthetic_ecg_classifier.keras"
    if (MODEL_DIRECTORY / "synthetic_ecg_classifier.keras").is_file()
    else MODEL_DIRECTORY / "synthetic_ecg_model_compat.keras"
)
CNN_LOAD_ERROR: str | None = None


@dataclass(frozen=True)
class Prediction:
    """One synthetic-class prediction and the source of its score."""

    labels: list[str]
    probabilities: np.ndarray
    engine: str

    @property
    def label(self) -> str:
        return self.labels[int(np.argmax(self.probabilities))]

    @property
    def confidence(self) -> float:
        return float(np.max(self.probabilities)) * 100


def friendly_label(label: str) -> str:
    """Turn a generator identifier into a display-ready rhythm name."""
    replacements = {
        "atrial_fibrillation": "Atrial fibrillation",
        "atrial_flutter": "Atrial flutter",
        "premature_atrial_contraction": "Premature atrial contraction",
        "premature_ventricular_contraction": "Premature ventricular contraction",
        "ventricular_tachycardia": "Ventricular tachycardia",
        "ventricular_fibrillation": "Ventricular fibrillation",
        "first_degree_av_block": "First-degree AV block",
        "second_degree_av_block_mobitz1": "Second-degree AV block (Mobitz I)",
        "second_degree_av_block_mobitz2": "Second-degree AV block (Mobitz II)",
        "third_degree_av_block": "Third-degree AV block",
        "left_bundle_branch_block": "Left bundle branch block",
        "right_bundle_branch_block": "Right bundle branch block",
        "st_elevation": "ST elevation",
        "st_depression": "ST depression",
    }
    return replacements.get(label, label.replace("_", " ").title())


@lru_cache(maxsize=1)
def load_cnn_model():
    """Load the CNN once per Python session; model loading is comparatively slow."""
    global CNN_LOAD_ERROR
    try:
        import tensorflow as tf
    except Exception as error:
        CNN_LOAD_ERROR = f"TensorFlow could not be imported ({error})"
        return None

    if not MODEL_PATH.is_file():
        CNN_LOAD_ERROR = f"Bundled model was not found at {MODEL_PATH}"
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as error:
        CNN_LOAD_ERROR = f"CNN model could not load ({type(error).__name__})"
        return None


def cnn_prediction(signal: np.ndarray, labels: list[str]) -> Prediction | None:
    """Use the bundled TensorFlow CNN when its optional runtime is installed."""
    model = load_cnn_model()
    if model is None:
        return None

    # Match the per-window standardisation used by train_ecg_ai.py.
    normalised = (signal - signal.mean()) / (signal.std() + 1e-8)
    probabilities = model.predict(normalised[np.newaxis, :, np.newaxis], verbose=0)[0]
    if len(probabilities) != len(labels):
        raise RuntimeError("The bundled CNN output does not match the current rhythm list.")
    return Prediction(labels, np.asarray(probabilities, dtype=float), "CNN trained on synthetic ECGs")


@lru_cache(maxsize=1)
def reference_templates(labels: tuple[str, ...]) -> np.ndarray:
    """Create fallback templates once, rather than once for every prediction."""
    reference_generator = ECGGenerator(
        sampling_rate=SAMPLING_RATE,
        duration=DURATION_SECONDS,
        heart_rate=72,
        heart_rate_std=0,
        beat_variability=0,
        white_noise_std=0,
        muscle_noise_std=0,
        baseline_wander_amplitude=0,
        powerline_amplitude=0,
        random_seed=0,
    )
    templates = []
    for rhythm in labels:
        reference = reference_generator.generate(rhythm, seed=0)["Voltage"].to_numpy()
        templates.append((reference - reference.mean()) / (reference.std() + 1e-9))
    return np.asarray(templates)


def template_prediction(signal: np.ndarray, labels: list[str], _seed: int | None) -> Prediction:
    """Provide a runnable, explicitly non-AI fallback when TensorFlow is absent."""
    target = (signal - signal.mean()) / (signal.std() + 1e-9)
    distances = np.mean((reference_templates(tuple(labels)) - target) ** 2, axis=1)

    # Convert similarity distances into scores that sum to 100% for the chart.
    scaled = -distances / (np.std(distances) + 1e-9)
    probabilities = np.exp(scaled - scaled.max())
    probabilities /= probabilities.sum()
    return Prediction(labels, probabilities, "Synthetic template matcher (not an AI model)")


def make_prediction(signal: np.ndarray, labels: list[str], seed: int | None) -> Prediction:
    """Prefer the CNN; otherwise return the transparent non-AI fallback."""
    return cnn_prediction(signal, labels) or template_prediction(signal, labels, seed)


def guess_label(prediction: Prediction) -> str:
    """Avoid presenting the non-AI fallback as an AI prediction."""
    return "AI guess" if prediction.engine.startswith("CNN") else "Similarity guess"


def draw_result(
    time_axis: np.ndarray,
    signal: np.ndarray,
    true_label: str,
    heart_rate: int,
    prediction: Prediction,
    top_count: int,
):
    """Build a presentation-ready ECG plot and confidence comparison."""
    import matplotlib.pyplot as plt

    ranking = np.argsort(prediction.probabilities)[::-1][:top_count]
    is_correct = prediction.label == true_label
    color = "#16803c" if is_correct else "#c0392b"

    figure = plt.figure(figsize=(14, 8), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(2.25, 1))
    ecg_axis = figure.add_subplot(grid[0])
    score_axis = figure.add_subplot(grid[1])

    ecg_axis.set_facecolor("#fff7f7")
    ecg_axis.plot(time_axis, signal, color=color, linewidth=1.3)
    ecg_axis.set_title(
        f"Synthetic ECG case | {guess_label(prediction)}: {friendly_label(prediction.label)} "
        f"({prediction.confidence:.1f}%)",
        fontweight="bold",
    )
    ecg_axis.set_xlabel("Time (seconds)")
    ecg_axis.set_ylabel("Voltage (mV)")
    ecg_axis.grid(True, which="major", color="#efb5b5", linewidth=0.7)
    ecg_axis.minorticks_on()
    ecg_axis.grid(True, which="minor", color="#f8dddd", linewidth=0.4)
    ecg_axis.text(
        0.01,
        0.95,
        f"Generated case: {friendly_label(true_label)}\n"
        f"Generated pulse: {heart_rate} bpm\n"
        f"Result: {'match' if is_correct else 'different guess'}\n"
        f"Engine: {prediction.engine}",
        transform=ecg_axis.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": color, "alpha": 0.95},
    )

    scores = prediction.probabilities[ranking] * 100
    names = [friendly_label(prediction.labels[index]) for index in ranking]
    bars = score_axis.barh(names[::-1], scores[::-1], color="#4c78a8")
    bars[-1].set_color(color)
    score_axis.set_xlim(0, 100)
    score_axis.set_xlabel("Model score (%)")
    score_axis.set_title("Top synthetic-class scores", loc="left", fontweight="bold")
    score_axis.grid(axis="x", alpha=0.25)
    for bar, score in zip(bars[::-1], scores):
        score_axis.text(score + 1, bar.get_y() + bar.get_height() / 2, f"{score:.1f}%", va="center")

    figure.text(
        0.5,
        0.005,
        "Synthetic demonstration only — model score is not clinical confidence or a diagnosis.",
        ha="center",
        color="#6b1b1b",
        fontsize=9,
    )
    return figure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and classify a random synthetic ECG case.")
    parser.add_argument("--seed", type=int, help="Optional seed for a repeatable case")
    parser.add_argument("--top", type=int, default=5, help="Number of class scores to display (default: 5)")
    parser.add_argument("--save", type=Path, help="Optional path to save a copy; the chart still opens unless --no-show is used")
    parser.add_argument("--no-show", action="store_true", help="Create the result without opening a window")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.top < 1:
        raise SystemExit("--top must be at least 1")

    labels = available_rhythms()
    randomizer = random.Random(args.seed)
    true_label = randomizer.choice(labels)
    rate_low, rate_high = recommended_heart_rate_range(true_label)
    heart_rate = randomizer.randint(rate_low, rate_high)
    generator = ECGGenerator(
        sampling_rate=SAMPLING_RATE,
        duration=DURATION_SECONDS,
        heart_rate=heart_rate,
        heart_rate_std=1.5,
        beat_variability=0.05,
        white_noise_std=0.01,
        random_seed=args.seed,
    )
    ecg = generator.generate(true_label)
    signal = ecg["Voltage"].to_numpy()
    prediction = make_prediction(signal, labels, args.seed)

    print("\n=== SYNTHETIC ECG AI DEMO ===")
    print(f"Generated case: {friendly_label(true_label)}")
    print(f"Generated pulse: {heart_rate} bpm")
    print(f"{guess_label(prediction)}: {friendly_label(prediction.label)} ({prediction.confidence:.1f}%)")
    print(f"Engine:         {prediction.engine}")
    if CNN_LOAD_ERROR:
        print(f"CNN status:     unavailable — using fallback ({CNN_LOAD_ERROR})")
    print("Note: this score is for synthetic classes only, not clinical confidence.")

    if args.no_show and not args.save:
        return 0

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required to display the ECG result. Run: pip install -r requirements.txt") from error
    backend = plt.get_backend()
    if backend.lower() == "agg":
        raise SystemExit(
            "Matplotlib is using the non-interactive Agg backend, so it cannot open a plot window. "
            "Run the script from a desktop Python environment, or use --save result.png."
        )

    print(f"Opening ECG result window (Matplotlib backend: {backend})…")
    figure = draw_result(ecg["Time"].to_numpy(), signal, true_label, heart_rate, prediction, min(args.top, len(labels)))
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.save, dpi=180, bbox_inches="tight")
        print(f"Saved result: {args.save.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
