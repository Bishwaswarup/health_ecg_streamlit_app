"""Public, educational dashboard for synthetic ECG and manually entered vitals.

This app deliberately does not diagnose.  The ECG traces and classifier were
created from synthetic data and are only suitable for demonstrating the UI and
the model workflow.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from ecg_ai import (
    DURATION_SECONDS,
    SAMPLING_RATE,
    CNN_LOAD_ERROR,
    draw_result,
    friendly_label,
    guess_label,
    make_prediction,
)
from trainer_generator.generator import ECGGenerator
from trainer_generator.rhythms import available_rhythms, recommended_heart_rate_range

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VitalFlag:
    name: str
    value: str
    level: str
    note: str


LEVEL_ORDER = {"green": 0, "yellow": 1, "red": 2}


def assess_vitals(temperature_c: float, spo2_percent: int, heart_rate_bpm: int) -> list[VitalFlag]:
    """Create conservative, adult-at-rest informational flags, not a diagnosis."""
    if spo2_percent <= 92:
        oxygen = VitalFlag("SpO₂", f"{spo2_percent}%", "red", "Low reading — repeat it promptly and seek urgent local medical assessment, especially with symptoms.")
    elif spo2_percent <= 94:
        oxygen = VitalFlag("SpO₂", f"{spo2_percent}%", "yellow", "Below the usual home-monitoring range — repeat after resting and seek prompt clinical advice if it persists or you feel unwell.")
    else:
        oxygen = VitalFlag("SpO₂", f"{spo2_percent}%", "green", "Not flagged by this simple adult-at-rest screen.")

    if temperature_c < 35.0 or temperature_c >= 39.0:
        temperature = VitalFlag("Temperature", f"{temperature_c:.1f} °C", "red", "Markedly outside this simple screen — repeat the measurement and seek urgent local medical advice if it persists or there are symptoms.")
    elif temperature_c < 36.0 or temperature_c >= 38.0:
        temperature = VitalFlag("Temperature", f"{temperature_c:.1f} °C", "yellow", "Outside the usual range; 38 °C or above is commonly treated as a fever in adults.")
    else:
        temperature = VitalFlag("Temperature", f"{temperature_c:.1f} °C", "green", "Not flagged by this simple adult-at-rest screen.")

    if heart_rate_bpm <= 40 or heart_rate_bpm >= 131:
        pulse = VitalFlag("Heart rate", f"{heart_rate_bpm} bpm", "red", "Markedly outside this simple adult-at-rest screen — repeat after resting and seek urgent local advice if it persists or there are symptoms.")
    elif heart_rate_bpm <= 49 or heart_rate_bpm >= 101:
        pulse = VitalFlag("Heart rate", f"{heart_rate_bpm} bpm", "yellow", "Outside the usual resting range — repeat after resting and discuss a persistent reading with a clinician.")
    else:
        pulse = VitalFlag("Heart rate", f"{heart_rate_bpm} bpm", "green", "Not flagged by this simple adult-at-rest screen.")
    return [temperature, oxygen, pulse]


def overall_level(flags: list[VitalFlag]) -> str:
    return max(flags, key=lambda flag: LEVEL_ORDER[flag.level]).level

st.set_page_config(page_title="BayMax : A Care Buddy", page_icon="❤️", layout="wide")
st.markdown(
    """<style>
    .stApp { background: radial-gradient(circle at top right, #1d3557 0%, #0b1020 42%, #070b14 100%); }
    [data-testid="stHeader"] { background: rgba(7, 11, 20, 0.78); }
    [data-testid="stMetric"] { background: rgba(24, 35, 58, 0.88); border: 1px solid #2a4168; border-radius: 14px; padding: 16px; }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 10px; }
    [data-testid="stTabs"] button { border-radius: 10px 10px 0 0; }
    .stButton > button { border-radius: 10px; font-weight: 650; }
    .vital-status { border-radius: 14px; padding: 15px 17px; min-height: 126px; border: 1px solid; }
    .vital-status h4 { margin: 0 0 8px; font-size: 1rem; }
    .vital-status .value { font-size: 1.65rem; font-weight: 700; margin-bottom: 8px; }
    .vital-status p { margin: 0; font-size: 0.88rem; line-height: 1.38; }
    .vital-status.green { background: #103c32; border-color: #2ecf8f; }
    .vital-status.yellow { background: #4a3b13; border-color: #f7c948; }
    .vital-status.red { background: #4a1f2b; border-color: #ff6b81; }
    </style>""",
    unsafe_allow_html=True,
)


def get_api_key() -> str | None:
    """Read a deployment secret without ever displaying it in the page."""
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def vital_summary(temperature_c: float, spo2_percent: int, heart_rate_bpm: int, flags: list[VitalFlag]) -> tuple[str, bool]:
    """Return a bounded, non-diagnostic educational summary."""
    api_key = get_api_key()
    if not api_key:
        return (
            "AI summary is not configured. Add `GEMINI_API_KEY` to the app's "
            "Streamlit secrets to enable it. These values are not stored by this app.",
            False,
        )
    try:
        from google import genai

        flag_details = "; ".join(f"{flag.name} {flag.value}: {flag.level.upper()} — {flag.note}" for flag in flags)
        prompt = f"""Give an educational, non-diagnostic explanation of 100–140 words.
Inputs: temperature {temperature_c:.1f} °C, SpO2 {spo2_percent}%, heart rate {heart_rate_bpm} bpm.
The app's deterministic informational flags are: {flag_details}
Explicitly discuss every flagged value and its colour level. Do not diagnose, claim medical clearance,
or recommend treatment. Say that a measurement can be inaccurate and that severe symptoms require
local emergency care. Use warm, plain language, not a generic disclaimer."""
        # Keep the client referenced until after the request completes. The
        # previous inline expression created a temporary client that could be
        # closed by the runtime before its models service sent the request.
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite", contents=prompt
        )
        return " ".join((response.text or "No summary was returned.").split()[:55]), True
    except Exception:
        logger.exception("Gemini summary request failed")
        return (
            "The AI summary could not be generated. Check that `GEMINI_API_KEY` "
            "is a valid Gemini API key in Streamlit Secrets and that its project "
            "has access to the configured Gemini model. See the app logs for the exact error.",
            False,
        )


@st.cache_data(show_spinner=False)
def create_case(rhythm: str, heart_rate: int, seed: int):
    generator = ECGGenerator(
        sampling_rate=SAMPLING_RATE,
        duration=DURATION_SECONDS,
        heart_rate=heart_rate,
        heart_rate_std=1.5,
        beat_variability=0.05,
        white_noise_std=0.01,
        random_seed=seed,
    )
    ecg = generator.generate(rhythm, seed=seed)
    labels = available_rhythms()
    prediction = make_prediction(ecg["Voltage"].to_numpy(), labels, seed)
    return ecg, prediction


st.title("BayMax : A Care Buddy")
st.caption("An educational demo to diagnose illness.")
st.warning("If someone has severe symptoms such as chest pain, trouble breathing, fainting, or signs of stroke, seek local emergency care now—not an online result.", icon="⚠️")

vitals_tab, ecg_tab = st.tabs(["Manual vital signs", "Synthetic ECG + model guess"])

with vitals_tab:
    st.subheader("Enter measurements")
    with st.form("vital_form"):
        first, second, third = st.columns(3)
        with first:
            temperature = st.number_input("Temperature (°C)", min_value=20.0, max_value=45.0, value=36.8, step=0.1)
        with second:
            spo2 = st.number_input("SpO₂ (%)", min_value=50, max_value=100, value=98, step=1)
        with third:
            bpm = st.number_input("Heart rate (bpm)", min_value=20, max_value=250, value=72, step=1)
        summarize = st.form_submit_button("Generate educational AI summary", type="primary")
    if summarize:
        flags = assess_vitals(float(temperature), int(spo2), int(bpm))
        st.session_state.vital_flags = flags
        with st.spinner("Creating a cautious summary…"):
            summary, succeeded = vital_summary(float(temperature), int(spo2), int(bpm), flags)
        st.session_state.vital_summary = summary
        st.session_state.vital_summary_succeeded = succeeded
    if "vital_flags" in st.session_state:
        flags = st.session_state.vital_flags
        level = overall_level(flags)
        heading = {"green": "Green: no value was flagged by this simple screen", "yellow": "Yellow: one or more values deserve a repeat check", "red": "Red: one or more values need urgent attention"}[level]
        st.subheader(heading)
        columns = st.columns(3)
        for column, flag in zip(columns, flags):
            column.markdown(
                f'<div class="vital-status {flag.level}"><h4>{flag.name} · {flag.level.upper()}</h4>'
                f'<div class="value">{flag.value}</div><p>{flag.note}</p></div>',
                unsafe_allow_html=True,
            )
        summary = st.session_state.vital_summary
        succeeded = st.session_state.vital_summary_succeeded
        (st.success if succeeded and level == "green" else st.warning if succeeded else st.error)(summary)
        st.caption("Traffic-light flags are an educational adult-at-rest screen, not a diagnosis. Recheck unexpected readings after resting; personalised targets may differ.")

with ecg_tab:
    st.subheader("Generate a synthetic test case")
    labels = available_rhythms()
    chosen_label = st.selectbox(
        "Generated rhythm", labels, format_func=friendly_label, help="This controls an artificial signal; it is not patient data."
    )
    low, high = recommended_heart_rate_range(chosen_label)
    rate = st.slider("Generated heart rate (bpm)", min_value=low, max_value=high, value=min(max(72, low), high))
    if "ecg_seed" not in st.session_state:
        st.session_state.ecg_seed = random.randint(1, 2_000_000_000)
    if st.button("Generate a new synthetic ECG", type="primary"):
        st.session_state.ecg_seed = random.randint(1, 2_000_000_000)

    with st.spinner("Generating and scoring a synthetic waveform…"):
        ecg, prediction = create_case(chosen_label, rate, st.session_state.ecg_seed)
        figure = draw_result(
            ecg["Time"].to_numpy(), ecg["Voltage"].to_numpy(), chosen_label, rate, prediction, top_count=5
        )
    st.pyplot(figure, clear_figure=True, width="stretch")
    plt.close(figure)
    one, two, three = st.columns(3)
    one.metric("Generated label", friendly_label(chosen_label))
    two.metric(guess_label(prediction), friendly_label(prediction.label))
    three.metric("Synthetic-class score", f"{prediction.confidence:.1f}%")
    st.caption(f"Scoring engine: {prediction.engine}. A score is not medical confidence or a diagnosis.")
    if CNN_LOAD_ERROR:
        st.info(f"The CNN was unavailable, so the app used the transparent template-matching fallback. Details: {CNN_LOAD_ERROR}")
