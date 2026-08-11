"""Public, educational dashboard for synthetic ECG and manually entered vitals.

This app deliberately does not diagnose.  The ECG traces and classifier were
created from synthetic data and are only suitable for demonstrating the UI and
the model workflow.
"""

from __future__ import annotations

import logging
import random

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

st.set_page_config(page_title="Synthetic ECG Explorer", page_icon="❤️", layout="wide")
st.markdown(
    """<style>
    .stApp { background: radial-gradient(circle at top right, #1d3557 0%, #0b1020 42%, #070b14 100%); }
    [data-testid="stHeader"] { background: rgba(7, 11, 20, 0.78); }
    [data-testid="stMetric"] { background: rgba(24, 35, 58, 0.88); border: 1px solid #2a4168; border-radius: 14px; padding: 16px; }
    [data-testid="stTabs"] [data-baseweb="tab-list"] { gap: 10px; }
    [data-testid="stTabs"] button { border-radius: 10px 10px 0 0; }
    .stButton > button { border-radius: 10px; font-weight: 650; }
    </style>""",
    unsafe_allow_html=True,
)


def get_api_key() -> str | None:
    """Read a deployment secret without ever displaying it in the page."""
    try:
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def vital_summary(temperature_c: float, spo2_percent: int, heart_rate_bpm: int) -> tuple[str, bool]:
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

        prompt = f"""Give an educational, non-diagnostic summary in 55 words or fewer.
Inputs: temperature {temperature_c:.1f} °C, SpO2 {spo2_percent}%, heart rate {heart_rate_bpm} bpm.
Do not diagnose, claim normality, or recommend treatment. Clearly state that a measurement can be inaccurate
and urgent symptoms require local emergency care. Use plain language."""
        response = genai.Client(api_key=api_key).models.generate_content(
            model="gemini-3.6-flash", contents=prompt
        )
        return " ".join((response.text or "No summary was returned.").split()[:55]), True
    except Exception:
        logger.exception("Gemini summary request failed")
        return (
            "The AI summary could not be generated. Check that `GEMINI_API_KEY` "
            "is a valid Gemini API key in Streamlit Secrets and that its project "
            "has access to Gemini 3.6 Flash. See the app logs for the exact error.",
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


st.title("Synthetic ECG Explorer")
st.caption("An educational demo. It does not diagnose illness, interpret a real ECG, or replace professional care.")
st.warning("If someone has severe symptoms such as chest pain, trouble breathing, fainting, or signs of stroke, seek local emergency care now—not an online result.", icon="⚠️")

vitals_tab, ecg_tab = st.tabs(["Manual vital signs", "Random synthetic ECG + model guess"])

with vitals_tab:
    st.subheader("Enter measurements")
    with st.form("vital_form"):
        first, second, third = st.columns(3)
        with first:
            temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=36.8, step=0.1)
        with second:
            spo2 = st.number_input("SpO₂ (%)", min_value=50, max_value=100, value=98, step=1)
        with third:
            bpm = st.number_input("Heart rate (bpm)", min_value=20, max_value=250, value=72, step=1)
        summarize = st.form_submit_button("Generate educational AI summary", type="primary")
    if summarize:
        with st.spinner("Creating a cautious summary…"):
            summary, succeeded = vital_summary(float(temperature), int(spo2), int(bpm))
        (st.success if succeeded else st.error)(summary)
        st.caption("This summary is generated from values you entered manually; it cannot validate sensor accuracy or make a diagnosis.")

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
