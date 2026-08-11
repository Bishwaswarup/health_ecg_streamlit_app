# Synthetic ECG Explorer

Public Streamlit app for two **educational, non-diagnostic** demonstrations:

- manually entering temperature, SpO₂, and heart-rate values for an optional
  cautious AI-generated summary; and
- generating artificial ECG waveforms and showing a model guess trained only
  on synthetic ECG classes.

It must not be used to diagnose, monitor, or make decisions about a person's
health. It does not upload or retain the manually entered values.

## Run locally

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

## Add the Gemini API key locally (optional)

Create `.streamlit/secrets.toml` on your own computer. It is ignored by Git:

```toml
GEMINI_API_KEY = "paste-your-key-here"
```

Without this key, the synthetic ECG page still works; only the optional AI
summary is disabled.

## Publish to Streamlit Community Cloud

1. Create a new GitHub repository and upload the *contents* of this folder.
2. At [Streamlit Community Cloud](https://share.streamlit.io/), select **Create app**.
3. Choose the repository and set the entrypoint to `app.py`.
4. In **Advanced settings → Secrets**, paste:

   ```toml
   GEMINI_API_KEY = "paste-your-key-here"
   ```

5. Deploy. Do not put the key in `app.py`, `requirements.txt`, or a GitHub
   commit.

The TensorFlow CNN is optional at runtime: when it cannot load, the app clearly
labels and uses its non-AI synthetic-template matcher instead.

The optional summary uses the stable `gemini-3.6-flash` model.
