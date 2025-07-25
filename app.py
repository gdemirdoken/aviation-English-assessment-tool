import streamlit as st
import openai
import tempfile
import os

# Set API Key from environment variable
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ API key not found! Please set OPENAI_API_KEY in Streamlit Secrets.")
    st.stop()

# Configure OpenAI client
from openai import OpenAI
client = OpenAI(api_key=api_key)

# App UI
st.title("✈️ Aviation English Speaking Tutor")
st.write("Practice Aviation English speaking skills with AI feedback based on ICAO standards.")

# Scenario selection
scenario = st.selectbox("Choose a scenario:", ["Normal Operations", "Emergency"])
if scenario == "Normal Operations":
    prompt_text = "You are a pilot reporting your altitude and heading to ATC."
else:
    prompt_text = "Declare an engine failure and request priority landing."
st.write(f"### Scenario: {prompt_text}")

# Upload audio
audio_file = st.file_uploader("Upload your voice (MP3/WAV)", type=["mp3", "wav"])
if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    st.info("Transcribing your speech...")

    # ✅ Use OpenAI Speech-to-Text API (Whisper hosted by OpenAI)
    with open(temp_audio_path, "rb") as audio:
        transcription_response = client.audio.transcriptions.create(
            model="whisper-1",  # OpenAI's fast transcription model
            file=audio
        )

    transcription = transcription_response.text
    st.write("### Your Transcription:")
    st.write(transcription)

    # ✅ Generate AI Feedback
    st.info("Generating ICAO feedback...")
    gpt_prompt = f"""
    You are an ICAO Aviation English evaluator.
    The user said: "{transcription}".
    Evaluate based on ICAO descriptors: Pronunciation, Structure, Vocabulary, Fluency.
    Provide:
    - Score for each category (0-5)
    - Brief feedback in plain English
    - Suggested ICAO phraseology if needed.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Fast, affordable, latest model
        messages=[{"role": "system", "content": "You are an expert ICAO Aviation English evaluator."},
                  {"role": "user", "content": gpt_prompt}]
    )

    feedback = response.choices[0].message.content
    st.write("### AI Feedback:")
    st.write(feedback)
