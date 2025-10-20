import streamlit as st
import tempfile
import whisper
from transformers import pipeline

# ---------------------
# Load models
# ---------------------
st.title("✈️ ICAO Aviation English Proficiency Assessment")

@st.cache_resource
def load_models():
    # Whisper for speech-to-text
    whisper_model = whisper.load_model("base")

whisper_model = load_models()

# Upload audio
audio_file = st.file_uploader ("Upload the readback audio (WAV or MP3)", type=["wav", "mp3"])

# Expected readback reference (you can change this dynamically later)
expected_text = st.text_input("Expected readback:", "QNH one zero one three, cleared for takeoff runway two four.")

if audio_file is not None:
    st.audio(audio_file)

    if st.button("Run ICAO Assessment"):
        with st.spinner("Transcribing and evaluating..."):
            # Step 1: Transcription using Whisper
            transcript = openai.Audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

            transcription_text = transcript.text.strip()

            # Step 2: ICAO Rating Prompt
            rating_prompt = f"""
            You are an ICAO-qualified English Language Proficiency rater assessing student pilots’ spoken performance in aviation communication tasks.
            Apply the ICAO English Language Proficiency Rating Scale (Doc 9835) to evaluate the following readback.

            EXPECTED READBACK:
            "{expected_text}"

            TRANSCRIBED READBACK:
            "{transcription_text}"

            Provide ratings (1–6) and comments for:
            Pronunciation, Structure, Vocabulary, Fluency, Comprehension, and Interactions.
            Conclude with an overall ICAO level and a short feedback paragraph.
            Output structured JSON.
            """

            response = openai.Chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": rating_prompt}]
            )

            result = response.choices[0].message.content

        st.subheader("🗒️ Transcription")
        st.write(transcription_text)

        st.subheader("📊 ICAO Rating Result")
        st.json(result)

