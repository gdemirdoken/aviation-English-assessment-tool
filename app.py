import streamlit as st
import io
import os
from openai import OpenAI

# Create OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("🛫 ICAO English Proficiency Assessment Tool")

from openai.error import RateLimitError
import time

while True:
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": rating_prompt}]
        )
        break  # success
    except RateLimitError:
        st.warning("Rate limit reached. Waiting 5 seconds...")
        time.sleep(5)

audio_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])
expected_text = st.text_input("Expected readback:", "QNH one zero one three, cleared for takeoff runway two four.")

if audio_file and st.button("Run ICAO Assessment"):
    with st.spinner("Transcribing and evaluating..."):
        audio_bytes = io.BytesIO(audio_file.read())
        audio_bytes.name = audio_file.name

        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bytes
        )
        transcription_text = transcript.text.strip()

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

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": rating_prompt}]
        )

        result = response.choices[0].message.content

    st.subheader("🗒️ Transcription")
    st.write(transcription_text)

    st.subheader("📊 ICAO Rating Result")
    st.json(result)



