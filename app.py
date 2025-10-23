import streamlit as st
import io
import os
import time
from openai import OpenAI

# --- Initialize OpenAI client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("🎧 Whisper Transcription Test")
st.write("Upload an audio file to test transcription only (no GPT scoring).")

audio_file = st.file_uploader("Upload audio file (WAV or MP3)", type=["wav", "mp3"])

if audio_file and st.button("Transcribe"):
    audio_bytes = io.BytesIO(audio_file.read())
    audio_bytes.name = audio_file.name

    with st.spinner("Transcribing with Whisper..."):
        transcription_text = None
        for attempt in range(5):
            try:
                # Use Whisper API
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_bytes
                )
                transcription_text = transcript.text.strip()
                break  # success
            except Exception as e:
                if "rate limit" in str(e).lower():
                    wait_time = 2 ** attempt
                    st.warning(f"Rate limit hit. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    st.error(f"Error: {e}")
                    break

    if transcription_text:
        st.success("✅ Transcription successful!")
        st.text_area("Transcribed Text:", transcription_text, height=200)
