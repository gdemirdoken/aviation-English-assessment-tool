import streamlit as st
import io
import os
import json
import time
from openai import OpenAI
from openai.error import OpenAIError  # Correct for v2.5

# --- Initialize OpenAI client ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("🛫 ICAO English Proficiency Assessment Tool")
st.write("Upload an audio recording to get ICAO English proficiency ratings.")

# --- File upload ---
audio_file = st.file_uploader("Upload audio file (WAV or MP3)", type=["wav", "mp3"])
expected_text = st.text_input(
    "Expected readback:",
    "Wind 250 at 11 knots, cleared for takeoff runway 24L, Turkish 26G."
)

# --- Persistent transcription cache ---
if "transcript_cache" not in st.session_state:
    st.session_state["transcript_cache"] = {}

transcript_cache = st.session_state["transcript_cache"]

if audio_file and st.button("Run ICAO Assessment"):
    file_name = audio_file.name

    # --- Step 1: Transcription (with caching) ---
    if file_name in transcript_cache:
        transcription_text = transcript_cache[file_name]
        st.info("Using cached transcription.")
    else:
        with st.spinner("Transcribing audio..."):
            audio_bytes = io.BytesIO(audio_file.read())
            audio_bytes.name = file_name

            for attempt in range(5):
                try:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_bytes
                    )
                    transcription_text = transcript.text.strip()
                    transcript_cache[file_name] = transcription_text
                    break
                except OpenAIError as e:
                    if "Rate limit" in str(e):
                        wait_time = 2 ** attempt
                        st.warning(f"Whisper rate limit reached. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        raise e
            else:
                st.error("Max retries reached for transcription.")
                st.stop()

    st.subheader("🗒️ Transcription")
    st.write(transcription_text)

    # --- Step 2: ICAO Scoring ---
    rating_prompt = f"""
    You are an ICAO-qualified English Language Proficiency rater assessing student pilots’ spoken performance.
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

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": rating_prompt}]
            )
            result_text = response.choices[0].message.content
            break
        except OpenAIError as e:
            if "Rate limit" in str(e):
                wait_time = 2 ** attempt
                st.warning(f"GPT rate limit reached. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise e
    else:
        st.error("Max retries reached for ICAO scoring.")
        result_text = "Error: Could not get ICAO rating due to rate limits."

    # --- Step 3: Display ICAO Ratings ---
    st.subheader("📊 ICAO Rating Result")
    try:
        result_json = json.loads(result_text)
        st.json(result_json)
    except json.JSONDecodeError:
        st.text(result_text)
