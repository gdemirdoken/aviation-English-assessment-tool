import streamlit as st
import os
from openai import OpenAI

# Initialize OpenAI client with API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("✈️ Aviation English Speaking Tutor")
st.write("Practice Aviation English speaking skills with AI feedback based on ICAO standards.")

# Scenario selection
scenario = st.selectbox("Choose a scenario:", ["Normal Operations", "Emergency"])
if scenario == "Normal Operations":
    prompt_text = "You are a pilot reporting your altitude and heading to ATC."
else:
    prompt_text = "Declare an engine failure and request priority landing."
st.write(f"### Scenario: {prompt_text}")

# Audio upload
audio_file = st.file_uploader("Upload your voice (MP3/WAV)", type=["mp3", "wav"])
if audio_file is not None:
    with open("temp_audio_file", "wb") as f:
        f.write(audio_file.read())
    st.info("Processing your speech...")

    # Transcription using OpenAI Whisper API
    try:
        with open("temp_audio_file", "rb") as audio:
            transcription_response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio
            )
        transcription = transcription_response.text
        st.write("### Your Transcription:")
        st.write(transcription)
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        st.stop()

    # Generate feedback from GPT
    gpt_prompt = f"""
    You are an ICAO Aviation English evaluator.
    The user said: "{transcription}".
    Evaluate based on ICAO descriptors: Pronunciation, Structure, Vocabulary, Fluency.
    Give feedback and a score for each category (0-5).
    Suggest improvements in plain English and ICAO phraseology if needed.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert aviation English evaluator."},
                {"role": "user", "content": gpt_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        feedback = response.choices[0].message.content
        st.write("### AI Feedback:")
        st.write(feedback)
    except Exception as e:
        st.error(f"Error during AI feedback generation: {e}")
