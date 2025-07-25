import streamlit as st
import tempfile
import os
from openai import OpenAI

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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    st.info("Transcribing audio locally... This may take a moment.")

    result = model.transcribe(temp_audio_path)
    transcription = result["text"]

    st.write("### Your Transcription:")
    st.write(transcription)

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
    messages=[{"role": "user", "content": "Hello!"}]
)
        feedback = response.choices[0].message.content
        st.write("### AI Feedback:")
        st.write(feedback)
    except Exception as e:
        st.error(f"Error during AI feedback generation: {e}")
