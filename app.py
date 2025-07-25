import streamlit as st
import whisper
from openai import OpenAI
import tempfile
import os

# Load Whisper model
model = whisper.load_model("base")  # Options: tiny, base, small, medium, large

# Set up OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Streamlit UI
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
    with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    st.info("Processing your speech...")

    # Transcribe speech using Whisper
    result = model.transcribe(temp_audio_path)
    transcription = result['text']

    st.write("### Your Transcription:")
    st.write(transcription)

    # Prepare GPT prompt
    gpt_prompt = f"""
    You are an ICAO Aviation English evaluator.
    The user said: "{transcription}".
    Evaluate based on ICAO descriptors: Pronunciation, Structure, Vocabulary, Fluency.
    Give feedback and a score for each category (0-5).
    Suggest improvements in plain English and ICAO phraseology if needed.
    """

    # Get feedback from GPT using the new SDK
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an ICAO Aviation English evaluator."},
            {"role": "user", "content": gpt_prompt}
        ]
    )

    feedback = response.choices[0].message.content

    st.write("### AI Feedback:")
    st.write(feedback)
