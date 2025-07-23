import whisper
import streamlit as st
import openai
from openai import OpenAI # Moved to top
import tempfile
import os

# Load Whisper model
# Consider caching this with st.cache_resource if it's large and you want faster reruns
# @st.cache_resource
model = whisper.load_model("base")  # Options: tiny, base, small, medium, large

# Initialize the OpenAI client (will automatically use OPENAI_API_KEY from env/Streamlit Secrets)
# This line replaces the old openai.api_key setting.
client = OpenAI()

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

# Audio upload
audio_file = st.file_uploader("Upload your voice (MP3/WAV)", type=["mp3", "wav"])

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio: # Added suffix for clarity
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    st.info("Processing your speech...")
    # Transcription using Whisper
    try:
        result = model.transcribe(temp_audio_path)
        transcription = result['text']
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        transcription = "Transcription failed." # Fallback
    finally:
        # Ensure temporary file is cleaned up even if transcription fails
        os.unlink(temp_audio_path)

    st.write("### Your Transcription:")
    st.write(transcription)

    # Only proceed if transcription was successful and not empty
    if transcription and transcription != "Transcription failed.":
        # Generate feedback from GPT
        gpt_prompt = f"""
        You are an ICAO Aviation English evaluator.
        The user said: "{transcription}".
        Evaluate based on ICAO descriptors: Pronunciation, Structure, Vocabulary, Fluency.
        Give feedback and a score for each category (0-5).
        Suggest improvements in plain English and ICAO phraseology if needed.
        """

        st.info("Generating AI feedback...")
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": gpt_prompt}] # CORRECTED: Use gpt_prompt
            )

            feedback = response.choices[0].message.content # CORRECTED: Use dot notation
            st.write("### AI Feedback:")
            st.write(feedback)

        except Exception as e:
            st.error(f"Error generating AI feedback: {e}")
            st.warning("Could not get feedback. Please try again or check your OpenAI API key.")
    else:
        st.warning("No valid transcription to generate feedback from.")
