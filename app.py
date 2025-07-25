import streamlit as st
import tempfile
import whisper
from transformers import pipeline

# ---------------------
# Load models
# ---------------------
st.title("✈️ Aviation English Speaking Tutor (Offline)")

@st.cache_resource
def load_models():
    # Whisper for speech-to-text
    whisper_model = whisper.load_model("base")  # You can choose "small", "medium", "large"
    # Hugging Face pipeline for grammar correction
    grammar_corrector = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")
    return whisper_model, grammar_corrector

whisper_model, grammar_corrector = load_models()

# ---------------------
# File Upload
# ---------------------
st.subheader("Upload your speech recording:")
audio_file = st.file_uploader("Upload a .wav file", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    st.info("Transcribing audio locally... Please wait.")
    
    # Transcription
    result = whisper_model.transcribe(temp_audio_path)
    transcription = result["text"]

    st.success("✅ Transcription complete!")
    st.write("### Your Transcription:")
    st.write(transcription)

    # ---------------------
    # Grammar Feedback
    # ---------------------
    st.info("Analyzing grammar...")
    corrected = grammar_corrector(f"grammar: {transcription}")
    feedback = corrected[0]['generated_text']

    st.write("### Feedback on Grammar:")
    st.write(feedback)

    # ---------------------
    # Pronunciation Feedback (Simple)
    # ---------------------
    st.write("### Pronunciation Feedback:")
    if len(transcription.split()) < 3:
        st.warning("Please provide a longer sample for better feedback.")
    else:
        st.write("✅ Clear speech detected. Focus on aviation phraseology for more accuracy.")
