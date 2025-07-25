import streamlit as st
import tempfile
import whisper
import torch

# Try importing transformers (for grammar feedback)
try:
    from transformers import pipeline
    transformers_available = True
except ImportError:
    transformers_available = False

# Title
st.title("✈️ Aviation English Speaking Tutor (Offline)")

# Load Whisper model locally
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")  # Options: tiny, base, small, medium, large

whisper_model = load_whisper_model()

# If transformers available, load grammar correction model
if transformers_available:
    @st.cache_resource
    def load_grammar_model():
        return pipeline("text2text-generation", model="pszemraj/flan-t5-large-grammar-synthesis")
    grammar_model = load_grammar_model()

# File uploader
audio_file = st.file_uploader("🎤 Upload your spoken Aviation English response (WAV or MP3)", type=["wav", "mp3"])

if audio_file is not None:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_file.read())
        temp_audio_path = temp_audio.name

    st.info("Transcribing audio locally... This may take a moment.")

    # Transcribe with Whisper
    result = whisper_model.transcribe(temp_audio_path)
    transcription = result["text"]

    st.write("### Your Transcription:")
    st.success(transcription)

    # Grammar Feedback
    if transformers_available:
        st.write("### Grammar Feedback:")
        with st.spinner("Analyzing grammar..."):
            correction = grammar_model(f"Correct the grammar of this sentence: {transcription}", max_length=256)[0]['generated_text']
            st.info(correction)
    else:
        st.warning("⚠️ Grammar feedback unavailable (install 'transformers' to enable this feature).")

    # Pronunciation feedback (basic heuristic)
    st.write("### Pronunciation Feedback:")
    avg_conf = sum([w['confidence'] for w in result['segments'] if 'confidence' in w]) / len(result['segments'])
    if avg_conf > 0.85:
        st.success("✅ Good pronunciation!")
    elif avg_conf > 0.7:
        st.warning("👍 Fair pronunciation, but could improve clarity on some words.")
    else:
        st.error("⚠️ Pronunciation needs improvement (many words unclear).")
