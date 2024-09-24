import streamlit as st
from faster_whisper import WhisperModel
import os
from io import BytesIO

# Set up page configuration
st.set_page_config(
    page_title="AI Podcast Transcriber",
    page_icon="🎙️",
    layout="centered",
     initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Main background */
    body {
        background-color: #f0f2f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Header styling */
    h1 {
        color: #3366cc;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 1rem;
    }

    /* Sidebar background and headers */
    .sidebar .sidebar-content {
        background-color: #e6e6ff;
        padding: 1rem;
    }
    
    h2, .stSidebar .stButton {
        color: #ff6600;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #ff6600 !important;
        color: white !important;
        border-radius: 10px;
        width: 100%;
        padding: 10px;
        font-size: 1.2rem;
    }

    /* Progress bar color */
    .stProgress {
        background-color: #4caf50 !important;
    }

    /* Text area */
    .stTextArea textarea {
        font-size: 1.2rem;
        color: #333;
        background-color: #e6f2ff !important;
    }

    /* Checkbox styling */
    .stCheckbox {
        font-size: 1.1rem;
    }

    /* Markdown bullets */
    ul {
        font-size: 1.2rem;
        color: #003366;
        list-style-type: '⦿';
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #999;
        font-size: 0.9rem;
        margin-top: 2rem;
    }

    </style>
""", unsafe_allow_html=True)

# App title and header
st.title("🎙️ Talk2Text AI Podcast Transcriber")
st.sidebar.header("Transcription Section")

# Ensure transcript is stored in session_state
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

# Ensure audio file name is stored in session_state
if "audio_file_name" not in st.session_state:
    st.session_state.audio_file_name = ""

# Ensure that the checkbox value for displaying the summary persists
if "display_summary" not in st.session_state:
    st.session_state.display_summary = False

# Upload the audio file
audio_file = st.file_uploader("Upload audio file (WAV or MP3)", type=["wav", "mp3"])

start_transcribe_button = st.sidebar.button("🚀 Start Transcribing")

# Caching the model loading
@st.cache_resource
def load_model():
    model_size = "tiny"
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    return model

# Transcription process
if start_transcribe_button:
    
    if audio_file is not None:
        st.sidebar.markdown("### 🔄 AI Acting on it! Please wait...")

        model = load_model()

        st.sidebar.info("Transcribing audio...")

        audio_bytes = BytesIO(audio_file.read())
        st.session_state.audio_file_name = audio_file.name  # Store audio file name

        # Display a progress bar during processing
        with st.spinner('Transcribing...'):
            # segments, info = model.transcribe(audio_bytes, beam_size=5, language="en", condition_on_previous_text=False)
            segments, info = model.transcribe(audio_bytes, beam_size=5, condition_on_previous_text=False)

        # Show detected language and probability
        st.write(f"### Detected language: {info.language} with probability **{info.language_probability:.2f}**")

        # Store transcript in session state
        transcript = ""
        for segment in segments:
            st.markdown(f"##### {segment.text}")  # Display the transcribed text
            transcript += f"{segment.text}"
        
        st.session_state.transcript = transcript  # Save transcript to session state
        
        st.sidebar.success("🎉 Transcription Complete")
        st.sidebar.audio(audio_file)
    else:
         st.sidebar.error("Please upload an episode.")

# Display the transcript if it exists
if st.session_state.transcript:
    st.text_area("📜 Full Transcript", st.session_state.transcript, height=300)

    # Allow users to download the transcript
    st.download_button(label="💾 Download Transcript",
        data=st.session_state.transcript,
        file_name=f"{st.session_state.audio_file_name}_transcript.txt" if st.session_state.audio_file_name else 'transcript.txt',
        mime='text/plain',
        key="download"
    )

    # Option to display summary as bullet points, store checkbox value in session_state
    st.session_state.display_summary = st.checkbox("📝 Display transcribe in bullet points", value=st.session_state.display_summary)

    if st.session_state.display_summary:
        bullets = st.session_state.transcript.split(". ")
        for bullet in bullets:
            if bullet.strip():
                st.markdown(f"- {bullet.strip()}")

# Footer
st.markdown("""
    <div class="footer">
        Made by [Kore Sampath Kumar] | Powered by AI Models
    </div>
   
""", unsafe_allow_html=True)

st.markdown("[LinkedIn Profile](https://www.linkedin.com/in/kore-sampath-kumar-618b3762/)")

# ---- END OF CODE ----
