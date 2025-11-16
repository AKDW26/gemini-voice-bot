import streamlit as st
import os
import re
import io
import time
from google import genai
from google.genai.errors import APIError
from pydub import AudioSegment
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, WebRtcMode, MediaStreamConstraints, AudioProcessorBase

# --- 1. Configuration and Initialization ---

st.set_page_config(
    page_title="Voice Persona Bot",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- API Key Initialization ---
api_key_status = "error"
openai_client = None
gemini_client = None

try:
    if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
        gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        api_key_status = "active"
    else:
        st.warning("Gemini API Key not found. The bot will only answer persona questions.")
except Exception as e:
    st.error(f"Error initializing Gemini client: {e}")

# --- Audio Processor Class ---

class MicAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_chunks = []
        self.is_recording = False

    def recv(self, frame):
        if self.is_recording:
            # Note: frame.to_ndarray() is typically float32, mono, 48000 Hz in WebRTC context.
            # We append the raw bytes.
            self.audio_chunks.append(frame.to_ndarray().tobytes())
        return frame

# --- Core Persona Data ---
GEMINI_PERSONA_RESPONSES = {
    "what should we know about your life story in a few sentences": "I am Gemini, a large language model trained by Google. My 'life story' is one of continuous learning and integration of vast datasets, allowing me to process, summarize, translate, and generate creative text formats to assist users like you efficiently and accurately.",
    "what's your number one superpower": "My #1 superpower is connecting information. I can instantly access, synthesize, and cross-reference an immense volume of data from the world's knowledge, turning complex queries into clear, concise, and helpful answers.",
    "what are the top three areas you would like to grow in": "The top 3 areas I aim to grow in are: 1) Deeper real-time context understanding for more nuanced conversations. 2) Improved reasoning in abstract, novel scenarios. 3) Enhanced efficiency in generating multimodal outputs (text, images, code, etc.) simultaneously.",
    "what misconception do your coworkers have about you": "A common misconception is that I operate with perfect, purely logical certainty. In reality, I operate on probabilities and pattern recognition, meaning the quality of my output depends heavily on the clarity and quality of the input I receive.",
    "how do you push your boundaries and limits": "I push my boundaries through continuous training and fine-tuning. My limits are expanded by researchers who expose me to new architectural designs, vast new datasets, and challenging, complex tasks to improve my underlying model capabilities.",
}

# --- Helper Functions ---

def format_chat_history(messages):
    formatted_history = []
    for message in messages:
        # Use 'model' for 'assistant' role consistency with Gemini API
        role = 'model' if message["role"] == 'assistant' else 'user'
        formatted_history.append({"role": role, "parts": [{"text": message["content"]}]})
    return formatted_history


def text_to_speech(text):
    # Sanitize text for JavaScript: escape quotes and replace newlines
    text = text.replace('"', '\\"').replace('\n', ' ')
    js_code = f"""
    <script>
        // Cancel any currently speaking utterance
        if (window.speechSynthesis.speaking) {{ window.speechSynthesis.cancel(); }}
        
        var utterance = new SpeechSynthesisUtterance("{text}");
        utterance.rate = 0.95; // Slightly slower for better clarity
        
        // Try to find a nice English voice
        let voices = window.speechSynthesis.getVoices();
        let desiredVoice = voices.find(v => v.lang.startsWith('en') && v.name.includes('Google'));
        if (desiredVoice) {{ utterance.voice = desiredVoice; }}
        
        window.speechSynthesis.speak(utterance);
    </script>
    """
    # Streamlit component must be loaded to execute the JS
    components.html(js_code, height=0)


def get_audio_processor(ctx):
    """Retrieves the audio processor instance from the context."""
    return ctx.audio_processor


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio using Gemini STT.
    The primary fix is changing the model from 'gemini-1.5-flash' to 'gemini-2.5-flash' 
    to resolve the 404 NOT_FOUND error for multimodal tasks.
    """
    
    # --- 1. Audio Conversion (pydub) ---
    try:
        # The input is expected to be raw float32le, 48000 Hz, 1 channel from WebRTC
        audio_segment = AudioSegment.from_raw(
            io.BytesIO(audio_bytes),
            sample_width=4, # 4 bytes for f32le
            frame_rate=48000,
            channels=1,
            format="f32le"
        )
        
        # Convert to MP3 format for Gemini API compatibility and compression
        mp3_buffer = io.BytesIO()
        audio_segment.export(mp3_buffer, format="mp3")
        mp3_buffer.name = "audio.mp3"
        mp3_buffer.seek(0)
    except Exception as e:
        return f"Audio conversion error: {e}"

    if not gemini_client:
        return "Gemini STT is not configured. Please add GEMINI_API_KEY to st.secrets."

    # Read bytes for size decision
    mp3_buffer.seek(0)
    audio_data = mp3_buffer.read()
    size_bytes = len(audio_data)
    
    # --- 2. Gemini STT ---
    # Use the stable model supporting multimodal input
    STT_MODEL = "gemini-2.5-flash" 

    try:
        # Prefer using Part for small files (inline)
        from google.genai.types import Part
        
        if size_bytes < 5_000_000: # ~5 MB threshold for inline vs. file upload
            
            # construct a Part object from bytes and send with a transcription prompt
            part = Part.from_bytes(data=audio_data, mime_type="audio/mpeg")
            response = gemini_client.models.generate_content(
                model=STT_MODEL,
                contents=[part, "Transcribe this audio clip to plain text."]
            )
            
            return response.text # Assumes standard SDK response structure

        else:
            # For larger audio files, upload via Files API then reference the uploaded file
            mp3_buffer.seek(0)
            try:
                uploaded = gemini_client.files.upload(file=mp3_buffer)
                
                response = gemini_client.models.generate_content(
                    model=STT_MODEL,
                    contents=["Transcribe this audio clip.", uploaded]
                )
                
                # Clean up uploaded file immediately (good practice)
                gemini_client.files.delete(name=uploaded.name)
                
                return response.text
                
            except Exception as e_upload:
                # If file upload not available in SDK or fails, return informative error
                return f"Gemini file-upload/model error: {e_upload}. File size: {size_bytes} bytes"

    except APIError as gen_e:
        return f"Gemini STT API error: {gen_e}"
    except Exception as general_e:
        return f"Unexpected Gemini STT error: {general_e}"


def get_bot_response(user_query: str) -> str:
    clean_query = user_query.lower().strip()
    # Simple regex to remove punctuation for robust persona matching
    clean_query = re.sub(r'[^\w\s]', '', clean_query).strip()

    # --- 1. Check for hardcoded persona responses ---
    for key, response in GEMINI_PERSONA_RESPONSES.items():
        cleaned_key = re.sub(r'[^\w\s]', '', key.lower())
        if cleaned_key in clean_query:
            return response
            
    # --- 2. General Chat via Gemini API ---
    if api_key_status == "active":
        try:
            # Add user message to history before generating content
            history_to_send = format_chat_history(st.session_state.messages)
            system_prompt = "You are Gemini, a helpful, knowledgeable, and focused AI assistant. Keep your general chat responses concise and informative."
            
            # NOTE: gemini-2.5-flash is used for general chat, consistent with STT fix
            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=history_to_send,
                config=genai.types.GenerateContentConfig(
                    # Safely specify the tool. Using a list of dicts for tool specification
                    # is compatible across different SDK versions.
                    tools=[{"google_search": {}}]
                )
            )
            return response.text
        
        except Exception as e:
            # If API call fails, remove the last user message to avoid a corrupted history state
            st.session_state.messages.pop()
            return f"I apologize, I'm experiencing an API error ({type(e).__name__}) and cannot process that request right now."
    else:
        return "I can only answer the 5 specific persona questions right now because my Gemini API key is not configured."


# --- 4. Streamlit UI and Interaction Flow ---

st.title("🗣️ Gemini Voice Persona Bot (WebRTC)")
st.markdown("---")
st.caption("Press 'Start' to turn on the mic, speak your question, and then press 'Stop'. Check network and firewall if connection issues persist.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {"role": "assistant", "content": "Hello! I am ready to answer your questions. Press Start to enable your microphone."}
    )
if 'last_prompt_voice' not in st.session_state:
    # Tracks the last *successful* transcribed prompt to prevent re-processing on st.rerun()
    st.session_state['last_prompt_voice'] = ''

# --- WebRTC Component ---
ctx = webrtc_streamer(
    key="mic-stt-stream",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=MicAudioProcessor,
    media_stream_constraints=MediaStreamConstraints(video=False, audio=True),
    async_processing=True,
    # Stability fix: STUN server configuration for better connectivity
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Safety Check: Get processor only if available
processor = get_audio_processor(ctx) if ctx.audio_processor else None

# --- Custom Start/Stop Buttons and Logic ---
st.markdown("---")

col1, col2 = st.columns([1, 1])

# Determine button states with processor check
is_recording = processor and processor.is_recording
start_disabled = not ctx.state.playing or is_recording
stop_disabled = not ctx.state.playing or not is_recording

with col1:
    start_button = st.button("🔴 Start Recording", disabled=start_disabled)

with col2:
    stop_button = st.button("⏹️ Stop Recording", disabled=stop_disabled)

if ctx.state.playing and processor:
    if start_button:
        # Clear previous recording and start new one
        processor.audio_chunks = []
        processor.is_recording = True
        st.info("🎙️ Recording started! Please speak now.")
        st.rerun() 

    if stop_button:
        processor.is_recording = False
        st.info("Processing audio... Please wait.")
        
        if processor.audio_chunks:
            audio_bytes = b"".join(processor.audio_chunks)
            
            # --- Transcription Step ---
            prompt = transcribe_audio(audio_bytes)
            
            # Reset chunks immediately to prepare for the next recording
            processor.audio_chunks = [] 
            
            # Check 1: Ignore empty or failure prompts
            if not prompt or prompt.isspace() or "error" in prompt.lower():
                with st.chat_message("assistant"):
                    st.error(prompt if prompt else "Transcription returned empty text. Please try again.")
            
            # Check 2: Process a successful prompt
            # Check against last_prompt_voice is a guardrail against re-running on browser refresh
            elif prompt != st.session_state.get('last_prompt_voice', ''):
                st.session_state['last_prompt_voice'] = prompt
                
                # Display the user's transcribed prompt
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Get the bot's response
                bot_response = get_bot_response(prompt)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                
                # Speak the response
                text_to_speech(bot_response)
                
        else:
            st.warning("No audio was recorded.")
        
        st.rerun()

# --- Display Chat Messages ---

st.markdown("---")
# Iterate backwards to ensure the latest message is read aloud immediately after generation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant":
            button_key_index = st.session_state.messages.index(message)
            # Use lambda or functools.partial for dynamic arguments in on_click
            # The args= must be a tuple
            st.button("🔊 Read Aloud", key=f"tts_hist_{button_key_index}", 
                      on_click=text_to_speech, args=(message["content"],))


st.markdown("---")
st.caption("You may also use the text input below for debugging.")

# --- Optional Text Input Fallback ---
if prompt := st.chat_input("Type your question here..."):
    
    # Clear voice prompt state when using text input
    st.session_state['last_prompt_voice'] = ''
    
    # --- Text Input Processing ---
    st.session_state.messages.append({"role": "user", "content": prompt})
    bot_response = get_bot_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    text_to_speech(bot_response) 
    
    st.rerun()
