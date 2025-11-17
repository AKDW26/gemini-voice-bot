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
        role = 'model' if message["role"] == 'assistant' else 'user'
        formatted_history.append({"role": role, "parts": [{"text": message["content"]}]})
    return formatted_history


def text_to_speech(text):
    if text is None:
        text = "Sorry, there was an error in generating the response."

    text = str(text).replace('"', '\\"').replace('\n', ' ')
    js_code = f"""
    <script>
        try {{
            if (window.speechSynthesis && window.speechSynthesis.speaking) {{ window.speechSynthesis.cancel(); }}
            var utterance = new SpeechSynthesisUtterance("{text}");
            utterance.rate = 0.95;
            let voices = window.speechSynthesis.getVoices();
            let desiredVoice = voices.find(v => v.lang && v.lang.startsWith('en') && v.name && v.name.includes('Google'));
            if (desiredVoice) {{ utterance.voice = desiredVoice; }}
            window.speechSynthesis.speak(utterance);
        }} catch(e) {{
            console.error('TTS error', e);
        }}
    </script>
    """
    try:
        components.html(js_code, height=0)
    except Exception:
        pass


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Convert raw captured audio frames -> valid MP3 bytes and call Gemini generate_content
    Returns response.text or an error string.
    """
    # Quick guard
    if not audio_bytes:
        return "No audio bytes provided."

    # Convert raw frames into MP3 using pydub.
    try:
        # Try to interpret incoming bytes as raw float32 frames if that's what you recorded.
        # If that fails, fallback to trying to read it as an already valid audio container.
        mp3_buffer = io.BytesIO()

        try:
            audio_segment = AudioSegment.from_raw(
                io.BytesIO(audio_bytes),
                sample_width=4,   # f32le -> 4 bytes/sample; change if your frames are int16
                frame_rate=48000,
                channels=1,
                format="f32le"
            )
        except Exception:
            # Fallback: maybe audio_bytes are already an MP3/WAV container
            try:
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            except Exception as e_inner:
                return f"Audio conversion error (both raw & container decode failed): {e_inner}"

        # Export to MP3 bytes (valid audio container)
        audio_segment.export(mp3_buffer, format="mp3")
        mp3_buffer.seek(0)
        audio_data = mp3_buffer.read()

    except Exception as e:
        return f"Audio conversion error: {e}"

    # Ensure Gemini client present
    if not gemini_client:
        return "Gemini STT is not configured. Please add GEMINI_API_KEY."

    # Build Part and call generate_content (text prompt first, then binary Part)
    try:
        from google.genai import types as genai_types

        part = genai_types.Part.from_bytes(data=audio_data, mime_type="audio/mpeg")

        # Put the textual instruction first, then audio part
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",   # change to the model that supports audio in your account if needed
            contents=[
                "Transcribe this audio",
                part
            ],
        )

        return response.text

    except Exception as e:
        # If the SDK raises an APIError, show more detail
        try:
            from google.genai.errors import APIError
            if isinstance(e, APIError):
                return f"STT API error: {e.message if getattr(e, 'message', None) else str(e)}"
        except Exception:
            pass
        return f"STT error: {type(e).__name__} - {e}"



def get_bot_response(user_query: str) -> str:
    clean_query = re.sub(r'[^\w\s]', '', user_query.lower().strip())

    for key, response in GEMINI_PERSONA_RESPONSES.items():
        if re.sub(r'[^\w\s]', '', key.lower()) in clean_query:
            return response
            
    # --- REPLACED: use single-text prompt to avoid SDK 'oneof data' errors ---
    if api_key_status == "active":
        try:
            system_prompt = "You are Gemini, a helpful, knowledgeable, and focused AI assistant. Keep your responses concise."
            N_HISTORY = 8
            recent = st.session_state.messages[-N_HISTORY:] if "messages" in st.session_state else []
            convo_lines = []
            for m in recent:
                speaker = "Assistant" if m["role"] == "assistant" else "User"
                content = str(m["content"]).replace("\n", " ")
                convo_lines.append(f"{speaker}: {content}")
            convo_text = "\n".join(convo_lines)
            full_prompt = system_prompt + "\n\nConversation:\n" + convo_text + "\n\nUser: " + user_query

            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[full_prompt],
                config=genai.types.GenerateContentConfig(
                    tools=[{"google_search": {}}]
                )
            )
            return response.text

        except Exception as e:
            try:
                st.session_state.messages.pop()
            except Exception:
                pass
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
    st.session_state['last_prompt_voice'] = ''

# --- WebRTC Component ---
ctx = webrtc_streamer(
    key="mic-stt-stream",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=MicAudioProcessor,
    media_stream_constraints=MediaStreamConstraints(video=False, audio=True),
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Safety Check: Get processor only if available
processor = ctx.audio_processor if ctx and ctx.audio_processor else None

# --- Custom Start/Stop Buttons and Logic ---
st.markdown("---")

col1, col2 = st.columns([1, 1])

is_recording = processor and processor.is_recording
start_disabled = not (ctx and ctx.state.playing) or is_recording
stop_disabled = not (ctx and ctx.state.playing) or not is_recording

with col1:
    start_button = st.button("🔴 Start Recording", disabled=start_disabled)

with col2:
    stop_button = st.button("⏹️ Stop Recording", disabled=stop_disabled)

if ctx and ctx.state.playing and processor:
    if start_button:
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
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant":
            st.button("🔊 Read Aloud",
                      key=f"tts_hist_{idx}",
                      on_click=text_to_speech,
                      args=(message["content"],))


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


