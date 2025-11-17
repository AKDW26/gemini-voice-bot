import streamlit as st
import os
import re
import io
import time
import logging
import sys
import html
from google import genai
from google.genai.errors import APIError
from pydub import AudioSegment
import streamlit.components.v1 as components
from streamlit_webrtc import webrtc_streamer, WebRtcMode, MediaStreamConstraints, AudioProcessorBase

# ------------------------- Logging -------------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gemini_voice_bot")

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
    logger.exception("Error initializing Gemini client: %s", e)
    st.error(f"Error initializing Gemini client: {e}")

# --- Audio Processor Class ---

class MicAudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_chunks = []
        self.is_recording = False

    # NOTE: recv may drop frames on busy hosts; consider implementing recv_queued()
    # to avoid dropped frames. For now we keep recv to match original behaviour.
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


# Defensive TTS: replaced original implementation to avoid crashes when text is None
def text_to_speech(text):
    if text is None:
        logger.warning("text_to_speech called with None; using fallback message.")
        text = "Sorry — I couldn't generate a response right now. Please try again."

    try:
        text = str(text)
    except Exception as e:
        logger.exception("Failed to convert TTS text to str: %s", e)
        text = "Sorry — an internal error occurred preparing audio output."

    safe_text = html.escape(text).replace("\n", " ")

    js_code = f"""
    <script>
        try {{
            if (window.speechSynthesis && window.speechSynthesis.speaking) {{ window.speechSynthesis.cancel(); }}
            var utterance = new SpeechSynthesisUtterance("{safe_text}");
            utterance.rate = 0.95; // Slightly slower for better clarity

            let voices = window.speechSynthesis.getVoices();
            let desiredVoice = voices.find(v => v.lang.startsWith('en') && v.name.includes('Google'));
            if (desiredVoice) {{ utterance.voice = desiredVoice; }}

            window.speechSynthesis.speak(utterance);
        }} catch (e) {{ console.error('TTS injection error', e); }}
    </script>
    """

    try:
        components.html(js_code, height=0)
    except Exception:
        logger.exception("components.html injection for TTS failed")


def get_audio_processor(ctx):
    try:
        return ctx.audio_processor
    except Exception:
        return None


def transcribe_audio(audio_bytes: bytes) -> str:
    try:
        audio_segment = AudioSegment.from_raw(
            io.BytesIO(audio_bytes),
            sample_width=4, # 4 bytes for f32le
            frame_rate=48000,
            channels=1,
            format="f32le"
        )
        mp3_buffer = io.BytesIO()
        audio_segment.export(mp3_buffer, format="mp3")
        mp3_buffer.name = "audio.mp3"
        mp3_buffer.seek(0)
    except Exception as e:
        logger.exception("Audio conversion error: %s", e)
        return f"Audio conversion error: {e}"

    if not gemini_client:
        logger.warning("transcribe_audio called but gemini_client not configured")
        return "Gemini STT is not configured. Please add GEMINI_API_KEY to st.secrets."

    mp3_buffer.seek(0)
    audio_data = mp3_buffer.read()
    size_bytes = len(audio_data)

    STT_MODEL = "gemini-2.5-flash"

    try:
        from google.genai.types import Part

        if size_bytes < 5_000_000:
            part = Part.from_bytes(data=audio_data, mime_type="audio/mpeg")
            response = gemini_client.models.generate_content(
                model=STT_MODEL,
                contents=[part, "Transcribe this audio clip to plain text."]
            )
            # defensive: ensure response has text
            try:
                return response.text
            except Exception:
                logger.exception("Unexpected STT response structure: %s", repr(response)[:1000])
                return "Gemini STT returned an unexpected response format."

        else:
            mp3_buffer.seek(0)
            try:
                uploaded = gemini_client.files.upload(file=mp3_buffer)
                response = gemini_client.models.generate_content(
                    model=STT_MODEL,
                    contents=["Transcribe this audio clip.", uploaded]
                )
                gemini_client.files.delete(name=uploaded.name)
                try:
                    return response.text
                except Exception:
                    logger.exception("Unexpected STT response structure for uploaded file: %s", repr(response)[:1000])
                    return "Gemini STT returned an unexpected response format."
            except Exception as e_upload:
                logger.exception("Gemini upload or STT error: %s", e_upload)
                return f"Gemini file-upload/model error: {e_upload}. File size: {size_bytes} bytes"

    except APIError as gen_e:
        logger.exception("Gemini STT APIError: %s", gen_e)
        return f"Gemini STT API error: {gen_e}"
    except Exception as general_e:
        logger.exception("Unexpected Gemini STT error: %s", general_e)
        return f"Unexpected Gemini STT error: {general_e}"


# get_bot_response largely kept, but ensure defensive handling by callers

def get_bot_response(user_query: str) -> str:
    try:
        clean_query = user_query.lower().strip()
    except Exception:
        clean_query = ""

    clean_query = re.sub(r'[^\w\s]', '', clean_query).strip()

    for key, response in GEMINI_PERSONA_RESPONSES.items():
        cleaned_key = re.sub(r'[^\w\s]', '', key.lower())
        if cleaned_key in clean_query:
            return response

    if api_key_status == "active":
        try:
            history_to_send = format_chat_history(st.session_state.messages)
            system_prompt = "You are Gemini, a helpful, knowledgeable, and focused AI assistant. Keep your general chat responses concise and informative."

            response = gemini_client.models.generate_content(
                model='gemini-2.5-flash',
                contents=history_to_send,
                config=genai.types.GenerateContentConfig(
                    tools=[{"google_search": {}}]
                )
            )
            try:
                return response.text
            except Exception:
                logger.exception("Unexpected general chat response structure: %s", repr(response)[:1000])
                return "Sorry — the model returned an unexpected response format."

        except Exception as e:
            logger.exception("Error calling Gemini for general chat: %s", e)
            # avoid leaving session_state mutated incorrectly
            try:
                if st.session_state.messages:
                    st.session_state.messages.pop()
            except Exception:
                pass
            return f"I apologize, I'm experiencing an API error ({type(e).__name__}) and cannot process that request right now."
    else:
        return "I can only answer the 5 specific persona questions right now because my Gemini API key is not configured."


# ------------------------- Streamlit UI and Interaction Flow -------------------------

st.title("🗣️ Gemini Voice Persona Bot (WebRTC)")
st.markdown("---")
st.caption("Press 'Start' to turn on the mic, speak your question, and then press 'Stop'. Check network and firewall if connection issues persist.")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {"role": "assistant", "content": "Hello! I am ready to answer your questions. Press Start to enable your microphone."}
    )
if 'last_prompt_voice' not in st.session_state:
    st.session_state['last_prompt_voice'] = ''

# Safe wrapper for webrtc_streamer to avoid unhandled exceptions
def start_webrtc_streamer_safe(**kwargs):
    try:
        return webrtc_streamer(**kwargs)
    except Exception as e:
        logger.exception("webrtc_streamer failed to start: %s", e)
        st.warning("Audio input temporarily unavailable. Please reload the page.")
        return None

ctx = start_webrtc_streamer_safe(
    key="mic-stt-stream",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=MicAudioProcessor,
    media_stream_constraints=MediaStreamConstraints(video=False, audio=True),
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

processor = get_audio_processor(ctx) if ctx else None

st.markdown("---")
col1, col2 = st.columns([1, 1])

is_recording = bool(processor and getattr(processor, 'is_recording', False))
start_disabled = not (ctx and getattr(ctx, 'state', None) and getattr(ctx.state, 'playing', False)) or is_recording
stop_disabled = not (ctx and getattr(ctx, 'state', None) and getattr(ctx.state, 'playing', False)) or not is_recording

with col1:
    start_button = st.button("🔴 Start Recording", disabled=start_disabled)

with col2:
    stop_button = st.button("⏹️ Stop Recording", disabled=stop_disabled)

if ctx and getattr(ctx, 'state', None) and getattr(ctx.state, 'playing', False) and processor:
    if start_button:
        processor.audio_chunks = []
        processor.is_recording = True
        st.info("🎙️ Recording started! Please speak now.")
        st.experimental_rerun()

    if stop_button:
        processor.is_recording = False
        st.info("Processing audio... Please wait.")

        if processor.audio_chunks:
            audio_bytes = b"".join(processor.audio_chunks)
            processor.audio_chunks = []

            prompt = transcribe_audio(audio_bytes)

            if not prompt or prompt.isspace() or "error" in prompt.lower():
                with st.chat_message("assistant"):
                    st.error(prompt if prompt else "Transcription returned empty text. Please try again.")

            elif prompt != st.session_state.get('last_prompt_voice', ''):
                st.session_state['last_prompt_voice'] = prompt
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Safely get bot response
                try:
                    bot_response = get_bot_response(prompt)
                except Exception as e:
                    logger.exception("Unhandled error in get_bot_response: %s", e)
                    bot_response = "Sorry — couldn't generate a response right now."

                if not bot_response:
                    logger.warning("get_bot_response returned empty; using fallback")
                    bot_response = "Sorry — couldn't generate a response right now."

                st.session_state.messages.append({"role": "assistant", "content": bot_response})

                try:
                    text_to_speech(bot_response)
                except Exception:
                    logger.exception("text_to_speech failed")

        else:
            st.warning("No audio was recorded.")

        st.experimental_rerun()

# --- Display Chat Messages ---
st.markdown("---")
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 
        if message["role"] == "assistant":
            st.button("🔊 Read Aloud", key=f"tts_hist_{idx}", on_click=text_to_speech, args=(message["content"],))

st.markdown("---")
st.caption("You may also use the text input below for debugging.")

if prompt := st.chat_input("Type your question here..."):
    st.session_state['last_prompt_voice'] = ''
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        bot_response = get_bot_response(prompt)
    except Exception as e:
        logger.exception("Unhandled error in get_bot_response (text input): %s", e)
        bot_response = "Sorry — couldn't generate a response right now."

    if not bot_response:
        bot_response = "Sorry — couldn't generate a response right now."

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    try:
        text_to_speech(bot_response)
    except Exception:
        logger.exception("text_to_speech failed (text input)")

    st.experimental_rerun()
