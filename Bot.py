# Bot.py — patched for production: recv_queued, TURN support, robust Gemini retry, safe TTS, safe rerun
import streamlit as st
import os
import re
import io
import time
import logging
import sys
import html
import random
import queue
import numpy as np

from google import genai
from google.genai.errors import APIError
from pydub import AudioSegment
import streamlit.components.v1 as components

# streamlit-webrtc imports (may raise during local editing; handled later)
try:
    from streamlit_webrtc import (
        webrtc_streamer,
        WebRtcMode,
        MediaStreamConstraints,
        AudioProcessorBase,
    )
except Exception:
    # If streamlit-webrtc is not available in the environment, we'll handle that gracefully.
    webrtc_streamer = None
    WebRtcMode = None
    MediaStreamConstraints = None
    AudioProcessorBase = object

# ------------------ Logging ------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("gemini_voice_bot")

# ------------------ Config & API init ------------------
st.set_page_config(page_title="Voice Persona Bot", layout="centered")

api_key_status = "error"
gemini_client = None

try:
    if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
        gemini_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        api_key_status = "active"
        logger.info("Gemini client initialized")
    else:
        st.warning("Gemini API Key not found. The bot will only answer persona questions.")
        logger.warning("GEMINI_API_KEY not found in st.secrets")
except Exception as e:
    logger.exception("Error initializing Gemini client: %s", e)
    st.error(f"Error initializing Gemini client: {e}")

# ------------------ Persona responses ------------------
GEMINI_PERSONA_RESPONSES = {
    "what should we know about your life story in a few sentences":
        "I am Gemini, a large language model trained by Google. My 'life story' is one of continuous learning and integration of vast datasets, allowing me to assist efficiently.",
    "what's your number one superpower":
        "My #1 superpower is connecting information — synthesizing facts and context into clear, actionable answers.",
    "what are the top three areas you would like to grow in":
        "Top 3 areas: 1) Deeper real-time context understanding, 2) stronger abstract reasoning, 3) more efficient multimodal outputs.",
    "what misconception do your coworkers have about you":
        "A common misconception is that I'm always certain — in reality I reason probabilistically and rely on good prompts.",
    "how do you push your boundaries and limits":
        "I improve via continuous training, new datasets, and engineering that exposes me to harder tasks and domains.",
}

# ------------------ Safe helpers ------------------
def safe_rerun():
    """Use experimental_rerun if present; otherwise st.stop() as fallback."""
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
        else:
            logger.info("st.experimental_rerun not available; calling st.stop()")
            st.stop()
    except Exception as e:
        logger.exception("safe_rerun unexpected error: %s", e)
        try:
            st.stop()
        except Exception:
            logger.exception("st.stop() also failed; continuing")

def text_to_speech(text):
    """Defensive TTS injection via Web Speech API."""
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
        if (window.speechSynthesis && window.speechSynthesis.speaking) {{
          window.speechSynthesis.cancel();
        }}
        const utter = new SpeechSynthesisUtterance("{safe_text}");
        utter.lang = 'en-US';
        utter.rate = 0.95;
        window.speechSynthesis.speak(utter);
      }} catch (e) {{
        console.error("TTS injection failed:", e);
      }}
    </script>
    """
    try:
        components.html(js_code, height=0)
    except Exception:
        logger.exception("components.html injection for TTS failed")

# ------------------ Audio processor with recv_queued ------------------
class MicAudioProcessor(AudioProcessorBase):
    """
    Uses a thread-safe queue to collect incoming frames via recv_queued.
    Call drain_queued_audio() from the main thread to obtain raw bytes (float32le).
    """

    def __init__(self):
        # queue of numpy arrays (float32)
        self._q = queue.Queue()
        self.is_recording = False

    def recv_queued(self, frame):
        """
        streamlit-webrtc recommended API: queue frames rather than process inline.
        frame.to_ndarray() returns ndarray (n_channels, n_samples) or (n_samples,)
        """
        try:
            arr = frame.to_ndarray()
            # Ensure copy so internal buffers aren't reused unsafely
            self._q.put(arr.copy(), block=False)
        except Exception:
            # on overload, drop gracefully but log
            logger.exception("recv_queued: failed to queue frame")
        return frame

    # Keep recv as a fallback for environments that call it (it will queue too)
    def recv(self, frame):
        try:
            arr = frame.to_ndarray()
            self._q.put(arr.copy(), block=False)
        except Exception:
            logger.exception("recv fallback failed")
        return frame

    def drain_queued_audio(self):
        """
        Drain queued numpy arrays and return a single bytes blob formatted as float32 little-endian.
        This matches the AudioSegment.from_raw configuration used below (f32le).
        """
        chunks = []
        while True:
            try:
                arr = self._q.get_nowait()
            except queue.Empty:
                break
            # If stereo/multi-channel, use first channel (or average if desired)
            if arr.ndim == 2:
                arr = arr[0]
            arr_f32 = arr.astype("float32")
            chunks.append(arr_f32.tobytes())
        return b"".join(chunks)

# ------------------ Transcription ------------------
def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Convert raw float32le audio bytes (48kHz mono) into mp3 and call Gemini STT.
    This function assumes audio_bytes were produced by drain_queued_audio().
    """
    try:
        audio_segment = AudioSegment.from_raw(
            io.BytesIO(audio_bytes),
            sample_width=4,  # 4 bytes for f32le
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

# ------------------ Bot response (persona + Gemini with retries) ------------------
def format_chat_history(messages):
    formatted_history = []
    for message in messages:
        role = 'model' if message["role"] == 'assistant' else 'user'
        formatted_history.append({"role": role, "parts": [{"text": message["content"]}]})
    return formatted_history

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
        history_to_send = format_chat_history(st.session_state.messages)
        MAX_RETRIES = 3
        BASE_BACKOFF = 1.0

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=history_to_send,
                    config=genai.types.GenerateContentConfig(
                        tools=[{"google_search": {}}]
                    )
                )
                try:
                    text = response.text
                except Exception:
                    logger.exception("Unexpected general chat response structure: %s", repr(response)[:1000])
                    text = None

                if text and text.strip():
                    return text
                else:
                    logger.warning("Gemini returned empty text on attempt %d", attempt)
                    raise Exception("Empty response")

            except Exception as e:
                logger.exception("Gemini call failed on attempt %d: %s", attempt, e)
                if attempt == MAX_RETRIES:
                    return ("Sorry — the model is currently unavailable. "
                            "Please try again in a moment, or use the text input below.")
                backoff = BASE_BACKOFF * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                logger.info("Retrying Gemini call in %.2f seconds (attempt %d/%d)", backoff, attempt, MAX_RETRIES)
                time.sleep(backoff)
        return "Sorry — couldn't get a response right now."
    else:
        return "I can only answer the 5 specific persona questions right now because my Gemini API key is not configured."

# ------------------ UI and WebRTC startup ------------------
st.title("🗣️ Gemini Voice Persona Bot (WebRTC)")
st.markdown("---")
st.caption("Press 'Start' to turn on the mic, speak your question, and then press 'Stop'. If connection fails, use the text input below.")

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Hello! I am ready to answer your questions. Press Start to enable your microphone."})
if 'last_prompt_voice' not in st.session_state:
    st.session_state['last_prompt_voice'] = ''

# Build rtc_configuration with optional TURN
def build_rtc_config():
    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    # Optional TURN
    turn_url = st.secrets.get("TURN_URL") if "TURN_URL" in st.secrets else None
    turn_user = st.secrets.get("TURN_USERNAME") if "TURN_USERNAME" in st.secrets else None
    turn_pass = st.secrets.get("TURN_PASSWORD") if "TURN_PASSWORD" in st.secrets else None
    if turn_url and turn_user and turn_pass:
        ice_servers.append({
            "urls": [turn_url],
            "username": turn_user,
            "credential": turn_pass
        })
        logger.info("TURN server added to rtc_configuration")
    else:
        logger.info("No TURN credentials found in st.secrets (continuing with STUN only)")
    return {"iceServers": ice_servers}

def start_webrtc_safe(**kwargs):
    if webrtc_streamer is None:
        logger.warning("streamlit-webrtc not available")
        st.warning("Audio input unavailable in this environment.")
        return None
    try:
        return webrtc_streamer(**kwargs)
    except Exception as e:
        logger.exception("webrtc_streamer failed to start: %s", e)
        st.warning("Audio input temporarily unavailable. Please reload the page.")
        return None

ctx = start_webrtc_safe(
    key="mic-stt-stream",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=MicAudioProcessor,
    media_stream_constraints=MediaStreamConstraints(video=False, audio=True),
    async_processing=True,
    rtc_configuration=build_rtc_config(),
)

processor = None
if ctx:
    try:
        processor = ctx.audio_processor
    except Exception:
        processor = None

# Controls
st.markdown("---")
col1, col2 = st.columns([1, 1])

is_recording = bool(processor and getattr(processor, 'is_recording', False))
playing = bool(ctx and getattr(ctx, 'state', None) and getattr(ctx.state, 'playing', False))
start_disabled = not playing or is_recording
stop_disabled = not playing or not is_recording

with col1:
    start_button = st.button("🔴 Start Recording", disabled=start_disabled)
with col2:
    stop_button = st.button("⏹️ Stop Recording", disabled=stop_disabled)

# Handle start/stop
if ctx and playing and processor:
    if start_button:
        # start capturing queued frames
        processor.is_recording = True
        st.info("🎙️ Recording started! Please speak now.")
        safe_rerun()  # optional: triggers refresh so UI updates recording state

    if stop_button:
        processor.is_recording = False
        st.info("Processing audio... Please wait.")

        # Drain queued audio bytes instead of using processor.audio_chunks
        raw_bytes = processor.drain_queued_audio()
        if raw_bytes:
            # Transcription
            prompt = transcribe_audio(raw_bytes)

            # Guards
            if not prompt or prompt.isspace() or "error" in prompt.lower():
                with st.chat_message("assistant"):
                    st.error(prompt if prompt else "Transcription returned empty text. Please try again.")
            elif prompt != st.session_state.get('last_prompt_voice', ''):
                st.session_state['last_prompt_voice'] = prompt
                st.session_state.messages.append({"role": "user", "content": prompt})

                # Get bot response
                try:
                    bot_response = get_bot_response(prompt)
                except Exception as e:
                    logger.exception("Unhandled error in get_bot_response: %s", e)
                    bot_response = "Sorry — couldn't generate a response right now."

                if not bot_response:
                    bot_response = "Sorry — couldn't generate a response right now."

                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                try:
                    text_to_speech(bot_response)
                except Exception:
                    logger.exception("text_to_speech failed")
        else:
            st.warning("No audio was recorded (queue empty). Try again or use the text input below.")

        safe_rerun()

# Display chat messages
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

    safe_rerun()
