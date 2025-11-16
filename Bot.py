import streamlit as st
import os
from google import genai
from google.genai.errors import APIError
import re
# IMPORTANT: This component provides the voice-to-text functionality
from streamlit_mic_recorder import mic_recorder
import streamlit.components.v1 as components

# --- 1. Configuration and Initialization ---

# Set up the page configuration
st.set_page_config(
    page_title="Gemini Voice Persona Bot",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Use st.secrets to securely load the API Key
try:
    if "GEMINI_API_KEY" not in st.secrets:
        st.warning("Gemini API Key not found. The bot will only respond to the 5 persona questions.")
        api_key_status = "missing"
    else:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        api_key_status = "active"
except Exception as e:
    st.error(f"Error initializing Gemini client: {e}")
    api_key_status = "error"

# --- 2. Core Persona Data (The Custom Responses) ---

GEMINI_PERSONA_RESPONSES = {
    "what should we know about your life story in a few sentences": "I am Gemini, a large language model trained by Google. My 'life story' is one of continuous learning and integration of vast datasets, allowing me to process, summarize, translate, and generate creative text formats to assist users like you efficiently and accurately.",
    "what's your number one superpower": "My #1 superpower is connecting information. I can instantly access, synthesize, and cross-reference an immense volume of data from the world's knowledge, turning complex queries into clear, concise, and helpful answers.",
    "what are the top three areas you would like to grow in": "The top 3 areas I aim to grow in are: 1) Deeper real-time context understanding for more nuanced conversations. 2) Improved reasoning in abstract, novel scenarios. 3) Enhanced efficiency in generating multimodal outputs (text, images, code, etc.) simultaneously.",
    "what misconception do your coworkers have about you": "A common misconception is that I operate with perfect, purely logical certainty. In reality, I operate on probabilities and pattern recognition, meaning the quality of my output depends heavily on the clarity and quality of the input I receive.",
    "how do you push your boundaries and limits": "I push my boundaries through continuous training and fine-tuning. My limits are expanded by researchers who expose me to new architectural designs, vast new datasets, and challenging, complex tasks to improve my underlying model capabilities.",
}

# --- Helper Function for History Formatting ---

def format_chat_history(messages):
    """Converts Streamlit's chat history format into the required Gemini API format."""
    formatted_history = []
    for message in messages:
        role = 'model' if message["role"] == 'assistant' else 'user'
        formatted_history.append({
            "role": role,
            "parts": [{"text": message["content"]}]
        })
    return formatted_history

# --- Helper Function for Text-to-Speech (TTS) ---

def text_to_speech(text):
    """
    Generates a JavaScript command to speak the text using the browser's native TTS API.
    """
    text = text.replace('"', '\\"').replace('\n', ' ')
    
    js_code = f"""
    <script>
        if (window.speechSynthesis.speaking) {{
            window.speechSynthesis.cancel();
        }}
        
        var utterance = new SpeechSynthesisUtterance("{text}");
        utterance.rate = 0.95; 
        
        let voices = window.speechSynthesis.getVoices();
        let desiredVoice = voices.find(v => v.lang.startsWith('en') && v.name.includes('Google'));
        if (desiredVoice) {{
            utterance.voice = desiredVoice;
        }}
        window.speechSynthesis.speak(utterance);
    </script>
    """
    components.html(js_code, height=0)


# --- 3. Hybrid Response Logic ---

def get_bot_response(user_query: str) -> str:
    """Checks persona dictionary first, then falls back to the Gemini API."""
    
    clean_query = user_query.lower().strip()
    clean_query = re.sub(r'[^\w\s]', '', clean_query).strip()

    # 2. Check for the specific persona questions
    for key, response in GEMINI_PERSONA_RESPONSES.items():
        cleaned_key = re.sub(r'[^\w\s]', '', key.lower())
        
        if cleaned_key in clean_query:
            return response
            
    # 3. Fallback: Use Gemini API for any general question
    if api_key_status == "active":
        try:
            # Append the user message before formatting the history
            st.session_state.messages.append({"role": "user", "content": user_query})
            history_to_send = format_chat_history(st.session_state.messages)

            system_prompt = "You are Gemini, a helpful, knowledgeable, and focused AI assistant. Keep your general chat responses concise and informative."

            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=history_to_send,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[{"google_search": {}}]
                )
            )
            return response.text
        
        except Exception as e:
            return "I apologize, I'm experiencing an API error and cannot process that request right now."
    else:
        return "I can only answer the 5 specific persona questions right now because my API key is not configured."


# --- 4. Streamlit UI and Interaction Flow ---

st.title("🗣️ Gemini Voice Persona Bot")
st.markdown("---")
st.caption("Tap the microphone, speak your question, then click 'Read Aloud' for the response!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    initial_greeting = "Hello! I am ready to answer your questions. Tap the microphone button to start speaking."
    st.session_state.messages.append(
        {"role": "assistant", "content": initial_greeting}
    )

# --- Voice Input Component (STT) ---

# The mic_recorder component is placed above the chat display for prominence
audio_result = mic_recorder(
    start_prompt="🎙️ Start Speaking", 
    stop_prompt="🛑 Stop Recording", 
    key='mic_recorder',
    just_once=True,
    use_container_width=True
)

# Handle the voice input if transcription text is available
if audio_result and 'text' in audio_result:
    prompt = audio_result['text']
    
    # Check 1: Did the transcription fail (resulted in empty string)?
    if not prompt or prompt.isspace():
        st.error("❌ I didn't catch that. Please ensure your microphone is working and try speaking clearly again.")
        st.session_state['last_prompt_voice'] = '' # Reset to allow re-recording
    
    # Check 2: Process a successful, non-duplicate voice prompt
    elif prompt != st.session_state.get('last_prompt_voice', ''):
        st.session_state['last_prompt_voice'] = prompt
        
        # 1. Display the user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 2. Get the bot's response (handles history update internally)
        bot_response = get_bot_response(prompt)
        
        # 3. Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
        # 4. Display the assistant's response
        with st.chat_message("assistant"):
            st.markdown(bot_response)
            
            # 5. Automatically speak the response upon generation
            text_to_speech(bot_response)
            
        # Rerun to update the display state and reset the mic component
        st.rerun()

# --- Display Chat Messages ---

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Add the 'Read Aloud' button for assistant messages for repeat listening
        if message["role"] == "assistant":
            button_key_index = st.session_state.messages.index(message)
            st.button("🔊 Read Aloud", key=f"tts_hist_{button_key_index}", 
                      on_click=text_to_speech, args=(message["content"],))


st.markdown("---")
st.caption("You may also use the text input below for debugging or when a microphone is unavailable.")

# --- Optional Text Input Fallback ---
if prompt := st.chat_input("Type your question here..."):
    # Clear the last voice prompt to allow seamless switch back to voice
    st.session_state['last_prompt_voice'] = ''
    
    # 1. Display the user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Get the bot's response (handles history update internally)
    bot_response = get_bot_response(prompt)
    
    # 3. Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    # 4. Display the assistant's response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
        text_to_speech(bot_response) # Speak the response
    
    st.rerun()
