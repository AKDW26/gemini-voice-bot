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
    page_title="Voice Persona Bot",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Use st.secrets to securely load the API Key
try:
    if "GEMINI_API_KEY" not in st.secrets:
        st.warning("Gemini API Key not found. The bot will only respond to the 5 persona questions.")
        api_key_status = "missing"
    else:
        # Check if the API key is actually available in the environment/secrets
        if st.secrets["GEMINI_API_KEY"]:
            client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
            api_key_status = "active"
        else:
            st.warning("Gemini API Key is found in secrets but is empty.")
            api_key_status = "missing"
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
    Note: The component.html call is executed on the frontend.
    """
    # Escape quotes and newlines for JavaScript
    text = text.replace('"', '\\"').replace('\n', ' ')
    
    js_code = f"""
    <script>
        // Stop any currently speaking utterance
        if (window.speechSynthesis.speaking) {{
            window.speechSynthesis.cancel();
        }}
        
        var utterance = new SpeechSynthesisUtterance("{text}");
        utterance.rate = 0.95; // Slightly slower speech rate
        
        // Try to find a high-quality English voice
        let voices = window.speechSynthesis.getVoices();
        let desiredVoice = voices.find(v => v.lang.startsWith('en') && v.name.includes('Google'));
        if (!desiredVoice) {{
             desiredVoice = voices.find(v => v.lang.startsWith('en') && v.name.includes('Samantha'));
        }}
        if (desiredVoice) {{
            utterance.voice = desiredVoice;
        }}
        
        window.speechSynthesis.speak(utterance);
    </script>
    """
    # Use components.html with height=0 to embed the script without taking up screen space
    components.html(js_code, height=0)


# --- 3. Hybrid Response Logic ---

def get_bot_response(user_query: str) -> str:
    """Checks persona dictionary first, then falls back to the Gemini API."""
    
    clean_query = user_query.lower().strip()
    # Remove all punctuation for a cleaner match against the persona keys
    clean_query = re.sub(r'[^\w\s]', '', clean_query).strip()

    # 1. Check for the specific persona questions
    for key, response in GEMINI_PERSONA_RESPONSES.items():
        cleaned_key = re.sub(r'[^\w\s]', '', key.lower())
        
        # Check if the cleaned key is fully contained in the cleaned query
        if cleaned_key in clean_query:
            return response
            
    # 2. Fallback: Use Gemini API for any general question
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
                    # Enable Google Search grounding tool
                    tools=[{"google_search": {}}]
                )
            )
            return response.text
        
        except APIError as api_e:
            return f"An API error occurred: {api_e}. Please check your API key and network connection."
        except Exception as e:
            # Revert the history append if an error occurred during API call
            st.session_state.messages.pop() 
            return "I apologize, I'm experiencing an internal error and cannot process that request right now."
    else:
        # Revert the history append if an error occurred during API call
        st.session_state.messages.append({"role": "user", "content": user_query})
        return "I can only answer the 5 specific persona questions right now because my API key is not configured or is invalid."


# --- 4. Streamlit UI and Interaction Flow ---

st.title("🗣️ Gemini Voice Persona Bot")
st.markdown("---")
st.caption("Tap the microphone, speak your question, then click 'Stop Recording' (or wait for auto-stop).")

# Initialize chat history and state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
    initial_greeting = "Hello! I am ready to answer your questions. Tap the microphone button to start speaking."
    st.session_state.messages.append(
        {"role": "assistant", "content": initial_greeting}
    )

# Initialize a key to store the last voice prompt processed (for de-duplication)
if 'last_prompt_voice' not in st.session_state:
    st.session_state['last_prompt_voice'] = ''

# Initialize key for audio component result
if 'audio_result' not in st.session_state:
    st.session_state.audio_result = None

# --- Voice Input Component (STT) ---

# The mic_recorder component is called here and its output is stored in st.session_state.audio_result
st.session_state.audio_result = mic_recorder(
    start_prompt="🎙️ Start Speaking", 
    stop_prompt="🛑 Stop Recording", 
    key='mic_recorder',
    # Note: just_once=True means it resets after a successful recording
    just_once=True,
    use_container_width=True 
)

# --- Voice Input Debug ---
st.markdown("---")
with st.expander("🎤 Voice Input Debug: What the bot heard"):
    audio_result = st.session_state.audio_result
    
    # Check if a result dict exists
    if audio_result:
        transcribed_text = audio_result.get('text', 'No text key found in result.')
        
        # Specific check for empty transcription (common failure mode)
        if 'text' in audio_result and not audio_result['text']:
            st.warning("Transcription failed (Empty Text). Check browser mic settings and speak clearly.")
            transcribed_text = "Transcription failed (result was empty string)."
    else:
        transcribed_text = 'No audio recorded yet.'

    st.text_area("Transcribed Text", transcribed_text, height=50, disabled=True)
st.markdown("---")
# End Voice Input Debug

# --- Process Voice Input ---

# Check if a new, successful transcription text is available in the session state
audio_result = st.session_state.audio_result
if audio_result and 'text' in audio_result:
    prompt = audio_result['text']
    
    # Check 1: Ignore empty or purely whitespace prompts
    if not prompt or prompt.isspace():
        # Clear the result to prevent processing it again on rerun
        st.session_state.audio_result = None
        pass
    
    # Check 2: Process a successful, non-duplicate voice prompt
    elif prompt != st.session_state.get('last_prompt_voice', ''):
        
        st.session_state['last_prompt_voice'] = prompt # Update de-duplication key
        
        # 1. Get the bot's response (handles history update internally)
        # Note: We skip displaying the user prompt here as it will be done 
        # when we iterate over st.session_state.messages later.
        bot_response = get_bot_response(prompt)
        
        # 2. Add assistant message to history (user message was added in get_bot_response)
        # The history list is now complete: [..., user_prompt, bot_response]
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        
        # 3. Speak the response (needs to happen before rerun)
        text_to_speech(bot_response)
        
        # 4. Clear the audio result to prevent the prompt being reprocessed
        st.session_state.audio_result = None
        
        # 5. Rerun to update the display state with the new chat messages
        st.rerun()

# --- Display Chat Messages ---

# This loop ensures all messages (including the one just added) are displayed
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Add the 'Read Aloud' button for assistant messages for repeat listening
        if message["role"] == "assistant":
            # Use index as part of the key to ensure uniqueness for each button
            button_key_index = st.session_state.messages.index(message)
            st.button("🔊 Read Aloud", key=f"tts_hist_{button_key_index}", 
                      on_click=text_to_speech, args=(message["content"],))


st.markdown("---")
st.caption("You may also use the text input below for debugging or when a microphone is unavailable.")

# --- Optional Text Input Fallback ---
if prompt := st.chat_input("Type your question here..."):
    
    # Clear the last voice prompt state to allow seamless switch back to voice
    st.session_state['last_prompt_voice'] = ''
    
    # 1. Get the bot's response (handles history update internally)
    bot_response = get_bot_response(prompt)
    
    # 2. Add assistant message to history (user message was added in get_bot_response)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    # 3. Speak the response
    text_to_speech(bot_response) 
    
    # 4. Rerun to display the new messages
    st.rerun()
