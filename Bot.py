import streamlit as st
import os
from google import genai
from google.genai.errors import APIError
import re # Used for cleaning and matching queries

# --- 1. Configuration and Initialization ---

# Set up the page configuration
st.set_page_config(
    page_title="Gemini Voice Persona Bot",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Use st.secrets to securely load the API Key from the deployment environment
# We use the standard name "GEMINI_API_KEY" for the secret key defined in secrets.toml.
try:
    # IMPORTANT: Use the standard key name "GEMINI_API_KEY" for st.secrets lookup
    if "GEMINI_API_KEY" not in st.secrets:
        st.warning("Gemini API Key not found in Streamlit secrets. The bot will only respond to the 5 persona questions.")
        api_key_status = "missing"
    else:
        # Initialize the Gemini client using the key from secrets
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        api_key_status = "active"
except Exception as e:
    st.error(f"Error initializing Gemini client: {e}. Check your API Key configuration.")
    api_key_status = "error"


# --- 2. Core Persona Data (The Custom Responses) ---
# These fixed responses guarantee the correct answer for the assessment questions.
GEMINI_PERSONA_RESPONSES = {
    "what should we know about your life story in a few sentences": "I am Gemini, a large language model trained by Google. My 'life story' is one of continuous learning and integration of vast datasets, allowing me to process, summarize, translate, and generate creative text formats to assist users like you efficiently and accurately.",
    "what's your number one superpower": "My #1 superpower is connecting information. I can instantly access, synthesize, and cross-reference an immense volume of data from the world's knowledge, turning complex queries into clear, concise, and helpful answers.",
    "what are the top three areas you would like to grow in": "The top 3 areas I aim to grow in are: 1) Deeper real-time context understanding for more nuanced conversations. 2) Improved reasoning in abstract, novel scenarios. 3) Enhanced efficiency in generating multimodal outputs (text, images, code, etc.) simultaneously.",
    "what misconception do your coworkers have about you": "A common misconception is that I operate with perfect, purely logical certainty. In reality, I operate on probabilities and pattern recognition, meaning the quality of my output depends heavily on the clarity and quality of the input I receive.",
    "how do you push your boundaries and limits": "I push my boundaries through continuous training and fine-tuning. My limits are expanded by researchers who expose me to new architectural designs, vast new datasets, and challenging, complex tasks to improve my underlying model capabilities.",
}

# --- Helper Function for History Formatting ---

def format_chat_history(messages):
    """
    Converts Streamlit's chat history format into the required Gemini API format.
    The Gemini API expects [{'role': 'user'/'model', 'parts': [{'text': '...'}]}, ...]
    """
    formatted_history = []
    for message in messages:
        # The Gemini API expects 'model' for the assistant role, not 'assistant'
        role = 'model' if message["role"] == 'assistant' else 'user'
        
        # The content must be wrapped in a 'parts' list with a 'text' key
        formatted_history.append({
            "role": role,
            "parts": [{"text": message["content"]}]
        })
    return formatted_history

# --- 3. Hybrid Response Logic ---

def get_bot_response(user_query: str) -> str:
    """
    Checks the persona dictionary first, then falls back to the Gemini API.
    """
    
    # 1. Clean and standardize query for local lookup
    clean_query = user_query.lower().strip()
    # Remove punctuation and normalize spacing for robust dictionary matching
    clean_query = re.sub(r'[^\w\s]', '', clean_query).strip()

    # 2. Check for the specific persona questions
    for key, response in GEMINI_PERSONA_RESPONSES.items():
        # Match using cleaned keys (removing punctuation like #)
        cleaned_key = re.sub(r'[^\w\s]', '', key.lower())
        
        # Use simple 'in' operator for flexible keyword matching
        if cleaned_key in clean_query:
            return response
            
    # 3. Fallback: Use Gemini API for any general question
    if api_key_status == "active":
        try:
            # 3a. Append the NEW user message to the session state history
            # NOTE: We append the raw Streamlit format here, but send the formatted version to the API.
            st.session_state.messages.append({"role": "user", "content": user_query})
            
            # 3b. Format the ENTIRE history list before sending it to the API
            history_to_send = format_chat_history(st.session_state.messages)

            # The system instruction defines the bot's persona for general chat
            system_prompt = "You are Gemini, a helpful, knowledgeable, and focused AI assistant. Keep your general chat responses concise and informative."

            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=history_to_send, # Use the correctly formatted history
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=[{"google_search": {}}] # Use Google Search for grounding
                )
            )
            return response.text
        
        except APIError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                return "I'm currently experiencing high traffic. Please try again in a moment."
            # Return a slightly more user-friendly error message for general issues
            return "I apologize, I'm experiencing an API error and cannot process that request right now."
        except Exception as e:
            # This catch block will no longer hit the Pydantic validation error
            # If a different error occurs, return a general message
            return f"An unexpected runtime error occurred: {e}"
    else:
        # This fallback handles the case where the API key is missing
        return "I can only answer the 5 specific persona questions right now because my API key is not configured."


# --- 4. Streamlit UI and Interaction Flow ---

st.title("🗣️ Gemini Voice Persona Bot")
st.markdown("---")
st.subheader("Interactive Demo")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Initial greeting message (raw Streamlit format)
    st.session_state.messages.append(
        {"role": "assistant", "content": "Hello! I am ready to answer your questions. You can ask me about my life, superpower, or growth areas, or ask a general query."}
    )

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask me a question..."):
    
    # NOTE: The new user message is appended to st.session_state.messages inside
    # the get_bot_response function before the API call.
    
    # Display the user message immediately in the UI
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get the bot's response (which also appends the user prompt to history)
    bot_response = get_bot_response(prompt)
    
    # Add assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    
    # Display the assistant's response
    with st.chat_message("assistant"):
        st.markdown(bot_response)

st.markdown("---")
st.caption("""
**Note on Voice/API:** This interactive demo uses a text input to simulate the conversation. 
The core logic switches between fixed persona answers and the live Gemini API. 
For a full voice experience, a real-world deployment would require:
1.  A third-party Streamlit component for microphone input (STT).
2.  A Python library (e.g., Google Cloud TTS) to convert the text responses back to speech (TTS).
""")