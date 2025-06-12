# import streamlit as st 
# from rag_chatbot import vector_store,rag_chain

# st.title("MG ZS Warning Messages Chatbot")

# user_input = st.text_input("Ask about a warning message:",value="what should I do if I see 'Engine coolant temperature high'?")
# if user_input:
#     response = rag_chain.invoke(user_input)
#     st.write("Answer:",response)


import streamlit as st 
from rag_chatbot import vector_store,rag_chain
import pyttsx3  # for text to speech 

# intialize session state for  chat history 
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# custom Css for styling the chatbot 

st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .stTextInput > div > div > input { border-radius: 10px; padding: 10px; }
    .stButton > button { background-color: #4CAF50; color: white; border-radius: 10px; }
    .chat-message { padding: 10px; margin: 5px; border-radius: 10px; }
    .user-message { background-color: #d1e7dd; text-align: right; }
    .bot-message { background-color: #e9ecef; text-align: left; }
    </style>
""",unsafe_allow_html=True) 

## app title and header 
st.title("🚗 MG ZS Warning Messages Chatbot")
st.subheader("Ask about car warning messages and get instant answeres!")

# sidebar for settings 
with st.sidebar:
    st.header("Settings")
    voice_output = st.checkbox("Enable Voice Output", value=True)
    clear_history = st.button("Clear Chat History")
    if clear_history:
        st.session_state.chat_history = []
        st.rerun() # Rerun the app to clear the chat history

# chat input 

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask about a warning message:", placeholder="E.g., What should I do if I see 'Engine Coolant Temperature High'?")
    submit_button = st.form_submit_button("Send")

# proces query 
if submit_button and user_input:
    response = rag_chain.invoke(user_input)
    st.session_state.chat_history.append({"user": user_input, "bot": response})
    if voice_output:
        engine = pyttsx3.init()
        engine.say(response)
        engine.runAndWait()

# Display chat history
st.write("### Chat History")
for chat in st.session_state.chat_history:
    st.markdown(f'<div class="chat-message user-message">{chat["user"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="chat-message bot-message">{chat["bot"]}</div>', unsafe_allow_html=True)