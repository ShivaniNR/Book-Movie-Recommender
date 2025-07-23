# app.py
import streamlit as st
from main import Recommender

st.set_page_config(page_title="Book & Movie Recommender", page_icon="🎬📚", layout="centered")

st.title("🎬📚 Chat with the Recommender Bot")

# Initialize session state for chat history and recommender
if "recommender" not in st.session_state:
    st.session_state.recommender = Recommender()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_prompt = st.chat_input("Ask me for a movie or book recommendation!")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.recommender.get_chat_response(user_prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})