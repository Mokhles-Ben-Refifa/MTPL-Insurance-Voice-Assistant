import streamlit as st
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sidebar import display_sidebar
from src.chat_interface import display_chat_interface
st.title("MTPL Insurance Voice Assistant")
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None
display_sidebar()
display_chat_interface()