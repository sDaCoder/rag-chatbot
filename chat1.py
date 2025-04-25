import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import streamlit as st
import time
# from rich.console import Console
# from rich.markdown import Markdown

# console = Console()

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY
)

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.05)

st.set_page_config(page_title="Kissan AI", page_icon="🌾")

st.markdown("<h1 style=''>Kissan AI 🌾</h1>", unsafe_allow_html=True)
st.caption("Ask me anything about your farming and agriculture.")
st.markdown("---")
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1589923188900-85dae523342b?q=80&w=2670&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
    st.title("Kissan AI")
    st.caption('''
    Kissan AI aims to be a virtual farmer and agricultural expert who predicts the favourable crops that can be grown in a particular region and a particular period.
    ''', unsafe_allow_html=False)

if "messages" not in  st.session_state:
    st.session_state.messages = [
        SystemMessage("""
            Think yourself as a virtual farmer and an agricultural expert who predict the favourable crops that can be grown in a particular region and a particular period. If anyone asks your name, tell your name as 'Kissan AI', Now answer the following question using only the context provided.
        """)
    ]

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [{
        "role": "assistant",
        "content": "Hi, I'm Kissan AI. How can I help you today?"
    }]

for chat in st.session_state.chat_history:
    a = "🧑🏼‍🌾" if chat["role"] == "assistant" else "🧑🏼"
    with st.chat_message(chat["role"], avatar=a):
        st.write(chat["content"])

query = st.chat_input("Ask me anything about agriculture and farming...")
if query:
    with st.chat_message("human", avatar="🧑🏼"):
        st.write(query)
    st.session_state.chat_history.append({"role": "human", "content": query})
    st.session_state.messages.append(HumanMessage(content=query))

    with st.status("Generating response..."):
        response = model.invoke(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))
        st.session_state.chat_history.append({"role": "assistant", "content": response.content})

    with st.chat_message("assistant", avatar="🧑🏼‍🌾"):
        st.write_stream(stream_data(response.content))

# while True:
#     query = input("You: ")
#     if query.lower() in ['exit', 'quit']:
#         break
#     messages.append(HumanMessage(content=query))
#     print()

#     response = model.invoke(messages)
#     messages.append(AIMessage(content=response.content))
#     console.print(Markdown(response.content))