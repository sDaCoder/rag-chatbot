import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# from rich.console import Console
# from rich.markdown import Markdown
import streamlit as st

# console = Console()

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY
)

st.set_page_config(page_title="Kissan AI", page_icon="🌾")
st.title("🌾 Kissan AI")
st.caption("Ask me anything! Type 'exit' to quit.")

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
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

query = st.chat_input("Ask me anything about agriculture and farming...")
if query:
    with st.chat_message("human"):
        st.write(query)
    st.session_state.chat_history.append({"role": "human", "content": query})
    st.session_state.messages.append(HumanMessage(content=query))

    response = model.invoke(st.session_state.messages)
    st.session_state.messages.append(AIMessage(content=response.content))
    st.session_state.chat_history.append({"role": "assistant", "content": response.content})

    with st.chat_message("assistant"):
        st.write(response.content)

# while True:
#     query = input("You: ")
#     if query.lower() in ['exit', 'quit']:
#         break
#     messages.append(HumanMessage(content=query))
#     print()

#     response = model.invoke(messages)
#     messages.append(AIMessage(content=response.content))
#     console.print(Markdown(response.content))