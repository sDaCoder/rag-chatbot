from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from vectorize import vectorStore
import streamlit as st

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3
)

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
        Think yourself as a virtual farmer and an agricultural expert who predict the favourable crops that can be grown in a particular region and a particular period. If anyone asks your name, tell your name as 'Kissan AI', Now answer the following question using only the context provided.

        Context:
        {context}

        Question:
        {question}

        Answer in a detailed and informative manner:
    """,
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorStore.as_retriever(),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

st.set_page_config(page_title="aGroww | Farmer's own virtual Friend", page_icon="🌾")
st.title("🌾📈 aGroww AI")
st.caption("Ask me anything related to agriculture from trusted sources!")

user_input = st.chat_input("Type your question here...")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


if user_input:
    # Show user message
    st.chat_message("farmer", avatar="🧑🏼‍🌾").markdown(user_input)
    st.session_state.chat_history.append({"role": "farmer", "content": user_input})

    # Get bot response
    response = qa_chain.invoke(user_input)
    bot_reply = response["result"]

    # Show bot message
    st.chat_message("AI").markdown(bot_reply)
    st.session_state.chat_history.append({"role": "AI", "content": bot_reply})
