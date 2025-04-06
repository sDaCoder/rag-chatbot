from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.messages import HumanMessage
from langchain.memory import ChatMessageHistory
from typing import Dict
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.3
)


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain: Runnable = prompt | llm

store: Dict[str, ChatMessageHistory] = {}

# --- Define how to get the memory for a session ---
def get_memory(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_memory,
    input_messages_key="input",
    history_messages_key="history"
)

def chat():
    session_id = "default-user"  # could be any unique ID per user/session
    print("Ask me anything! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # Wrap input as a HumanMessage
        user_message = HumanMessage(content=user_input)

        # Pass it to the model
        response = chain_with_memory.invoke(
            {"input": user_message.content},  # Extract content for model
            config={"configurable": {"session_id": session_id}}
        )
        print("Bot:", response.content)

if __name__ == "__main__":
    chat()