import openai
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
print(os.environ.get("GEMINI_API_KEY"))

model = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")
messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]
model.invoke(messages)