import os
from dotenv import load_dotenv
from chat_models.base_chat_model import BaseChatModel

from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

class GeminiChatModel(ChatGoogleGenerativeAI):

    def __init__(self, model_name: str = "gemini-2.0-flash", temperature: float = 0.0):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("MISSING GOOGLE_API_KEY in environment variables. Please set it to use GeminiChatModel.")
        super().__init__(model=model_name, temperature=temperature)
