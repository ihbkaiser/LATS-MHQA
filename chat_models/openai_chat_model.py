import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class OpenAIChatModel(ChatOpenAI):
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("MISSING OPENAI_API_KEY in environment variables. Please set it to use OpenAIChatModel.")
        super().__init__(model_name=model_name, temperature=temperature)