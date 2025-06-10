import os
from typing import List
from dotenv import load_dotenv

from .base_searcher import BaseSearcher
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search.tool import TavilySearchResults

load_dotenv()

class TavilySearcher(BaseSearcher):

    def __init__(self, max_results: int = 5):
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("Missing TAVILY_API_KEY in environment variables. Please set it to use TavilySearcher.")
        self.wrapper = TavilySearchAPIWrapper()
        self.tool = TavilySearchResults(api_wrapper=self.wrapper, max_results=max_results)

    def retrieve(self, query: str) -> List[str]:
        results = self.tool.invoke(query)
        if isinstance(results, str):
            return []
        return [item["content"] for item in results]
