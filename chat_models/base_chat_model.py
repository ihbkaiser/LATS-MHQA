from abc import ABC, abstractmethod
from langchain_core.runnables import RunnableConfig

class BaseChatModel(ABC):
    @abstractmethod
    def bind_tools(self, **kwargs):

        pass

    @abstractmethod
    def __ror__(self, other):
        pass
