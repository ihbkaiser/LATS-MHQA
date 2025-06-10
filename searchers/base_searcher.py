from abc import ABC, abstractmethod
from typing import List

class BaseSearcher(ABC):

    @abstractmethod
    def retrieve(self, query: str) -> List[str]:

        pass
