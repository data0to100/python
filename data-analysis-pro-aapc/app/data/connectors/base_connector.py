from abc import ABC, abstractmethod
from typing import Any


class BaseConnector(ABC):
    @abstractmethod
    def fetch(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError