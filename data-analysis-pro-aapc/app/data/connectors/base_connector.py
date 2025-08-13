from __future__ import annotations

import abc
import contextlib
from typing import Any, Dict, Generator, Iterable, Optional, Sequence


class DatabaseConnectionError(RuntimeError):
    pass


class DatabaseQueryError(RuntimeError):
    pass


class BaseConnector(abc.ABC):
    @abc.abstractmethod
    def connect(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def execute_non_query(self, query: Any, params: Optional[Dict[str, Any]] = None) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def execute_many(self, query: Any, param_list: Iterable[Dict[str, Any]]) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_all(self, query: Any, params: Optional[Dict[str, Any]] = None) -> Sequence[Dict[str, Any]]:
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_one(self, query: Any, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @contextlib.contextmanager
    @abc.abstractmethod
    def transaction(self) -> Generator[None, None, None]:
        raise NotImplementedError