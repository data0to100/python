import time
import functools
from typing import Callable, TypeVar, Any

F = TypeVar("F", bound=Callable[..., Any])


def timeit(func: F) -> F:  # type: ignore[override]
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - start) * 1000
        print(f"{func.__name__} took {elapsed:.2f} ms")
        return result

    return wrapper  # type: ignore[return-value]