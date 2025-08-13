import logging
import os
from typing import Optional

_DEFAULT_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def configure_logging(level: Optional[str] = None, fmt: Optional[str] = None) -> None:
    log_level_name = (level or os.getenv("APP_LOG_LEVEL", "INFO")).upper()
    log_format = fmt or os.getenv("APP_LOG_FORMAT", _DEFAULT_LOG_FORMAT)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=log_level_name, format=log_format)
    else:
        logging.getLogger().setLevel(log_level_name)
        for handler in logging.getLogger().handlers:
            handler.setLevel(log_level_name)
            if isinstance(handler.formatter, logging.Formatter):
                handler.setFormatter(logging.Formatter(log_format))


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)