from __future__ import annotations

import logging
from typing import Optional

from rich.logging import RichHandler


def configure_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure structured console logging, and optional file logging."""
    handlers = [RichHandler(rich_tracebacks=True)]

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
