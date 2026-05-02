"""Minimal logging wrapper - no structlog dependency."""

import logging
from typing import Optional


def get_logger(name: str = "judge_memory") -> logging.Logger:
    """Get a logger instance with minimal setup.
    
    Uses NullHandler by default to avoid "No handler found" warnings.
    Applications can configure logging separately if desired.
    
    Args:
        name: Logger name (default: judge_memory)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handler if none exists
    if not logger.handlers:
        handler = logging.NullHandler()
        logger.addHandler(handler)
    
    return logger


def setup_logging(level: int = logging.INFO) -> None:
    """Optional setup for visible logging output.
    
    Only call this if you want to see log output.
    Safe to call multiple times.
    
    Args:
        level: Logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
