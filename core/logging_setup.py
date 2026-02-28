"""Centralized logging configuration with rotation."""
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


def teardown_logging(logger: Optional[logging.Logger] = None) -> None:
    """Flush, detach, and close all handlers from the provided logger."""
    target = logger or logging.getLogger()
    for handler in list(target.handlers):
        target.removeHandler(handler)
        try:
            handler.flush()
            handler.close()
        except Exception:
            # Best-effort cleanup; logging should never crash the app.
            pass


def setup_logging(
    log_level: str = 'INFO',
    logs_dir: Optional[Path] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up centralized logging with rotation.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logs_dir: Directory for log files. If None, uses 'logs/' in current directory.
        console_output: Whether to output logs to console
    
    Returns:
        Configured root logger
    """
    # Resolve log directory
    if logs_dir is None:
        logs_dir = Path.cwd() / 'logs'
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Close previous handlers so file descriptors are released on Windows.
    teardown_logging(logger)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with rotation (P0.5 - Windows compatible)
    log_file = logs_dir / 'bot.log'
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding='utf-8',
        delay=True  # Delay file opening for Windows compatibility (P0.5)
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Log startup
    logger.info(f"Logging initialized at {log_level} level")
    logger.info(f"Log file: {log_file}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
