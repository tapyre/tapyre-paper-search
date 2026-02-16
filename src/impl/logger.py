import logging
import os
from logging.handlers import RotatingFileHandler

# Internal registry to store already-created logger instances.
# This ensures that each logger name is only configured once
# and prevents duplicate handlers or duplicated log output.
_loggers = {}


def get_logger(name: str = "app"):
    """
    Create or retrieve a configured logger instance.

    Features:
    - Singleton-like behavior per logger name
    - Automatic log directory creation
    - Rotating file logging for long-running processes
    - Console logging for real-time monitoring
    - Configurable log level via environment variable

    Parameters:
    - name: logger identifier (usually __name__ of caller module)

    Returns:
    - logging.Logger instance ready for use

    Design rationale:
    Logging configuration is centralized here to guarantee
    consistent formatting, levels, and output destinations
    across the entire application.
    """

    # Return cached logger if already initialized
    # Prevents re-attaching handlers multiple times.
    if name in _loggers:
        return _loggers[name]

    # Directory where log files will be stored
    # Created dynamically to allow portable deployments.
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # Main log file path
    log_file = os.path.join(log_dir, "pipeline.log")

    # Standardized log message format
    # Includes timestamp, level, logger name, and message
    log_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # Explicit timestamp format for readability and sorting
    date_format = "%Y-%m-%d %H:%M:%S"

    # File handler with rotation support
    # Automatically rotates logs when size limit is reached.
    # This prevents disk exhaustion during long-running jobs.
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5_000_000,  # 5 MB per file
        backupCount=5        # keep last 5 rotated logs
    )
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Console handler for stdout logging
    # Useful for development, containers, and live monitoring.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Retrieve (or create) logger instance
    logger = logging.getLogger(name)

    # Determine log level from environment variable
    # Allows runtime configuration without code changes.
    level = os.getenv("LOG_LEVEL", "INFO").upper()

    # Apply resolved log level
    # Falls back to INFO if environment value is invalid.
    logger.setLevel(getattr(logging, level, logging.INFO))

    # Attach handlers to logger
    # Order does not matter; both receive all log records.
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Disable propagation to root logger
    # Prevents duplicate logs if root logger also has handlers.
    logger.propagate = False

    # Store configured logger in registry for reuse
    _loggers[name] = logger

    return logger
