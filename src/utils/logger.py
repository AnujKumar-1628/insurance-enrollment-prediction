import logging
from pathlib import Path

from src.utils.path import LOGS_DIR


def get_logger(
    name: str,
    log_file: str = "app.log",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create and configure a project logger.

    Parameters
    ----------
    name : str
        Name of the logger, usually __name__ of the calling module.

    log_file : str, default="app.log"
        Name of the log file.

    level : int, default=logging.INFO
        Logging level.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    # Make sure the logs directory exists.
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers when get_logger()
    # is called multiple times.
    if logger.handlers:
        return logger

    # Log format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    # File handler
    file_path = Path(LOGS_DIR) / log_file
    file_handler = logging.FileHandler(
        file_path,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Attach handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
