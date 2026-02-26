import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    # Ensure parent directory exists (RotatingFileHandler will not create it)
    try:
        parent = os.path.dirname(log_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
    except Exception:
        # Best effort; logger init should not crash app
        pass
    handler = RotatingFileHandler(log_file, maxBytes=100_000_000, backupCount=10, encoding='utf-8')
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

