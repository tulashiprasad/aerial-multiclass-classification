"""
Logging configuration for the application.
Provides structured logging with file rotation.
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger

from src.core.config import settings


def setup_logging() -> logging.Logger:
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("aerial_multiclass_classifier")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        return logger
    
    json_formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d',
        timestamp=True
    )
    
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    log_file_path = log_dir / "app.log"
    file_handler = RotatingFileHandler(
        log_file_path,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(json_formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


logger = setup_logging()
