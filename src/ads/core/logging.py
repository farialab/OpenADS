import logging
import os
import sys
from pathlib import Path
from typing import Optional

def setup_logger(name: str = "ads", 
                log_file: Optional[str] = None, 
                level: int = logging.INFO, 
                format_str: Optional[str] = None) -> logging.Logger:
    """
    Configure and return a logger with proper formatting
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
        format_str: Optional log format string
    
    Returns:
        Configured logger
    """
    if format_str is None:
        format_str = "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(format_str)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Default logger
logger = setup_logger()