"""
Logger configuration for the LLM Document Retrieval System
"""

import logging
import structlog
from typing import Any

def setup_logging():
    """Configure structured logging"""
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=None,
        level=logging.INFO,
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    return structlog.get_logger()

def get_logger(name: str = __name__):
    """Get a configured logger instance"""
    return structlog.get_logger(name)

# Initialize logger
logger = setup_logging()
