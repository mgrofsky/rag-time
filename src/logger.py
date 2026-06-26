"""
Centralized logging configuration for the time-aware RAG pipeline.

Provides structured logging with appropriate levels, formatting, and
output destinations for better debugging and monitoring.
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

def setup_logger(
    name: str = "rag_time",
    level: str = "INFO", 
    log_file: Optional[Path] = None,
    include_timestamps: bool = True
) -> logging.Logger:
    """
    Set up a logger with consistent formatting across the pipeline.
    
    Args:
        name: Logger name (typically module name)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging output
        include_timestamps: Whether to include timestamps in log messages
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Don't add handlers if they already exist
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    if include_timestamps:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class PipelineError(Exception):
    """Base exception class for pipeline-specific errors."""
    pass

class DataIngestionError(PipelineError):
    """Raised when data ingestion fails."""
    pass

class EmbeddingError(PipelineError):
    """Raised when embedding generation fails."""
    pass

class ClusteringError(PipelineError):
    """Raised when clustering fails."""
    pass

def log_pipeline_step(step_name: str, input_count: int, output_count: int, logger: logging.Logger) -> None:
    """
    Log a standardized pipeline step completion message.
    
    Args:
        step_name: Name of the pipeline step
        input_count: Number of input items processed
        output_count: Number of output items generated
        logger: Logger instance to use
    """
    logger.info(f"✓ {step_name}: {input_count} → {output_count} items")

def log_timing(func):
    """Decorator to log function execution time."""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger = logging.getLogger(func.__module__)
        logger.info(f"Starting {func.__name__}...")
        
        try:
            result = func(*args, **kwargs)
            duration = datetime.now() - start_time
            logger.info(f"✓ {func.__name__} completed in {duration.total_seconds():.2f}s")
            return result
        except Exception as e:
            duration = datetime.now() - start_time
            logger.error(f"✗ {func.__name__} failed after {duration.total_seconds():.2f}s: {str(e)}")
            raise
    
    return wrapper