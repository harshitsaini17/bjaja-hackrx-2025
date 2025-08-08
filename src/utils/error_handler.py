"""
Error handling utilities and decorators
"""

import functools
import traceback
import time
from typing import Callable, Any
import logging

from fastapi import HTTPException

logger = logging.getLogger(__name__)


def handle_exceptions(func: Callable) -> Callable:
    """
    Decorator to handle exceptions in API endpoints
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            return result
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
            
        except TimeoutError as e:
            processing_time = time.time() - start_time
            logger.error(
                f"Request timeout in {func.__name__}",
                extra={
                    "error": str(e),
                    "processing_time": processing_time,
                    "traceback": traceback.format_exc()
                }
            )
            raise HTTPException(
                status_code=504,
                detail=f"Request timeout after {processing_time:.1f} seconds"
            )
            
        except ValueError as e:
            processing_time = time.time() - start_time
            logger.error(
                f"Validation error in {func.__name__}",
                extra={
                    "error": str(e),
                    "processing_time": processing_time
                }
            )
            raise HTTPException(
                status_code=422,
                detail=f"Validation error: {str(e)}"
            )
            
        except ConnectionError as e:
            processing_time = time.time() - start_time
            logger.error(
                f"Connection error in {func.__name__}",
                extra={
                    "error": str(e),
                    "processing_time": processing_time
                }
            )
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable due to connection error"
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"Unexpected error in {func.__name__}",
                extra={
                    "error": str(e),
                    "processing_time": processing_time,
                    "traceback": traceback.format_exc()
                }
            )
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )
    
    return wrapper


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, exponential_backoff: bool = True):
    """
    Decorator to retry failed operations
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # Last attempt failed
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} retries",
                            extra={
                                "error": str(e),
                                "attempt": attempt + 1
                            }
                        )
                        raise
                    
                    # Calculate delay for next attempt
                    if exponential_backoff:
                        sleep_time = delay * (2 ** attempt)
                    else:
                        sleep_time = delay
                    
                    logger.warning(
                        f"Function {func.__name__} failed, retrying in {sleep_time}s",
                        extra={
                            "error": str(e),
                            "attempt": attempt + 1,
                            "max_retries": max_retries
                        }
                    )
                    
                    import asyncio
                    await asyncio.sleep(sleep_time)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


def timeout_handler(timeout_seconds: float):
    """
    Decorator to add timeout to async functions
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            import asyncio
            
            try:
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_seconds
                )
                return result
                
            except asyncio.TimeoutError:
                logger.error(
                    f"Function {func.__name__} timed out after {timeout_seconds}s"
                )
                raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
        
        return wrapper
    return decorator


class ErrorTracker:
    """Track and analyze errors across the system"""
    
    def __init__(self):
        self.error_counts = {}
        self.recent_errors = []
        self.max_recent_errors = 100
    
    def log_error(self, error_type: str, error_message: str, context: dict = None):
        """Log an error occurrence"""
        error_data = {
            "type": error_type,
            "message": error_message,
            "context": context or {},
            "timestamp": time.time()
        }
        
        # Update counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Add to recent errors
        self.recent_errors.append(error_data)
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)
        
        logger.error(
            f"Error tracked: {error_type}",
            extra={
                "error_message": error_message,
                "context": context,
                "total_count": self.error_counts[error_type]
            }
        )
    
    def get_error_summary(self) -> dict:
        """Get summary of tracked errors"""
        return {
            "error_counts": self.error_counts.copy(),
            "total_errors": sum(self.error_counts.values()),
            "recent_errors_count": len(self.recent_errors),
            "most_common_errors": sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def get_recent_errors(self, minutes: int = 5) -> list:
        """Get recent errors within specified time window"""
        cutoff_time = time.time() - (minutes * 60)
        return [
            error for error in self.recent_errors
            if error["timestamp"] >= cutoff_time
        ]


# Global error tracker instance
error_tracker = ErrorTracker()


def log_error(error_type: str, error_message: str, context: dict = None):
    """Convenience function to log errors"""
    error_tracker.log_error(error_type, error_message, context)


def validate_request_size(max_size_mb: float = 50.0):
    """
    Decorator to validate request size
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # This is a placeholder - actual implementation would check request size
            # In FastAPI, you can check request.headers.get('content-length')
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def rate_limit(requests_per_minute: int = 60):
    """
    Simple rate limiting decorator (placeholder implementation)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Placeholder for rate limiting logic
            # In production, you'd implement this with Redis or in-memory cache
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator
