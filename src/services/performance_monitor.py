"""
Performance monitoring service for tracking request metrics and system performance.
"""

import time
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RequestMetrics:
    """Metrics for a single request"""
    request_id: str
    start_time: float
    end_time: Optional[float] = None
    checkpoints: Dict[str, float] = field(default_factory=dict)
    total_time: Optional[float] = None
    success: bool = False
    error_message: Optional[str] = None


class PerformanceMonitor:
    """Monitor and track performance metrics for requests"""
    
    def __init__(self):
        self.active_requests: Dict[str, RequestMetrics] = {}
        self.completed_requests: List[RequestMetrics] = []
        self.checkpoint_times: Dict[str, List[float]] = defaultdict(list)
        self.lock = asyncio.Lock()
    
    def start_request(self, request_id: str) -> None:
        """Start tracking a new request"""
        self.active_requests[request_id] = RequestMetrics(
            request_id=request_id,
            start_time=time.time()
        )
        logger.debug("Started tracking request", request_id=request_id)
    
    def log_checkpoint(self, request_id: str, checkpoint_name: str, duration: float) -> None:
        """Log a checkpoint for a request"""
        if request_id in self.active_requests:
            self.active_requests[request_id].checkpoints[checkpoint_name] = duration
            self.checkpoint_times[checkpoint_name].append(duration)
            logger.debug(
                "Checkpoint logged",
                request_id=request_id,
                checkpoint=checkpoint_name,
                duration=round(duration, 3)
            )
    
    def log_request_completion(
        self,
        request_id: str,
        total_time: float,
        questions_count: int,
        success: bool,
        error_message: Optional[str] = None
    ) -> None:
        """Complete tracking for a request"""
        if request_id in self.active_requests:
            metrics = self.active_requests[request_id]
            metrics.end_time = time.time()
            metrics.total_time = total_time
            metrics.success = success
            metrics.error_message = error_message
            
            # Move to completed requests
            self.completed_requests.append(metrics)
            del self.active_requests[request_id]
            
            # Keep only last 100 completed requests to avoid memory issues
            if len(self.completed_requests) > 100:
                self.completed_requests = self.completed_requests[-100:]
            
            logger.info(
                "Request completed",
                request_id=request_id,
                total_time=round(total_time, 3),
                questions_count=questions_count,
                success=success,
                error_message=error_message
            )
    
    def get_average_times(self) -> Dict[str, float]:
        """Get average times for different checkpoints"""
        averages = {}
        for checkpoint, times in self.checkpoint_times.items():
            if times:
                averages[checkpoint] = sum(times) / len(times)
        return averages
    
    def get_recent_performance_summary(self, last_n: int = 10) -> Dict[str, Any]:
        """Get performance summary for recent requests"""
        recent_requests = self.completed_requests[-last_n:] if self.completed_requests else []
        
        if not recent_requests:
            return {
                "total_requests": 0,
                "average_response_time": 0.0,
                "success_rate": 0.0,
                "active_requests": len(self.active_requests)
            }
        
        total_times = [req.total_time for req in recent_requests if req.total_time]
        successful_requests = [req for req in recent_requests if req.success]
        
        return {
            "total_requests": len(recent_requests),
            "average_response_time": sum(total_times) / len(total_times) if total_times else 0.0,
            "success_rate": len(successful_requests) / len(recent_requests) if recent_requests else 0.0,
            "active_requests": len(self.active_requests),
            "checkpoint_averages": self.get_average_times()
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get current system health status"""
        recent_summary = self.get_recent_performance_summary()
        avg_response_time = recent_summary.get("average_response_time", 0.0)
        success_rate = recent_summary.get("success_rate", 0.0)
        
        # Determine health status
        if avg_response_time <= 25.0 and success_rate >= 0.9:
            status = "healthy"
        elif avg_response_time <= 35.0 and success_rate >= 0.8:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "average_response_time": round(avg_response_time, 2),
            "success_rate": round(success_rate * 100, 1),
            "active_requests": len(self.active_requests),
            "target_response_time": 25.0
        }
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        async with self.lock:
            self.active_requests.clear()
            self.completed_requests.clear()
            self.checkpoint_times.clear()
        logger.info("Performance monitor cleaned up")
