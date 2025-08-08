"""
Performance monitoring utilities
Tracks request processing times and system metrics
"""

import time
import statistics
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and track system performance metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        
        # Request tracking
        self.request_metrics: Dict[str, Dict[str, Any]] = {}
        self.request_history: deque = deque(maxlen=max_history)
        
        # Checkpoint tracking
        self.checkpoints: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # System metrics
        self.system_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "median_response_time": 0.0,
            "p95_response_time": 0.0,
            "p99_response_time": 0.0,
            "requests_under_target": 0,  # Under 25s target
            "requests_over_target": 0,   # Over 25s target
            "start_time": time.time()
        }
        
        # Performance targets from specifications
        self.targets = {
            "validation_time": 0.5,
            "document_processing_time": 5.0,
            "question_processing_time": 15.0,
            "response_compilation_time": 2.5,
            "total_response_time": 25.0
        }
    
    def log_checkpoint(self, request_id: str, checkpoint_name: str, duration: float):
        """Log a performance checkpoint"""
        checkpoint_data = {
            "name": checkpoint_name,
            "duration": duration,
            "timestamp": time.time()
        }
        
        self.checkpoints[request_id].append(checkpoint_data)
        
        # Check against targets
        target_time = self.targets.get(checkpoint_name)
        if target_time and duration > target_time:
            logger.warning(
                f"Checkpoint {checkpoint_name} exceeded target",
                extra={
                    "request_id": request_id,
                    "actual_time": duration,
                    "target_time": target_time,
                    "difference": duration - target_time
                }
            )
    
    def log_request_completion(
        self, 
        request_id: str, 
        total_time: float, 
        questions_count: int,
        success: bool,
        error: Optional[str] = None
    ):
        """Log completion of a request"""
        request_data = {
            "request_id": request_id,
            "total_time": total_time,
            "questions_count": questions_count,
            "success": success,
            "error": error,
            "timestamp": time.time(),
            "checkpoints": self.checkpoints.get(request_id, [])
        }
        
        # Store metrics
        self.request_metrics[request_id] = request_data
        self.request_history.append(request_data)
        
        # Update system metrics
        self._update_system_metrics(request_data)
        
        # Performance analysis
        self._analyze_request_performance(request_data)
    
    def _update_system_metrics(self, request_data: Dict[str, Any]):
        """Update overall system metrics"""
        self.system_metrics["total_requests"] += 1
        
        if request_data["success"]:
            self.system_metrics["successful_requests"] += 1
        else:
            self.system_metrics["failed_requests"] += 1
        
        # Check against target
        total_time = request_data["total_time"]
        if total_time <= self.targets["total_response_time"]:
            self.system_metrics["requests_under_target"] += 1
        else:
            self.system_metrics["requests_over_target"] += 1
        
        # Calculate statistics from recent requests
        recent_times = [r["total_time"] for r in list(self.request_history) if r["success"]]
        
        if recent_times:
            self.system_metrics["average_response_time"] = statistics.mean(recent_times)
            self.system_metrics["median_response_time"] = statistics.median(recent_times)
            
            if len(recent_times) >= 20:  # Need sufficient data for percentiles
                self.system_metrics["p95_response_time"] = self._percentile(recent_times, 95)
                self.system_metrics["p99_response_time"] = self._percentile(recent_times, 99)
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile from data"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _analyze_request_performance(self, request_data: Dict[str, Any]):
        """Analyze request performance against targets"""
        request_id = request_data["request_id"]
        total_time = request_data["total_time"]
        
        # Performance classification
        if total_time <= 20:
            performance_class = "excellent"
        elif total_time <= 25:
            performance_class = "good"
        elif total_time <= 30:
            performance_class = "acceptable"
        else:
            performance_class = "slow"
        
        # Bottleneck analysis
        bottlenecks = []
        checkpoints = {cp["name"]: cp["duration"] for cp in request_data["checkpoints"]}
        
        for checkpoint, target in self.targets.items():
            if checkpoint in checkpoints and checkpoints[checkpoint] > target:
                bottlenecks.append({
                    "checkpoint": checkpoint,
                    "actual": checkpoints[checkpoint],
                    "target": target,
                    "difference": checkpoints[checkpoint] - target
                })
        
        logger.info(
            f"Request performance analysis",
            extra={
                "request_id": request_id,
                "performance_class": performance_class,
                "total_time": total_time,
                "bottlenecks": bottlenecks,
                "questions_count": request_data["questions_count"]
            }
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        uptime = time.time() - self.system_metrics["start_time"]
        
        metrics = self.system_metrics.copy()
        metrics.update({
            "uptime_seconds": uptime,
            "requests_per_minute": self._calculate_requests_per_minute(),
            "success_rate": self._calculate_success_rate(),
            "target_achievement_rate": self._calculate_target_achievement_rate(),
            "current_request_count": len(self.request_history)
        })
        
        return metrics
    
    def get_request_metrics(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific request"""
        return self.request_metrics.get(request_id)
    
    def get_recent_performance(self, minutes: int = 5) -> Dict[str, Any]:
        """Get performance metrics for recent requests"""
        cutoff_time = time.time() - (minutes * 60)
        recent_requests = [
            r for r in self.request_history 
            if r["timestamp"] >= cutoff_time
        ]
        
        if not recent_requests:
            return {"message": f"No requests in the last {minutes} minutes"}
        
        successful_requests = [r for r in recent_requests if r["success"]]
        response_times = [r["total_time"] for r in successful_requests]
        
        metrics = {
            "time_period_minutes": minutes,
            "total_requests": len(recent_requests),
            "successful_requests": len(successful_requests),
            "failed_requests": len(recent_requests) - len(successful_requests),
            "success_rate": len(successful_requests) / len(recent_requests) if recent_requests else 0,
        }
        
        if response_times:
            metrics.update({
                "average_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "min_response_time": min(response_times),
                "max_response_time": max(response_times),
                "requests_under_target": sum(1 for t in response_times if t <= 25.0),
                "target_achievement_rate": sum(1 for t in response_times if t <= 25.0) / len(response_times)
            })
        
        return metrics
    
    def _calculate_requests_per_minute(self) -> float:
        """Calculate requests per minute over last 5 minutes"""
        cutoff_time = time.time() - 300  # 5 minutes
        recent_requests = [
            r for r in self.request_history 
            if r["timestamp"] >= cutoff_time
        ]
        
        if not recent_requests:
            return 0.0
        
        time_span = time.time() - recent_requests[0]["timestamp"]
        return (len(recent_requests) / time_span) * 60 if time_span > 0 else 0.0
    
    def _calculate_success_rate(self) -> float:
        """Calculate overall success rate"""
        total = self.system_metrics["total_requests"]
        successful = self.system_metrics["successful_requests"]
        
        return successful / total if total > 0 else 0.0
    
    def _calculate_target_achievement_rate(self) -> float:
        """Calculate percentage of requests meeting target time"""
        total = self.system_metrics["total_requests"]
        under_target = self.system_metrics["requests_under_target"]
        
        return under_target / total if total > 0 else 0.0
    
    def get_checkpoint_analysis(self) -> Dict[str, Any]:
        """Analyze checkpoint performance across all requests"""
        checkpoint_stats = defaultdict(list)
        
        # Collect all checkpoint data
        for request_metrics in self.request_metrics.values():
            for checkpoint in request_metrics.get("checkpoints", []):
                checkpoint_stats[checkpoint["name"]].append(checkpoint["duration"])
        
        # Calculate statistics for each checkpoint
        analysis = {}
        for checkpoint_name, durations in checkpoint_stats.items():
            if durations:
                target = self.targets.get(checkpoint_name, 0)
                analysis[checkpoint_name] = {
                    "count": len(durations),
                    "average": statistics.mean(durations),
                    "median": statistics.median(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "target": target,
                    "target_achievement_rate": sum(1 for d in durations if d <= target) / len(durations) if target > 0 else 1.0,
                    "over_target_count": sum(1 for d in durations if d > target) if target > 0 else 0
                }
        
        return analysis
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external analysis"""
        return {
            "system_metrics": self.get_metrics(),
            "checkpoint_analysis": self.get_checkpoint_analysis(),
            "recent_performance": self.get_recent_performance(),
            "targets": self.targets,
            "request_history": [
                {
                    "request_id": r["request_id"],
                    "total_time": r["total_time"],
                    "success": r["success"],
                    "timestamp": r["timestamp"],
                    "questions_count": r["questions_count"]
                }
                for r in list(self.request_history)
            ]
        }
