"""Performance monitoring and metrics collection for industrial RL."""

import time
import psutil
import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
from contextlib import contextmanager

import jax
import numpy as np

from .logger import get_logger


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    cpu_usage: float
    memory_usage: float
    gpu_memory_usage: Optional[float]
    training_step_time: float
    inference_time: float
    throughput: float
    error_count: int
    safety_violation_count: int


class PerformanceMonitor:
    """Real-time performance monitoring for RL agents and environments."""
    
    def __init__(
        self,
        name: str,
        sampling_interval: float = 1.0,
        history_size: int = 1000,
        enable_detailed_profiling: bool = False,
    ):
        """Initialize performance monitor.
        
        Args:
            name: Monitor name
            sampling_interval: Seconds between metric samples
            history_size: Number of historical samples to keep
            enable_detailed_profiling: Enable detailed JAX profiling
        """
        self.name = name
        self.sampling_interval = sampling_interval
        self.history_size = history_size
        self.enable_detailed_profiling = enable_detailed_profiling
        
        # Metrics storage
        self.metrics_history = deque(maxlen=history_size)
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        self.active_timers = {}
        
        # System monitoring
        self.process = psutil.Process()
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Logger
        self.logger = get_logger(f"performance_{name}")
        
        # JAX profiling
        self.profiler_active = False
        
    def start_monitoring(self) -> None:
        """Start background performance monitoring."""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info(f"Performance monitoring started for {self.name}")
        
    def stop_monitoring(self) -> None:
        """Stop background performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
        self.logger.info(f"Performance monitoring stopped for {self.name}")
        
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_usage = psutil.cpu_percent(interval=None)
                memory_info = self.process.memory_info()
                memory_usage = memory_info.rss / 1024 / 1024  # MB
                
                # Try to get GPU memory if available
                gpu_memory = None
                try:
                    # This would need proper GPU monitoring library
                    pass
                except:
                    pass
                
                # Calculate derived metrics
                recent_training_times = self.timers.get("training_step", [])
                avg_training_time = np.mean(recent_training_times[-100:]) if recent_training_times else 0.0
                
                recent_inference_times = self.timers.get("inference", [])
                avg_inference_time = np.mean(recent_inference_times[-100:]) if recent_inference_times else 0.0
                
                throughput = 1.0 / max(avg_training_time, 0.001)
                
                metrics = PerformanceMetrics(
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    gpu_memory_usage=gpu_memory,
                    training_step_time=avg_training_time,
                    inference_time=avg_inference_time,
                    throughput=throughput,
                    error_count=self.counters["errors"],
                    safety_violation_count=self.counters["safety_violations"]
                )
                
                self.metrics_history.append((time.time(), metrics))
                
                # Log warnings for concerning metrics
                if cpu_usage > 90:
                    self.logger.warning(f"High CPU usage: {cpu_usage:.1f}%")
                if memory_usage > 8000:  # 8GB
                    self.logger.warning(f"High memory usage: {memory_usage:.1f}MB")
                if avg_training_time > 1.0:  # 1 second per step
                    self.logger.warning(f"Slow training step: {avg_training_time:.3f}s")
                    
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                
            time.sleep(self.sampling_interval)
            
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        start_time = time.time()
        self.active_timers[operation_name] = start_time
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.timers[operation_name].append(elapsed)
            
            # Keep only recent timings
            if len(self.timers[operation_name]) > self.history_size:
                self.timers[operation_name] = self.timers[operation_name][-self.history_size:]
                
            if operation_name in self.active_timers:
                del self.active_timers[operation_name]
                
    def record_event(self, event_type: str, count: int = 1) -> None:
        """Record occurrence of an event."""
        self.counters[event_type] += count
        
    def record_safety_violation(self, violation_type: str, details: Dict[str, Any]) -> None:
        """Record a safety violation event."""
        self.counters["safety_violations"] += 1
        self.counters[f"safety_violations_{violation_type}"] += 1
        
        self.logger.safety_event(
            event_type=violation_type,
            severity="HIGH",
            details=details,
            agent_id=self.name
        )
        
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics."""
        if not self.metrics_history:
            return None
        return self.metrics_history[-1][1]
        
    def get_metrics_summary(self, window_seconds: float = 300) -> Dict[str, Any]:
        """Get summary statistics for recent metrics.
        
        Args:
            window_seconds: Time window for summary (seconds)
            
        Returns:
            Dictionary of summary statistics
        """
        if not self.metrics_history:
            return {}
            
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Filter recent metrics
        recent_metrics = [
            (timestamp, metrics) for timestamp, metrics in self.metrics_history
            if timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {}
            
        # Extract metric values
        cpu_values = [m.cpu_usage for _, m in recent_metrics]
        memory_values = [m.memory_usage for _, m in recent_metrics]
        training_times = [m.training_step_time for _, m in recent_metrics if m.training_step_time > 0]
        inference_times = [m.inference_time for _, m in recent_metrics if m.inference_time > 0]
        
        summary = {
            "window_seconds": window_seconds,
            "sample_count": len(recent_metrics),
            "cpu_usage": {
                "mean": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "min": np.min(cpu_values),
                "std": np.std(cpu_values),
            },
            "memory_usage_mb": {
                "mean": np.mean(memory_values),
                "max": np.max(memory_values),
                "min": np.min(memory_values),
                "std": np.std(memory_values),
            },
        }
        
        if training_times:
            summary["training_step_time"] = {
                "mean": np.mean(training_times),
                "max": np.max(training_times),
                "min": np.min(training_times),
                "std": np.std(training_times),
                "p95": np.percentile(training_times, 95),
                "p99": np.percentile(training_times, 99),
            }
            
        if inference_times:
            summary["inference_time"] = {
                "mean": np.mean(inference_times),
                "max": np.max(inference_times),
                "min": np.min(inference_times),
                "std": np.std(inference_times),
                "p95": np.percentile(inference_times, 95),
                "p99": np.percentile(inference_times, 99),
            }
            
        # Add counters
        summary["counters"] = dict(self.counters)
        
        return summary
        
    def log_performance_report(self, window_seconds: float = 300) -> None:
        """Log a comprehensive performance report."""
        summary = self.get_metrics_summary(window_seconds)
        
        if not summary:
            self.logger.warning("No performance metrics available for report")
            return
            
        self.logger.info(f"=== Performance Report ({self.name}) ===")
        self.logger.info(f"Time window: {window_seconds}s | Samples: {summary['sample_count']}")
        
        # CPU and Memory
        cpu = summary["cpu_usage"]
        memory = summary["memory_usage_mb"]
        self.logger.info(f"CPU: {cpu['mean']:.1f}% (max: {cpu['max']:.1f}%)")
        self.logger.info(f"Memory: {memory['mean']:.1f}MB (max: {memory['max']:.1f}MB)")
        
        # Training performance
        if "training_step_time" in summary:
            train = summary["training_step_time"]
            self.logger.info(f"Training step: {train['mean']:.3f}s (p95: {train['p95']:.3f}s)")
            
        if "inference_time" in summary:
            infer = summary["inference_time"]
            self.logger.info(f"Inference: {infer['mean']:.3f}s (p95: {infer['p95']:.3f}s)")
            
        # Events
        counters = summary["counters"]
        if counters:
            event_strs = [f"{k}: {v}" for k, v in counters.items()]
            self.logger.info(f"Events: {', '.join(event_strs)}")
            
    def start_profiling(self) -> None:
        """Start JAX profiling if enabled."""
        if not self.enable_detailed_profiling:
            return
            
        try:
            # Start JAX profiler
            jax.profiler.start_trace("./profiling")
            self.profiler_active = True
            self.logger.info("JAX profiling started")
        except Exception as e:
            self.logger.warning(f"Could not start JAX profiling: {e}")
            
    def stop_profiling(self) -> None:
        """Stop JAX profiling."""
        if not self.profiler_active:
            return
            
        try:
            jax.profiler.stop_trace()
            self.profiler_active = False
            self.logger.info("JAX profiling stopped - traces saved to ./profiling")
        except Exception as e:
            self.logger.warning(f"Could not stop JAX profiling: {e}")
            
    def health_check(self) -> Dict[str, str]:
        """Perform system health check.
        
        Returns:
            Dictionary with health status for different components
        """
        status = {}
        
        # Check current metrics
        current = self.get_current_metrics()
        if current is None:
            status["monitoring"] = "UNHEALTHY: No metrics available"
        else:
            status["monitoring"] = "HEALTHY"
            
            # Check system resources
            if current.cpu_usage > 95:
                status["cpu"] = "CRITICAL: CPU usage above 95%"
            elif current.cpu_usage > 85:
                status["cpu"] = "WARNING: CPU usage above 85%"
            else:
                status["cpu"] = "HEALTHY"
                
            if current.memory_usage > 16000:  # 16GB
                status["memory"] = "CRITICAL: Memory usage above 16GB"
            elif current.memory_usage > 8000:  # 8GB
                status["memory"] = "WARNING: Memory usage above 8GB"
            else:
                status["memory"] = "HEALTHY"
                
            # Check performance
            if current.training_step_time > 2.0:
                status["training_speed"] = "WARNING: Slow training steps"
            else:
                status["training_speed"] = "HEALTHY"
                
            # Check error rates
            if self.counters["errors"] > 100:
                status["errors"] = "WARNING: High error count"
            else:
                status["errors"] = "HEALTHY"
                
            # Check safety violations
            if self.counters["safety_violations"] > 10:
                status["safety"] = "CRITICAL: High safety violation count"
            elif self.counters["safety_violations"] > 0:
                status["safety"] = "WARNING: Safety violations detected"
            else:
                status["safety"] = "HEALTHY"
                
        return status


# Global monitor registry
_monitors: Dict[str, PerformanceMonitor] = {}


def get_performance_monitor(
    name: str,
    **kwargs
) -> PerformanceMonitor:
    """Get or create a performance monitor.
    
    Args:
        name: Monitor name
        **kwargs: Arguments for PerformanceMonitor
        
    Returns:
        Performance monitor instance
    """
    if name not in _monitors:
        _monitors[name] = PerformanceMonitor(name=name, **kwargs)
        
    return _monitors[name]


def get_all_monitors() -> Dict[str, PerformanceMonitor]:
    """Get all active performance monitors."""
    return _monitors.copy()


def stop_all_monitoring() -> None:
    """Stop all active performance monitors."""
    for monitor in _monitors.values():
        monitor.stop_monitoring()