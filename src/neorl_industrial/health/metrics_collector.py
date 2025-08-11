"""System metrics collection for health monitoring."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import threading
import queue


@dataclass
class HealthMetrics:
    """Container for system health metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    gpu_usage: Optional[float] = None
    gpu_memory: Optional[float] = None
    network_io: Optional[Dict[str, float]] = None
    process_metrics: Dict[str, Any] = field(default_factory=dict)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age_seconds(self) -> float:
        """Age of metrics in seconds."""
        return time.time() - self.timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage, 
            "disk_usage": self.disk_usage,
            "gpu_usage": self.gpu_usage,
            "gpu_memory": self.gpu_memory,
            "network_io": self.network_io,
            "process_metrics": self.process_metrics,
            "custom_metrics": self.custom_metrics,
            "age_seconds": self.age_seconds,
        }


class MetricsCollector:
    """Continuous system metrics collector."""
    
    def __init__(
        self,
        collection_interval: float = 60.0,
        max_history: int = 1440,  # 24 hours at 1min intervals
        auto_start: bool = True,
    ) -> None:
        """Initialize metrics collector.
        
        Args:
            collection_interval: Time between collections in seconds
            max_history: Maximum number of metric entries to keep
            auto_start: Start collection thread automatically
        """
        self.collection_interval = collection_interval
        self.max_history = max_history
        
        self._metrics_history: List[HealthMetrics] = []
        self._collection_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._metrics_queue: queue.Queue = queue.Queue()
        
        # Try to import optional dependencies
        self._psutil_available = False
        self._jax_available = False
        
        try:
            import psutil
            self._psutil = psutil
            self._psutil_available = True
        except ImportError:
            self._psutil = None
        
        try:
            import jax
            self._jax = jax
            self._jax_available = True
        except ImportError:
            self._jax = None
        
        if auto_start:
            self.start_collection()
    
    def start_collection(self) -> None:
        """Start continuous metrics collection."""
        if self._collection_thread and self._collection_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self._collection_thread.start()
    
    def stop_collection(self) -> None:
        """Stop metrics collection."""
        if self._collection_thread and self._collection_thread.is_alive():
            self._stop_event.set()
            self._collection_thread.join(timeout=5.0)
    
    def _collection_loop(self) -> None:
        """Main collection loop."""
        while not self._stop_event.wait(self.collection_interval):
            try:
                metrics = self._collect_current_metrics()
                self._metrics_queue.put(metrics)
                
                # Process queue
                self._process_metrics_queue()
                
            except Exception as e:
                # Log error but continue collection
                print(f"Metrics collection error: {e}")
                continue
    
    def _process_metrics_queue(self) -> None:
        """Process queued metrics."""
        while not self._metrics_queue.empty():
            try:
                metrics = self._metrics_queue.get_nowait()
                self._metrics_history.append(metrics)
                
                # Trim history if needed
                if len(self._metrics_history) > self.max_history:
                    self._metrics_history = self._metrics_history[-self.max_history:]
                    
            except queue.Empty:
                break
    
    def _collect_current_metrics(self) -> HealthMetrics:
        """Collect current system metrics."""
        timestamp = time.time()
        
        # Default values
        cpu_usage = 0.0
        memory_usage = 0.0
        disk_usage = 0.0
        gpu_usage = None
        gpu_memory = None
        network_io = None
        process_metrics = {}
        
        # Collect system metrics if psutil available
        if self._psutil_available:
            try:
                # CPU usage
                cpu_usage = self._psutil.cpu_percent(interval=1)
                
                # Memory usage
                memory = self._psutil.virtual_memory()
                memory_usage = memory.percent
                
                # Disk usage
                disk = self._psutil.disk_usage('/')
                disk_usage = (disk.used / disk.total) * 100
                
                # Network I/O
                net_io = self._psutil.net_io_counters()
                network_io = {
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                }
                
                # Current process metrics
                current_process = self._psutil.Process()
                process_metrics = {
                    "pid": current_process.pid,
                    "cpu_percent": current_process.cpu_percent(),
                    "memory_percent": current_process.memory_percent(),
                    "num_threads": current_process.num_threads(),
                    "create_time": current_process.create_time(),
                }
                
            except Exception as e:
                process_metrics["psutil_error"] = str(e)
        
        # Collect GPU metrics if JAX available
        if self._jax_available:
            try:
                devices = self._jax.devices()
                gpu_devices = [d for d in devices if d.device_kind == "gpu"]
                
                if gpu_devices:
                    # This is a simplified check - actual GPU monitoring
                    # would require additional libraries like nvidia-ml-py
                    gpu_usage = 0.0  # Placeholder
                    gpu_memory = 0.0  # Placeholder
                    
            except Exception:
                pass
        
        return HealthMetrics(
            timestamp=timestamp,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            gpu_usage=gpu_usage,
            gpu_memory=gpu_memory,
            network_io=network_io,
            process_metrics=process_metrics,
        )
    
    def get_latest_metrics(self) -> Optional[HealthMetrics]:
        """Get the most recent metrics."""
        # Process any pending metrics
        self._process_metrics_queue()
        
        if not self._metrics_history:
            return None
        
        return self._metrics_history[-1]
    
    def get_metrics_history(
        self,
        hours: Optional[float] = None,
        count: Optional[int] = None
    ) -> List[HealthMetrics]:
        """Get metrics history.
        
        Args:
            hours: Number of hours of history to return
            count: Number of most recent entries to return
            
        Returns:
            List of historical metrics
        """
        # Process any pending metrics
        self._process_metrics_queue()
        
        history = self._metrics_history
        
        if hours is not None:
            cutoff_time = time.time() - (hours * 3600)
            history = [m for m in history if m.timestamp >= cutoff_time]
        
        if count is not None:
            history = history[-count:]
        
        return history
    
    def get_metrics_summary(self, hours: float = 1.0) -> Dict[str, Any]:
        """Get summary statistics for recent metrics.
        
        Args:
            hours: Hours of history to summarize
            
        Returns:
            Summary statistics dictionary
        """
        metrics_list = self.get_metrics_history(hours=hours)
        
        if not metrics_list:
            return {"error": "No metrics available"}
        
        # Calculate statistics
        cpu_values = [m.cpu_usage for m in metrics_list]
        memory_values = [m.memory_usage for m in metrics_list]
        disk_values = [m.disk_usage for m in metrics_list]
        
        summary = {
            "period_hours": hours,
            "sample_count": len(metrics_list),
            "timestamp_range": {
                "start": metrics_list[0].timestamp,
                "end": metrics_list[-1].timestamp,
            },
            "cpu_usage": {
                "current": cpu_values[-1] if cpu_values else 0,
                "average": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "min": min(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
            },
            "memory_usage": {
                "current": memory_values[-1] if memory_values else 0,
                "average": sum(memory_values) / len(memory_values) if memory_values else 0,
                "min": min(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0,
            },
            "disk_usage": {
                "current": disk_values[-1] if disk_values else 0,
                "average": sum(disk_values) / len(disk_values) if disk_values else 0,
                "min": min(disk_values) if disk_values else 0,
                "max": max(disk_values) if disk_values else 0,
            },
        }
        
        # GPU statistics if available
        gpu_values = [m.gpu_usage for m in metrics_list if m.gpu_usage is not None]
        if gpu_values:
            summary["gpu_usage"] = {
                "current": gpu_values[-1],
                "average": sum(gpu_values) / len(gpu_values),
                "min": min(gpu_values),
                "max": max(gpu_values),
            }
        
        return summary
    
    def add_custom_metric(self, name: str, value: Any) -> None:
        """Add custom metric to next collection.
        
        Args:
            name: Metric name
            value: Metric value
        """
        # Get latest metrics or create new
        if self._metrics_history:
            latest = self._metrics_history[-1]
            if latest.age_seconds < self.collection_interval / 2:
                # Add to existing recent metrics
                latest.custom_metrics[name] = value
                return
        
        # Create new metrics entry with custom metric
        metrics = self._collect_current_metrics()
        metrics.custom_metrics[name] = value
        self._metrics_history.append(metrics)
        
        # Trim history if needed
        if len(self._metrics_history) > self.max_history:
            self._metrics_history = self._metrics_history[-self.max_history:]
    
    def export_metrics(
        self, 
        format: str = "json",
        hours: Optional[float] = None
    ) -> str:
        """Export metrics in specified format.
        
        Args:
            format: Export format ('json', 'csv')
            hours: Hours of history to export
            
        Returns:
            Formatted metrics string
        """
        metrics_list = self.get_metrics_history(hours=hours)
        
        if format.lower() == "json":
            import json
            return json.dumps([m.to_dict() for m in metrics_list], indent=2)
        
        elif format.lower() == "csv":
            lines = []
            # Header
            lines.append(
                "timestamp,cpu_usage,memory_usage,disk_usage,gpu_usage,gpu_memory"
            )
            
            # Data rows
            for metrics in metrics_list:
                lines.append(
                    f"{metrics.timestamp},{metrics.cpu_usage},"
                    f"{metrics.memory_usage},{metrics.disk_usage},"
                    f"{metrics.gpu_usage or ''},{metrics.gpu_memory or ''}"
                )
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_collection()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_collection()


def collect_system_metrics() -> HealthMetrics:
    """Collect current system metrics (one-time collection).
    
    Returns:
        Current system health metrics
    """
    collector = MetricsCollector(auto_start=False)
    return collector._collect_current_metrics()