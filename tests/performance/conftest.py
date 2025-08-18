"""Performance testing configuration and fixtures."""

import time
import psutil
import pytest
import threading
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from contextlib import contextmanager

import numpy as np
import jax
import jax.numpy as jnp


@dataclass
class PerformanceMetrics:
    """Performance measurement results."""
    execution_time: float
    memory_usage: float  # MB
    cpu_usage: float    # Percentage
    peak_memory: float  # MB
    throughput: Optional[float] = None  # Operations per second
    latency_p50: Optional[float] = None
    latency_p95: Optional[float] = None
    latency_p99: Optional[float] = None


class PerformanceMonitor:
    """Monitor performance metrics during test execution."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_samples = []
        self.cpu_samples = []
        self.latency_samples = []
        self._monitoring = False
        self._monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.perf_counter()
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring and return metrics."""
        self.end_time = time.perf_counter()
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        execution_time = self.end_time - self.start_time
        memory_usage = np.mean(self.memory_samples) if self.memory_samples else 0
        peak_memory = np.max(self.memory_samples) if self.memory_samples else 0
        cpu_usage = np.mean(self.cpu_samples) if self.cpu_samples else 0
        
        latency_p50 = np.percentile(self.latency_samples, 50) if self.latency_samples else None
        latency_p95 = np.percentile(self.latency_samples, 95) if self.latency_samples else None
        latency_p99 = np.percentile(self.latency_samples, 99) if self.latency_samples else None
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            peak_memory=peak_memory,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99
        )
    
    def _monitor_resources(self):
        """Monitor system resources in background thread."""
        process = psutil.Process()
        
        while self._monitoring:
            try:
                # Memory usage in MB
                memory_mb = process.memory_info().rss / (1024 * 1024)
                self.memory_samples.append(memory_mb)
                
                # CPU usage percentage
                cpu_percent = process.cpu_percent()
                self.cpu_samples.append(cpu_percent)
                
                time.sleep(0.1)  # Sample every 100ms
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
    
    def add_latency_sample(self, latency: float):
        """Add a latency measurement."""
        self.latency_samples.append(latency)
    
    @contextmanager
    def measure_operation(self):
        """Context manager to measure individual operation latency."""
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        self.add_latency_sample(end - start)


@pytest.fixture
def performance_monitor():
    """Provide performance monitoring capabilities."""
    return PerformanceMonitor()


@pytest.fixture
def performance_thresholds():
    """Define performance thresholds for different operations."""
    return {
        "policy_inference": {
            "max_latency_ms": 10.0,
            "min_throughput": 1000.0,  # operations per second
            "max_memory_mb": 100.0,
        },
        "training_step": {
            "max_latency_ms": 100.0,
            "min_throughput": 100.0,
            "max_memory_mb": 1000.0,
        },
        "environment_step": {
            "max_latency_ms": 1.0,
            "min_throughput": 10000.0,
            "max_memory_mb": 50.0,
        },
        "safety_check": {
            "max_latency_ms": 1.0,
            "min_throughput": 50000.0,
            "max_memory_mb": 10.0,
        }
    }


@pytest.fixture
def benchmark_dataset():
    """Generate benchmark datasets for performance testing."""
    sizes = {
        "small": 1000,
        "medium": 10000,
        "large": 100000,
        "xlarge": 1000000
    }
    
    datasets = {}
    for name, size in sizes.items():
        datasets[name] = {
            "observations": np.random.randn(size, 32).astype(np.float32),
            "actions": np.random.randn(size, 8).astype(np.float32),
            "rewards": np.random.randn(size).astype(np.float32),
            "next_observations": np.random.randn(size, 32).astype(np.float32),
            "terminals": np.random.choice([0, 1], size=size).astype(bool),
        }
    
    return datasets


@pytest.fixture
def stress_test_config():
    """Configuration for stress testing scenarios."""
    return {
        "concurrent_environments": [1, 10, 50, 100],
        "batch_sizes": [32, 128, 512, 2048],
        "episode_lengths": [100, 500, 1000, 5000],
        "memory_pressure": {
            "low": 0.1,      # 10% of available memory
            "medium": 0.5,   # 50% of available memory
            "high": 0.8,     # 80% of available memory
        },
        "duration_seconds": [10, 60, 300, 1800],  # 10s, 1m, 5m, 30m
    }


@pytest.fixture
def load_test_scenarios():
    """Define load testing scenarios."""
    return {
        "baseline": {
            "num_agents": 1,
            "num_environments": 1,
            "episode_length": 1000,
            "duration_seconds": 60,
        },
        "multi_agent": {
            "num_agents": 10,
            "num_environments": 10,
            "episode_length": 1000,
            "duration_seconds": 300,
        },
        "high_throughput": {
            "num_agents": 1,
            "num_environments": 100,
            "episode_length": 100,
            "duration_seconds": 120,
        },
        "long_episodes": {
            "num_agents": 5,
            "num_environments": 5,
            "episode_length": 10000,
            "duration_seconds": 600,
        }
    }


@pytest.fixture
def memory_profiler():
    """Memory profiling utilities for tests."""
    class MemoryProfiler:
        def __init__(self):
            self.snapshots = []
            self.baseline = None
        
        def take_snapshot(self, label: str = ""):
            """Take a memory snapshot."""
            import tracemalloc
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            
            current, peak = tracemalloc.get_traced_memory()
            snapshot = {
                "label": label,
                "current_mb": current / (1024 * 1024),
                "peak_mb": peak / (1024 * 1024),
                "timestamp": time.perf_counter()
            }
            self.snapshots.append(snapshot)
            return snapshot
        
        def set_baseline(self):
            """Set baseline memory usage."""
            self.baseline = self.take_snapshot("baseline")
        
        def get_memory_growth(self) -> float:
            """Get memory growth since baseline in MB."""
            if not self.baseline or not self.snapshots:
                return 0.0
            
            latest = self.snapshots[-1]
            return latest["current_mb"] - self.baseline["current_mb"]
        
        def check_memory_leak(self, threshold_mb: float = 10.0) -> bool:
            """Check if memory growth exceeds threshold."""
            growth = self.get_memory_growth()
            return growth > threshold_mb
        
        def stop_profiling(self):
            """Stop memory profiling."""
            import tracemalloc
            if tracemalloc.is_tracing():
                tracemalloc.stop()
    
    return MemoryProfiler()


@contextmanager
def performance_test(monitor: PerformanceMonitor, expected_ops: int = None):
    """Context manager for performance testing."""
    monitor.start_monitoring()
    start_time = time.perf_counter()
    
    try:
        yield monitor
    finally:
        end_time = time.perf_counter()
        metrics = monitor.stop_monitoring()
        
        if expected_ops:
            duration = end_time - start_time
            metrics.throughput = expected_ops / duration
        
        # Store metrics in monitor for later access
        monitor.final_metrics = metrics


@pytest.fixture
def gpu_benchmark_config():
    """Configuration for GPU-specific performance tests."""
    return {
        "device_memory_fraction": 0.8,
        "batch_sizes": [32, 64, 128, 256, 512, 1024],
        "model_sizes": ["small", "medium", "large"],
        "precision": ["float32", "float16", "bfloat16"],
        "compilation_modes": ["jit", "pmap", "xmap"],
    }


def assert_performance_thresholds(metrics: PerformanceMetrics, 
                                thresholds: Dict[str, Any],
                                operation_name: str = "operation"):
    """Assert that performance metrics meet thresholds."""
    if "max_latency_ms" in thresholds:
        max_latency = thresholds["max_latency_ms"] / 1000.0  # Convert to seconds
        assert metrics.execution_time <= max_latency, (
            f"{operation_name} took {metrics.execution_time:.4f}s, "
            f"exceeds threshold of {max_latency:.4f}s"
        )
    
    if "max_memory_mb" in thresholds:
        assert metrics.peak_memory <= thresholds["max_memory_mb"], (
            f"{operation_name} used {metrics.peak_memory:.2f}MB, "
            f"exceeds threshold of {thresholds['max_memory_mb']}MB"
        )
    
    if "min_throughput" in thresholds and metrics.throughput:
        assert metrics.throughput >= thresholds["min_throughput"], (
            f"{operation_name} throughput {metrics.throughput:.2f} ops/s, "
            f"below threshold of {thresholds['min_throughput']} ops/s"
        )