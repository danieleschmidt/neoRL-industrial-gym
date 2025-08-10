"""Advanced memory optimization for large-scale RL training."""

import os
import gc
import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Iterator
from dataclasses import dataclass
import psutil
from collections import deque
import weakref

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from ..monitoring.logger import get_logger
from ..core.types import Array


@dataclass
class MemoryConfig:
    """Configuration for memory optimization."""
    # Memory thresholds (as fraction of total memory)
    warning_threshold: float = 0.8
    critical_threshold: float = 0.9
    emergency_threshold: float = 0.95
    
    # Optimization settings
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision: bool = True
    enable_data_streaming: bool = True
    batch_size_adjustment: bool = True
    
    # Garbage collection
    gc_frequency: int = 100  # Every N training steps
    aggressive_gc: bool = False
    
    # JAX-specific
    jax_memory_fraction: float = 0.8
    enable_jax_memory_preallocation: bool = False
    
    # Data loading
    max_cached_batches: int = 10
    prefetch_factor: int = 2


class MemoryMonitor:
    """Monitor and track memory usage."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = get_logger("memory_monitor")
        
        # Memory tracking
        self.memory_history = deque(maxlen=1000)
        self.peak_memory = 0.0
        self.baseline_memory = 0.0
        
        # Alerts
        self.alert_callbacks = []
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
        # Get system memory info
        self.total_memory = psutil.virtual_memory().total
        self.baseline_memory = self.get_current_memory_usage()
        
        self.logger.info(f"Memory monitor initialized. Total memory: {self.total_memory / 1e9:.2f} GB")
    
    def get_current_memory_usage(self) -> float:
        """Get current memory usage in bytes."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss
    
    def get_memory_fraction(self) -> float:
        """Get current memory usage as fraction of total memory."""
        return self.get_current_memory_usage() / self.total_memory
    
    def record_memory_usage(self) -> Dict[str, float]:
        """Record current memory usage and return stats."""
        current_memory = self.get_current_memory_usage()
        memory_fraction = current_memory / self.total_memory
        
        # Update peak
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
        
        # Record in history
        timestamp = time.time()
        self.memory_history.append((timestamp, current_memory, memory_fraction))
        
        # Check thresholds and alert
        if memory_fraction > self.config.emergency_threshold:
            self._trigger_alert("EMERGENCY", memory_fraction)
        elif memory_fraction > self.config.critical_threshold:
            self._trigger_alert("CRITICAL", memory_fraction)
        elif memory_fraction > self.config.warning_threshold:
            self._trigger_alert("WARNING", memory_fraction)
        
        return {
            "current_memory_mb": current_memory / 1e6,
            "current_memory_fraction": memory_fraction,
            "peak_memory_mb": self.peak_memory / 1e6,
            "baseline_memory_mb": self.baseline_memory / 1e6,
            "memory_growth_mb": (current_memory - self.baseline_memory) / 1e6,
        }
    
    def _trigger_alert(self, level: str, memory_fraction: float):
        """Trigger memory alert."""
        message = f"Memory usage {level}: {memory_fraction:.1%} of total memory"
        self.logger.warning(message)
        
        for callback in self.alert_callbacks:
            try:
                callback(level, memory_fraction, message)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback):
        """Add callback for memory alerts."""
        self.alert_callbacks.append(callback)
    
    def start_continuous_monitoring(self, interval: float = 10.0):
        """Start continuous memory monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring_thread is not None and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        
        def monitor_loop():
            while not self._stop_monitoring.wait(interval):
                try:
                    self.record_memory_usage()
                except Exception as e:
                    self.logger.error(f"Memory monitoring error: {e}")
        
        self._monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitoring_thread.start()
        
        self.logger.info(f"Started continuous memory monitoring (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if self._monitoring_thread is None:
            return
        
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Memory monitoring stopped")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report."""
        current_stats = self.record_memory_usage()
        
        # Calculate statistics from history
        if len(self.memory_history) > 1:
            recent_memory = [mem for _, mem, _ in self.memory_history[-100:]]
            avg_memory = sum(recent_memory) / len(recent_memory)
            memory_trend = recent_memory[-1] - recent_memory[0] if len(recent_memory) > 1 else 0
        else:
            avg_memory = current_stats["current_memory_mb"] * 1e6
            memory_trend = 0
        
        return {
            "current": current_stats,
            "peak_memory_mb": self.peak_memory / 1e6,
            "average_memory_mb": avg_memory / 1e6,
            "memory_trend_mb": memory_trend / 1e6,
            "total_system_memory_gb": self.total_memory / 1e9,
            "history_length": len(self.memory_history),
        }


class AdaptiveBatchSizer:
    """Dynamically adjust batch size based on memory usage."""
    
    def __init__(self, initial_batch_size: int, min_batch_size: int = 8, max_batch_size: int = 1024):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.initial_batch_size = initial_batch_size
        
        self.adjustment_history = []
        self.last_adjustment_time = 0
        self.adjustment_cooldown = 60.0  # seconds
        
        self.logger = get_logger("adaptive_batch_sizer")
        
    def adjust_batch_size(self, memory_fraction: float, oom_occurred: bool = False) -> int:
        """Adjust batch size based on memory usage.
        
        Args:
            memory_fraction: Current memory usage fraction
            oom_occurred: Whether an OOM error occurred
            
        Returns:
            New batch size
        """
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_adjustment_time < self.adjustment_cooldown and not oom_occurred:
            return self.current_batch_size
        
        old_batch_size = self.current_batch_size
        
        if oom_occurred or memory_fraction > 0.9:
            # Aggressive reduction for OOM or very high memory usage
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
            reason = "OOM" if oom_occurred else "high_memory"
        elif memory_fraction > 0.8:
            # Moderate reduction for high memory usage
            self.current_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.8))
            reason = "memory_pressure"
        elif memory_fraction < 0.6 and self.current_batch_size < self.initial_batch_size:
            # Increase if memory usage is low and we're below initial size
            self.current_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
            reason = "low_memory"
        else:
            return self.current_batch_size  # No adjustment needed
        
        # Record adjustment
        if self.current_batch_size != old_batch_size:
            self.adjustment_history.append({
                "timestamp": current_time,
                "old_batch_size": old_batch_size,
                "new_batch_size": self.current_batch_size,
                "memory_fraction": memory_fraction,
                "reason": reason,
            })
            
            self.last_adjustment_time = current_time
            
            self.logger.info(
                f"Batch size adjusted: {old_batch_size} -> {self.current_batch_size} "
                f"(reason: {reason}, memory: {memory_fraction:.1%})"
            )
        
        return self.current_batch_size
    
    def reset(self):
        """Reset batch size to initial value."""
        self.current_batch_size = self.initial_batch_size
        self.adjustment_history.clear()
        self.logger.info(f"Batch size reset to {self.initial_batch_size}")


class StreamingDataLoader:
    """Memory-efficient data loader with streaming and caching."""
    
    def __init__(
        self,
        dataset: Dict[str, Array],
        batch_size: int,
        max_cached_batches: int = 10,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_cached_batches = max_cached_batches
        self.shuffle = shuffle
        
        self.n_samples = len(dataset["observations"])
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        
        # Caching
        self.batch_cache = {}
        self.cache_order = deque(maxlen=max_cached_batches)
        
        # Streaming state
        self.current_epoch = 0
        self.indices = list(range(self.n_samples))
        
        self.logger = get_logger("streaming_data_loader")
        self.logger.info(
            f"StreamingDataLoader initialized: {self.n_samples} samples, "
            f"{self.n_batches} batches, cache size {max_cached_batches}"
        )
    
    def _get_batch_indices(self, batch_idx: int) -> List[int]:
        """Get indices for a specific batch."""
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        return self.indices[start_idx:end_idx]
    
    def _create_batch(self, batch_indices: List[int]) -> Dict[str, Array]:
        """Create batch from indices."""
        batch = {}
        for key, data in self.dataset.items():
            if isinstance(data, list):
                batch[key] = [data[i] for i in batch_indices]
            else:
                # Assume numpy-like array
                batch[key] = data[batch_indices]
        return batch
    
    def get_batch(self, batch_idx: int) -> Dict[str, Array]:
        """Get a batch, using cache if available."""
        cache_key = (self.current_epoch, batch_idx)
        
        # Check cache first
        if cache_key in self.batch_cache:
            return self.batch_cache[cache_key]
        
        # Create batch
        batch_indices = self._get_batch_indices(batch_idx)
        batch = self._create_batch(batch_indices)
        
        # Cache batch if there's space
        if len(self.batch_cache) < self.max_cached_batches:
            self.batch_cache[cache_key] = batch
            self.cache_order.append(cache_key)
        else:
            # Remove oldest cached batch
            if self.cache_order:
                oldest_key = self.cache_order.popleft()
                self.batch_cache.pop(oldest_key, None)
            
            # Add new batch
            self.batch_cache[cache_key] = batch
            self.cache_order.append(cache_key)
        
        return batch
    
    def __iter__(self) -> Iterator[Dict[str, Array]]:
        """Iterate through batches."""
        # Shuffle indices if needed
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        
        for batch_idx in range(self.n_batches):
            yield self.get_batch(batch_idx)
    
    def new_epoch(self):
        """Start a new epoch (clears cache and reshuffles)."""
        self.current_epoch += 1
        self.batch_cache.clear()
        self.cache_order.clear()
        
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        
        self.logger.debug(f"Started epoch {self.current_epoch}")
    
    def clear_cache(self):
        """Clear all cached batches."""
        self.batch_cache.clear()
        self.cache_order.clear()
        gc.collect()
        self.logger.debug("Batch cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cached_batches": len(self.batch_cache),
            "max_cached_batches": self.max_cached_batches,
            "cache_hit_potential": len(self.batch_cache) / self.n_batches if self.n_batches > 0 else 0,
            "current_epoch": self.current_epoch,
        }


class MemoryOptimizer:
    """Comprehensive memory optimization system."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = get_logger("memory_optimizer")
        
        # Components
        self.monitor = MemoryMonitor(config)
        self.batch_sizer = None  # Will be initialized when needed
        
        # Optimization state
        self.gc_counter = 0
        self.optimization_active = False
        
        # JAX-specific setup
        if JAX_AVAILABLE and config.enable_mixed_precision:
            self._setup_jax_memory()
        
        # Register alert callback
        self.monitor.add_alert_callback(self._handle_memory_alert)
        
        self.logger.info("Memory optimizer initialized")
    
    def _setup_jax_memory(self):
        """Configure JAX memory settings."""
        try:
            # Configure memory preallocation
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(self.config.enable_jax_memory_preallocation).lower()
            
            # Set memory fraction
            os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(self.config.jax_memory_fraction)
            
            self.logger.info(f"JAX memory configured: preallocation={self.config.enable_jax_memory_preallocation}, fraction={self.config.jax_memory_fraction}")
        except Exception as e:
            self.logger.warning(f"Failed to configure JAX memory: {e}")
    
    def _handle_memory_alert(self, level: str, memory_fraction: float, message: str):
        """Handle memory alerts by triggering optimizations."""
        if level in ["CRITICAL", "EMERGENCY"]:
            self.emergency_cleanup()
        elif level == "WARNING":
            self.optimize_memory()
    
    def optimize_memory(self, force: bool = False):
        """Perform memory optimization."""
        if self.optimization_active and not force:
            return
        
        self.optimization_active = True
        self.logger.info("Starting memory optimization")
        
        try:
            # Record pre-optimization memory
            pre_stats = self.monitor.record_memory_usage()
            
            # Garbage collection
            if self.config.aggressive_gc:
                # Multiple GC passes
                for _ in range(3):
                    gc.collect()
            else:
                gc.collect()
            
            # JAX-specific optimizations
            if JAX_AVAILABLE:
                try:
                    # Clear JAX cache
                    jax.clear_caches()
                except Exception as e:
                    self.logger.warning(f"JAX cache clearing failed: {e}")
            
            # Record post-optimization memory
            post_stats = self.monitor.record_memory_usage()
            
            memory_freed = pre_stats["current_memory_mb"] - post_stats["current_memory_mb"]
            
            self.logger.info(
                f"Memory optimization complete. Freed: {memory_freed:.1f} MB "
                f"({pre_stats['current_memory_fraction']:.1%} -> {post_stats['current_memory_fraction']:.1%})"
            )
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}")
        finally:
            self.optimization_active = False
    
    def emergency_cleanup(self):
        """Perform emergency memory cleanup."""
        self.logger.warning("Performing emergency memory cleanup")
        
        # Aggressive garbage collection
        for _ in range(5):
            gc.collect()
        
        # Clear all caches
        if JAX_AVAILABLE:
            try:
                jax.clear_caches()
            except Exception:
                pass
        
        # Force batch size reduction if batch sizer exists
        if self.batch_sizer is not None:
            self.batch_sizer.adjust_batch_size(memory_fraction=0.95, oom_occurred=True)
        
        self.logger.warning("Emergency cleanup complete")
    
    def create_adaptive_batch_sizer(self, initial_batch_size: int) -> AdaptiveBatchSizer:
        """Create and register adaptive batch sizer."""
        self.batch_sizer = AdaptiveBatchSizer(initial_batch_size)
        return self.batch_sizer
    
    def create_streaming_dataloader(
        self,
        dataset: Dict[str, Array],
        batch_size: int,
        **kwargs
    ) -> StreamingDataLoader:
        """Create memory-efficient streaming data loader."""
        return StreamingDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            max_cached_batches=self.config.max_cached_batches,
            **kwargs
        )
    
    def should_perform_gc(self) -> bool:
        """Check if garbage collection should be performed."""
        self.gc_counter += 1
        if self.gc_counter >= self.config.gc_frequency:
            self.gc_counter = 0
            return True
        return False
    
    def training_step_hook(self):
        """Hook to call during training steps for memory management."""
        # Periodic garbage collection
        if self.should_perform_gc():
            gc.collect()
        
        # Check memory usage
        memory_fraction = self.monitor.get_memory_fraction()
        
        # Adjust batch size if needed
        if self.batch_sizer is not None and self.config.batch_size_adjustment:
            self.batch_sizer.adjust_batch_size(memory_fraction)
        
        # Automatic optimization if memory is high
        if memory_fraction > self.config.warning_threshold:
            self.optimize_memory()
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        report = {
            "memory_report": self.monitor.get_memory_report(),
            "config": {
                "warning_threshold": self.config.warning_threshold,
                "critical_threshold": self.config.critical_threshold,
                "gc_frequency": self.config.gc_frequency,
                "max_cached_batches": self.config.max_cached_batches,
            },
        }
        
        if self.batch_sizer is not None:
            report["batch_sizer"] = {
                "current_batch_size": self.batch_sizer.current_batch_size,
                "adjustment_history": self.batch_sizer.adjustment_history[-10:],  # Last 10 adjustments
            }
        
        return report
    
    def start_monitoring(self, interval: float = 30.0):
        """Start continuous memory monitoring."""
        self.monitor.start_continuous_monitoring(interval)
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitor.stop_monitoring()
    
    def __enter__(self):
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()


# Global memory optimizer
_global_memory_optimizer = None


def get_memory_optimizer(config: Optional[MemoryConfig] = None) -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    global _global_memory_optimizer
    if _global_memory_optimizer is None:
        if config is None:
            config = MemoryConfig()
        _global_memory_optimizer = MemoryOptimizer(config)
    return _global_memory_optimizer


def optimize_memory():
    """Convenience function for manual memory optimization."""
    optimizer = get_memory_optimizer()
    optimizer.optimize_memory(force=True)


def get_memory_report() -> Dict[str, Any]:
    """Get current memory usage report."""
    optimizer = get_memory_optimizer()
    return optimizer.get_optimization_report()
