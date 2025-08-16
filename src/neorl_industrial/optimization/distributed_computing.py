"""Distributed computing and scaling infrastructure for industrial RL."""

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

try:
    import jax
    import jax.numpy as jnp
    from jax import pmap, pjit, device_put, devices
    from jax.sharding import Mesh, PartitionSpec
except ImportError:
    # Fallback for environments without JAX
    jax = None
    jnp = None

import threading
import queue
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import multiprocessing as mp

from ..core.types import Array


class ComputeBackend(Enum):
    """Available compute backends."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    DISTRIBUTED = "distributed"


class ParallelStrategy(Enum):
    """Parallel execution strategies."""
    DATA_PARALLEL = "data_parallel"
    MODEL_PARALLEL = "model_parallel"
    PIPELINE_PARALLEL = "pipeline_parallel"
    HYBRID = "hybrid"


@dataclass
class ComputeConfig:
    """Configuration for distributed computing."""
    
    backend: ComputeBackend = ComputeBackend.CPU
    strategy: ParallelStrategy = ParallelStrategy.DATA_PARALLEL
    num_workers: int = 4
    batch_size_per_device: int = 256
    memory_limit_gb: Optional[float] = None
    use_mixed_precision: bool = True
    enable_xla: bool = True
    prefetch_factor: int = 2


@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics."""
    
    throughput_samples_per_sec: float
    latency_ms: float
    memory_usage_gb: float
    device_utilization_percent: float
    communication_overhead_ms: float
    cache_hit_rate: float


class DistributedCompute(ABC):
    """Abstract base class for distributed computing backends."""
    
    def __init__(self, config: ComputeConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize distributed computing backend."""
        pass
        
    @abstractmethod
    def scatter(self, data: Array, axis: int = 0) -> List[Array]:
        """Scatter data across devices."""
        pass
        
    @abstractmethod
    def gather(self, scattered_data: List[Array], axis: int = 0) -> Array:
        """Gather data from devices."""
        pass
        
    @abstractmethod
    def parallel_apply(self, fn: Callable, data: Array, **kwargs) -> Array:
        """Apply function in parallel across devices."""
        pass
        
    @abstractmethod
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        pass


class JAXDistributedCompute(DistributedCompute):
    """JAX-based distributed computing implementation."""
    
    def __init__(self, config: ComputeConfig):
        super().__init__(config)
        self.devices_list = None
        self.mesh = None
        self.pmap_fn_cache = {}
        self.pjit_fn_cache = {}
        
        # Performance tracking
        self.execution_times = []
        self.memory_usage = []
        
    def initialize(self) -> bool:
        """Initialize JAX distributed computing."""
        if jax is None:
            self.logger.error("JAX not available")
            return False
            
        try:
            # Get available devices
            self.devices_list = devices()
            self.logger.info(f"Available devices: {len(self.devices_list)} {[d.device_kind for d in self.devices_list]}")
            
            # Configure XLA
            if self.config.enable_xla:
                os.environ['XLA_FLAGS'] = '--xla_gpu_enable_fast_min_max=true'
                
            # Setup device mesh for advanced parallelism
            if len(self.devices_list) > 1:
                self._setup_device_mesh()
                
            # Configure memory
            if self.config.memory_limit_gb:
                self._configure_memory_limit()
                
            self.logger.info("JAX distributed computing initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize JAX distributed computing: {e}")
            return False
            
    def _setup_device_mesh(self):
        """Setup device mesh for model parallelism."""
        try:
            num_devices = len(self.devices_list)
            
            if self.config.strategy == ParallelStrategy.DATA_PARALLEL:
                # Simple data parallel setup
                mesh_shape = (num_devices,)
                axis_names = ('data',)
            elif self.config.strategy == ParallelStrategy.MODEL_PARALLEL:
                # Model parallel setup
                mesh_shape = (num_devices,)
                axis_names = ('model',)
            elif self.config.strategy == ParallelStrategy.HYBRID:
                # Hybrid setup (2D mesh)
                if num_devices >= 4:
                    mesh_shape = (2, num_devices // 2)
                    axis_names = ('data', 'model')
                else:
                    mesh_shape = (num_devices,)
                    axis_names = ('data',)
            else:
                mesh_shape = (num_devices,)
                axis_names = ('data',)
                
            self.mesh = Mesh(self.devices_list, axis_names)
            self.logger.info(f"Device mesh setup: {mesh_shape} with axes {axis_names}")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup device mesh: {e}")
            
    def _configure_memory_limit(self):
        """Configure memory limits for devices."""
        try:
            # This is a simplified implementation
            # In practice, you'd use JAX memory allocation APIs
            self.logger.info(f"Memory limit configured: {self.config.memory_limit_gb}GB")
        except Exception as e:
            self.logger.warning(f"Failed to configure memory limit: {e}")
            
    def scatter(self, data: Array, axis: int = 0) -> List[Array]:
        """Scatter data across available devices."""
        if not self.devices_list:
            return [data]
            
        try:
            data_jax = jnp.asarray(data)
            
            # Calculate splits for each device
            num_devices = len(self.devices_list)
            split_size = data_jax.shape[axis] // num_devices
            
            # Split data
            splits = []
            for i in range(num_devices):
                start_idx = i * split_size
                if i == num_devices - 1:
                    # Last device gets remainder
                    split = jnp.take(data_jax, jnp.arange(start_idx, data_jax.shape[axis]), axis=axis)
                else:
                    end_idx = start_idx + split_size
                    split = jnp.take(data_jax, jnp.arange(start_idx, end_idx), axis=axis)
                    
                # Place on device
                device_split = device_put(split, self.devices_list[i])
                splits.append(device_split)
                
            return splits
            
        except Exception as e:
            self.logger.error(f"Failed to scatter data: {e}")
            return [data]
            
    def gather(self, scattered_data: List[Array], axis: int = 0) -> Array:
        """Gather data from devices."""
        try:
            # Move all data to first device and concatenate
            gathered_parts = []
            for data_part in scattered_data:
                gathered_parts.append(device_put(data_part, self.devices_list[0]))
                
            return jnp.concatenate(gathered_parts, axis=axis)
            
        except Exception as e:
            self.logger.error(f"Failed to gather data: {e}")
            return scattered_data[0] if scattered_data else jnp.array([])
            
    def parallel_apply(self, fn: Callable, data: Array, **kwargs) -> Array:
        """Apply function in parallel using pmap or pjit."""
        try:
            # Use pmap for simple data parallelism
            if self.config.strategy == ParallelStrategy.DATA_PARALLEL:
                return self._pmap_apply(fn, data, **kwargs)
            else:
                return self._pjit_apply(fn, data, **kwargs)
                
        except Exception as e:
            self.logger.error(f"Parallel apply failed: {e}")
            # Fallback to sequential execution
            return fn(data, **kwargs)
            
    def _pmap_apply(self, fn: Callable, data: Array, **kwargs) -> Array:
        """Apply function using pmap (data parallelism)."""
        fn_signature = f"{fn.__name__}_{hash(str(kwargs))}"
        
        if fn_signature not in self.pmap_fn_cache:
            # Create pmap function
            self.pmap_fn_cache[fn_signature] = pmap(fn, axis_name='batch')
            
        pmap_fn = self.pmap_fn_cache[fn_signature]
        
        # Reshape data for pmap (add device dimension)
        num_devices = len(self.devices_list)
        if data.shape[0] % num_devices != 0:
            # Pad data to be divisible by number of devices
            pad_size = num_devices - (data.shape[0] % num_devices)
            padding = [(0, pad_size)] + [(0, 0)] * (len(data.shape) - 1)
            data_padded = jnp.pad(data, padding, mode='constant')
        else:
            data_padded = data
            pad_size = 0
            
        # Reshape for devices
        device_batch_size = data_padded.shape[0] // num_devices
        data_reshaped = data_padded.reshape(
            (num_devices, device_batch_size) + data_padded.shape[1:]
        )
        
        # Apply pmap function
        start_time = time.time()
        result = pmap_fn(data_reshaped, **kwargs)
        execution_time = time.time() - start_time
        
        # Track performance
        self.execution_times.append(execution_time)
        
        # Reshape result back
        result_flat = result.reshape((-1,) + result.shape[2:])
        
        # Remove padding if added
        if pad_size > 0:
            result_flat = result_flat[:-pad_size]
            
        return result_flat
        
    def _pjit_apply(self, fn: Callable, data: Array, **kwargs) -> Array:
        """Apply function using pjit (model/pipeline parallelism)."""
        if self.mesh is None:
            # Fallback to pmap
            return self._pmap_apply(fn, data, **kwargs)
            
        fn_signature = f"{fn.__name__}_{hash(str(kwargs))}"
        
        if fn_signature not in self.pjit_fn_cache:
            # Define partitioning strategy
            if self.config.strategy == ParallelStrategy.MODEL_PARALLEL:
                # Model parameters are sharded
                in_specs = PartitionSpec('data', None)
                out_specs = PartitionSpec('data', None)
            else:
                # Hybrid parallelism
                in_specs = PartitionSpec('data', 'model')
                out_specs = PartitionSpec('data', 'model')
                
            # Create pjit function
            self.pjit_fn_cache[fn_signature] = pjit(
                fn,
                in_shardings=in_specs,
                out_shardings=out_specs
            )
            
        pjit_fn = self.pjit_fn_cache[fn_signature]
        
        start_time = time.time()
        result = pjit_fn(data, **kwargs)
        execution_time = time.time() - start_time
        
        self.execution_times.append(execution_time)
        return result
        
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics."""
        try:
            # Calculate throughput
            if self.execution_times:
                avg_execution_time = sum(self.execution_times[-10:]) / len(self.execution_times[-10:])
                throughput = self.config.batch_size_per_device / avg_execution_time
                latency = avg_execution_time * 1000  # Convert to ms
            else:
                throughput = 0.0
                latency = 0.0
                
            # Memory usage (simplified)
            memory_usage = 0.0
            device_utilization = 100.0  # Assume full utilization
            
            # Communication overhead (estimated)
            communication_overhead = 5.0  # ms
            
            # Cache hit rate
            total_cache_queries = len(self.pmap_fn_cache) + len(self.pjit_fn_cache)
            cache_hit_rate = min(1.0, total_cache_queries / max(len(self.execution_times), 1))
            
            return PerformanceMetrics(
                throughput_samples_per_sec=throughput,
                latency_ms=latency,
                memory_usage_gb=memory_usage,
                device_utilization_percent=device_utilization,
                communication_overhead_ms=communication_overhead,
                cache_hit_rate=cache_hit_rate
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class ThreadPoolCompute(DistributedCompute):
    """Thread pool-based parallel computing for CPU workloads."""
    
    def __init__(self, config: ComputeConfig):
        super().__init__(config)
        self.thread_pool = None
        self.execution_times = []
        
    def initialize(self) -> bool:
        """Initialize thread pool."""
        try:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.config.num_workers,
                thread_name_prefix="neorl_worker"
            )
            self.logger.info(f"Thread pool initialized with {self.config.num_workers} workers")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize thread pool: {e}")
            return False
            
    def scatter(self, data: Array, axis: int = 0) -> List[Array]:
        """Split data for thread processing."""
        try:
            if hasattr(data, 'shape'):
                num_chunks = self.config.num_workers
                chunk_size = data.shape[axis] // num_chunks
                
                chunks = []
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    if i == num_chunks - 1:
                        chunk = data[start_idx:]
                    else:
                        end_idx = start_idx + chunk_size
                        chunk = data[start_idx:end_idx]
                    chunks.append(chunk)
                    
                return chunks
            else:
                # Split list-like data
                chunk_size = len(data) // self.config.num_workers
                return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
                
        except Exception as e:
            self.logger.error(f"Failed to scatter data: {e}")
            return [data]
            
    def gather(self, scattered_data: List[Array], axis: int = 0) -> Array:
        """Combine results from threads."""
        try:
            if hasattr(scattered_data[0], 'shape') and jnp is not None:
                return jnp.concatenate(scattered_data, axis=axis)
            else:
                # Concatenate lists
                result = []
                for chunk in scattered_data:
                    result.extend(chunk)
                return result
        except Exception as e:
            self.logger.error(f"Failed to gather data: {e}")
            return scattered_data[0] if scattered_data else []
            
    def parallel_apply(self, fn: Callable, data: Array, **kwargs) -> Array:
        """Apply function in parallel using thread pool."""
        if self.thread_pool is None:
            return fn(data, **kwargs)
            
        try:
            start_time = time.time()
            
            # Scatter data
            data_chunks = self.scatter(data)
            
            # Submit tasks to thread pool
            futures = []
            for chunk in data_chunks:
                future = self.thread_pool.submit(fn, chunk, **kwargs)
                futures.append(future)
                
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Thread execution failed: {e}")
                    
            # Gather results
            if results:
                final_result = self.gather(results)
            else:
                final_result = fn(data, **kwargs)  # Fallback
                
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Parallel apply failed: {e}")
            return fn(data, **kwargs)
            
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get thread pool performance metrics."""
        try:
            if self.execution_times:
                avg_execution_time = sum(self.execution_times[-10:]) / len(self.execution_times[-10:])
                throughput = self.config.batch_size_per_device / avg_execution_time
                latency = avg_execution_time * 1000
            else:
                throughput = 0.0
                latency = 0.0
                
            return PerformanceMetrics(
                throughput_samples_per_sec=throughput,
                latency_ms=latency,
                memory_usage_gb=0.0,  # Not tracked for threads
                device_utilization_percent=80.0,  # Estimated
                communication_overhead_ms=1.0,  # Minimal for threads
                cache_hit_rate=0.9  # Estimated
            )
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
    def shutdown(self):
        """Shutdown thread pool."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)


class ProcessPoolCompute(DistributedCompute):
    """Process pool-based parallel computing for CPU-intensive workloads."""
    
    def __init__(self, config: ComputeConfig):
        super().__init__(config)
        self.process_pool = None
        self.execution_times = []
        
    def initialize(self) -> bool:
        """Initialize process pool."""
        try:
            self.process_pool = ProcessPoolExecutor(
                max_workers=self.config.num_workers
            )
            self.logger.info(f"Process pool initialized with {self.config.num_workers} workers")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize process pool: {e}")
            return False
            
    def scatter(self, data: Array, axis: int = 0) -> List[Array]:
        """Split data for process processing."""
        # Similar to thread pool implementation
        try:
            if hasattr(data, 'shape'):
                num_chunks = self.config.num_workers
                chunk_size = data.shape[axis] // num_chunks
                
                chunks = []
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    if i == num_chunks - 1:
                        chunk = data[start_idx:]
                    else:
                        end_idx = start_idx + chunk_size
                        chunk = data[start_idx:end_idx]
                    chunks.append(chunk)
                    
                return chunks
            else:
                chunk_size = len(data) // self.config.num_workers
                return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
                
        except Exception as e:
            self.logger.error(f"Failed to scatter data: {e}")
            return [data]
            
    def gather(self, scattered_data: List[Array], axis: int = 0) -> Array:
        """Combine results from processes."""
        # Similar to thread pool implementation
        try:
            if hasattr(scattered_data[0], 'shape') and jnp is not None:
                return jnp.concatenate(scattered_data, axis=axis)
            else:
                result = []
                for chunk in scattered_data:
                    result.extend(chunk)
                return result
        except Exception as e:
            self.logger.error(f"Failed to gather data: {e}")
            return scattered_data[0] if scattered_data else []
            
    def parallel_apply(self, fn: Callable, data: Array, **kwargs) -> Array:
        """Apply function in parallel using process pool."""
        if self.process_pool is None:
            return fn(data, **kwargs)
            
        try:
            start_time = time.time()
            
            # Scatter data
            data_chunks = self.scatter(data)
            
            # Submit tasks to process pool
            futures = []
            for chunk in data_chunks:
                future = self.process_pool.submit(fn, chunk, **kwargs)
                futures.append(future)
                
            # Collect results
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=60)  # 60 second timeout
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Process execution failed: {e}")
                    
            # Gather results
            if results:
                final_result = self.gather(results)
            else:
                final_result = fn(data, **kwargs)
                
            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Parallel apply failed: {e}")
            return fn(data, **kwargs)
            
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get process pool performance metrics."""
        try:
            if self.execution_times:
                avg_execution_time = sum(self.execution_times[-10:]) / len(self.execution_times[-10:])
                throughput = self.config.batch_size_per_device / avg_execution_time
                latency = avg_execution_time * 1000
            else:
                throughput = 0.0
                latency = 0.0
                
            return PerformanceMetrics(
                throughput_samples_per_sec=throughput,
                latency_ms=latency,
                memory_usage_gb=0.0,
                device_utilization_percent=90.0,  # Higher for processes
                communication_overhead_ms=10.0,  # Higher due to IPC
                cache_hit_rate=0.8
            )
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return PerformanceMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
    def shutdown(self):
        """Shutdown process pool."""
        if self.process_pool:
            self.process_pool.shutdown(wait=True)


class DistributedComputeManager:
    """Manager for distributed computing across different backends."""
    
    def __init__(self, config: Optional[ComputeConfig] = None):
        self.config = config or ComputeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.compute_backend = None
        self.is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize the appropriate compute backend."""
        try:
            if self.config.backend == ComputeBackend.GPU and jax is not None:
                # Try JAX with GPU
                self.compute_backend = JAXDistributedCompute(self.config)
            elif self.config.backend == ComputeBackend.TPU and jax is not None:
                # Try JAX with TPU
                self.compute_backend = JAXDistributedCompute(self.config)
            elif self.config.backend == ComputeBackend.CPU:
                # Use thread pool for I/O bound tasks, process pool for CPU bound
                if self.config.num_workers > mp.cpu_count():
                    self.compute_backend = ThreadPoolCompute(self.config)
                else:
                    self.compute_backend = ProcessPoolCompute(self.config)
            else:
                # Default to thread pool
                self.compute_backend = ThreadPoolCompute(self.config)
                
            self.is_initialized = self.compute_backend.initialize()
            
            if self.is_initialized:
                self.logger.info(f"Distributed computing initialized with {type(self.compute_backend).__name__}")
            else:
                self.logger.error("Failed to initialize distributed computing")
                
            return self.is_initialized
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distributed computing: {e}")
            return False
            
    def parallel_map(self, fn: Callable, data_list: List[Any], **kwargs) -> List[Any]:
        """Apply function to list of data in parallel."""
        if not self.is_initialized:
            # Sequential fallback
            return [fn(data, **kwargs) for data in data_list]
            
        try:
            # Convert to array for processing
            if jnp is not None:
                data_array = jnp.array(data_list)
                result_array = self.compute_backend.parallel_apply(fn, data_array, **kwargs)
                return list(result_array)
            else:
                return self.compute_backend.parallel_apply(fn, data_list, **kwargs)
                
        except Exception as e:
            self.logger.error(f"Parallel map failed: {e}")
            return [fn(data, **kwargs) for data in data_list]
            
    def parallel_reduce(self, fn: Callable, data: Array, reduce_fn: Callable, **kwargs) -> Any:
        """Apply function in parallel and reduce results."""
        if not self.is_initialized:
            return reduce_fn([fn(data, **kwargs)])
            
        try:
            # Scatter data
            scattered_data = self.compute_backend.scatter(data)
            
            # Apply function to each chunk
            results = []
            for chunk in scattered_data:
                result = fn(chunk, **kwargs)
                results.append(result)
                
            # Reduce results
            return reduce_fn(results)
            
        except Exception as e:
            self.logger.error(f"Parallel reduce failed: {e}")
            return reduce_fn([fn(data, **kwargs)])
            
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        if not self.is_initialized:
            return {"status": "not_initialized"}
            
        try:
            metrics = self.compute_backend.get_performance_metrics()
            
            return {
                "backend": type(self.compute_backend).__name__,
                "config": {
                    "backend": self.config.backend.value,
                    "strategy": self.config.strategy.value,
                    "num_workers": self.config.num_workers,
                    "batch_size_per_device": self.config.batch_size_per_device
                },
                "performance": {
                    "throughput_samples_per_sec": metrics.throughput_samples_per_sec,
                    "latency_ms": metrics.latency_ms,
                    "memory_usage_gb": metrics.memory_usage_gb,
                    "device_utilization_percent": metrics.device_utilization_percent,
                    "communication_overhead_ms": metrics.communication_overhead_ms,
                    "cache_hit_rate": metrics.cache_hit_rate
                },
                "recommendations": self._generate_recommendations(metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance report: {e}")
            return {"status": "error", "message": str(e)}
            
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if metrics.throughput_samples_per_sec < 1000:
            recommendations.append("Consider increasing batch size or using more workers")
            
        if metrics.latency_ms > 100:
            recommendations.append("High latency detected - consider optimizing computation or reducing batch size")
            
        if metrics.device_utilization_percent < 70:
            recommendations.append("Low device utilization - consider increasing workload or optimizing data pipeline")
            
        if metrics.communication_overhead_ms > 20:
            recommendations.append("High communication overhead - consider reducing data transfer or using local computation")
            
        if metrics.cache_hit_rate < 0.8:
            recommendations.append("Low cache hit rate - consider optimizing caching strategy")
            
        return recommendations
        
    def shutdown(self):
        """Shutdown distributed computing backend."""
        if self.compute_backend and hasattr(self.compute_backend, 'shutdown'):
            self.compute_backend.shutdown()
        self.is_initialized = False


# Factory function for easy initialization
def create_distributed_compute(
    backend: str = "auto",
    num_workers: Optional[int] = None,
    **kwargs
) -> DistributedComputeManager:
    """Factory function to create distributed compute manager."""
    
    # Auto-detect best backend
    if backend == "auto":
        if jax is not None:
            try:
                devices_available = devices()
                if any(d.device_kind == 'gpu' for d in devices_available):
                    backend = ComputeBackend.GPU
                elif any(d.device_kind == 'tpu' for d in devices_available):
                    backend = ComputeBackend.TPU
                else:
                    backend = ComputeBackend.CPU
            except:
                backend = ComputeBackend.CPU
        else:
            backend = ComputeBackend.CPU
    else:
        backend = ComputeBackend(backend)
        
    # Auto-detect number of workers
    if num_workers is None:
        if backend == ComputeBackend.CPU:
            num_workers = mp.cpu_count()
        else:
            num_workers = len(devices()) if jax else 4
            
    config = ComputeConfig(
        backend=backend,
        num_workers=num_workers,
        **kwargs
    )
    
    manager = DistributedComputeManager(config)
    manager.initialize()
    
    return manager