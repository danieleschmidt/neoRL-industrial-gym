"""Performance optimization utilities for industrial RL."""

import functools
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import pmap, vmap

from ..monitoring.logger import get_logger
from .caching import cached_computation, get_global_cache


class PerformanceOptimizer:
    """Centralized performance optimization manager."""
    
    def __init__(self):
        self.logger = get_logger("performance_optimizer")
        self.optimizations_applied = []
        self.device_count = jax.device_count()
        self.local_device_count = jax.local_device_count()
        
        self.logger.info(f"Performance optimizer initialized - {self.device_count} devices available")
        
    def optimize_for_batch_processing(self, func: Callable, batch_size: int = 1024) -> Callable:
        """Optimize function for batch processing with automatic batching."""
        
        @functools.wraps(func)
        def batched_wrapper(*args, **kwargs):
            # Determine if inputs are already batched
            first_arg = args[0] if args else list(kwargs.values())[0]
            
            if isinstance(first_arg, (np.ndarray, jnp.ndarray)) and first_arg.ndim > 1:
                total_samples = first_arg.shape[0]
                
                if total_samples <= batch_size:
                    # Already optimal size
                    return func(*args, **kwargs)
                    
                # Process in batches
                results = []
                for i in range(0, total_samples, batch_size):
                    end_idx = min(i + batch_size, total_samples)
                    
                    # Create batch arguments
                    batch_args = []
                    for arg in args:
                        if isinstance(arg, (np.ndarray, jnp.ndarray)):
                            batch_args.append(arg[i:end_idx])
                        else:
                            batch_args.append(arg)
                            
                    batch_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, (np.ndarray, jnp.ndarray)):
                            batch_kwargs[k] = v[i:end_idx]
                        else:
                            batch_kwargs[k] = v
                            
                    batch_result = func(*batch_args, **batch_kwargs)
                    results.append(batch_result)
                    
                # Concatenate results
                if isinstance(results[0], (np.ndarray, jnp.ndarray)):
                    return jnp.concatenate(results, axis=0)
                elif isinstance(results[0], (list, tuple)):
                    return type(results[0])([jnp.concatenate([r[i] for r in results], axis=0) 
                                           for i in range(len(results[0]))])
                else:
                    return results
                    
            return func(*args, **kwargs)
            
        self.optimizations_applied.append(f"batch_processing_{func.__name__}")
        return batched_wrapper
        
    def parallelize_computation(self, func: Callable, axis: int = 0) -> Callable:
        """Parallelize computation across available devices using pmap."""
        
        if self.device_count == 1:
            self.logger.info("Single device detected, skipping parallelization")
            return func
            
        @functools.wraps(func)
        def parallel_wrapper(*args, **kwargs):
            # Check if inputs can be parallelized
            parallelizable_args = []
            non_parallel_args = []
            
            for i, arg in enumerate(args):
                if isinstance(arg, (np.ndarray, jnp.ndarray)) and arg.ndim > axis:
                    # Can be parallelized along specified axis
                    parallelizable_args.append((i, arg))
                else:
                    non_parallel_args.append((i, arg))
                    
            if not parallelizable_args:
                return func(*args, **kwargs)
                
            # Apply pmap if beneficial
            try:
                pmapped_func = pmap(func, axis_name='devices')
                
                # Prepare inputs for pmap
                parallel_args = list(args)
                for i, arg in parallelizable_args:
                    # Reshape for device distribution
                    if arg.shape[axis] >= self.device_count:
                        new_shape = (self.device_count, -1) + arg.shape[axis+1:]
                        parallel_args[i] = arg.reshape(new_shape)
                    else:
                        # Pad if necessary
                        pad_size = self.device_count - arg.shape[axis]
                        pad_shape = [(0, 0)] * arg.ndim
                        pad_shape[axis] = (0, pad_size)
                        parallel_args[i] = jnp.pad(arg, pad_shape)
                        parallel_args[i] = parallel_args[i].reshape(new_shape)
                        
                result = pmapped_func(*parallel_args, **kwargs)
                
                # Reshape result back
                if isinstance(result, (np.ndarray, jnp.ndarray)):
                    result = result.reshape(-1, *result.shape[2:])
                    
                return result
                
            except Exception as e:
                self.logger.warning(f"Parallelization failed, falling back to sequential: {e}")
                return func(*args, **kwargs)
                
        self.optimizations_applied.append(f"parallelization_{func.__name__}")
        return parallel_wrapper
        
    def vectorize_computation(self, func: Callable, signature: Optional[str] = None) -> Callable:
        """Vectorize computation using vmap."""
        
        @functools.wraps(func)
        def vectorized_wrapper(*args, **kwargs):
            try:
                if signature:
                    vmapped_func = vmap(func, signature=signature)
                else:
                    vmapped_func = vmap(func)
                    
                return vmapped_func(*args, **kwargs)
                
            except Exception as e:
                self.logger.warning(f"Vectorization failed, falling back to original: {e}")
                return func(*args, **kwargs)
                
        self.optimizations_applied.append(f"vectorization_{func.__name__}")
        return vectorized_wrapper
        
    def enable_jit_compilation(self, func: Callable, static_argnums: Optional[Tuple[int, ...]] = None) -> Callable:
        """Enable JIT compilation for function."""
        
        try:
            jitted_func = jax.jit(func, static_argnums=static_argnums)
            self.optimizations_applied.append(f"jit_{func.__name__}")
            return jitted_func
            
        except Exception as e:
            self.logger.warning(f"JIT compilation failed for {func.__name__}: {e}")
            return func
            
    def optimize_memory_usage(self, func: Callable) -> Callable:
        """Optimize memory usage with gradient checkpointing and efficient operations."""
        
        @functools.wraps(func)
        def memory_optimized_wrapper(*args, **kwargs):
            # Use gradient checkpointing for memory efficiency
            try:
                # Apply memory optimizations based on function characteristics
                if hasattr(func, '__name__') and 'loss' in func.__name__.lower():
                    # For loss functions, use checkpointing
                    checkpointed_func = jax.checkpoint(func)
                    return checkpointed_func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
                    
            except Exception as e:
                self.logger.warning(f"Memory optimization failed: {e}")
                return func(*args, **kwargs)
                
        self.optimizations_applied.append(f"memory_opt_{func.__name__}")
        return memory_optimized_wrapper
        
    def create_optimized_training_step(
        self,
        train_step_fn: Callable,
        enable_caching: bool = True,
        enable_jit: bool = True,
        enable_vectorization: bool = True,
        batch_size: int = 256,
    ) -> Callable:
        """Create fully optimized training step function."""
        
        optimized_fn = train_step_fn
        
        # Apply optimizations in order
        if enable_caching:
            cache = get_global_cache()
            optimized_fn = cached_computation(cache)(optimized_fn)
            
        if enable_vectorization:
            optimized_fn = self.vectorize_computation(optimized_fn)
            
        if enable_jit:
            optimized_fn = self.enable_jit_compilation(optimized_fn)
            
        optimized_fn = self.optimize_memory_usage(optimized_fn)
        optimized_fn = self.optimize_for_batch_processing(optimized_fn, batch_size)
        
        if self.device_count > 1:
            optimized_fn = self.parallelize_computation(optimized_fn)
            
        self.logger.info(f"Created optimized training step with {len(self.optimizations_applied)} optimizations")
        
        return optimized_fn
        
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get report of applied optimizations."""
        return {
            'optimizations_applied': self.optimizations_applied,
            'device_count': self.device_count,
            'local_device_count': self.local_device_count,
            'total_optimizations': len(self.optimizations_applied),
        }


def benchmark_function(func: Callable, *args, warmup_runs: int = 5, benchmark_runs: int = 10, **kwargs) -> Dict[str, float]:
    """Benchmark function performance.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        warmup_runs: Number of warmup runs
        benchmark_runs: Number of benchmark runs
        **kwargs: Function keyword arguments
        
    Returns:
        Performance statistics
    """
    logger = get_logger("benchmark")
    
    # Warmup
    logger.info(f"Running {warmup_runs} warmup iterations for {func.__name__}")
    for _ in range(warmup_runs):
        try:
            _ = func(*args, **kwargs)
            if hasattr(jax, 'device_count'):
                jax.devices()[0].synchronize_all_activity()  # Ensure completion
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
            
    # Benchmark
    logger.info(f"Running {benchmark_runs} benchmark iterations for {func.__name__}")
    times = []
    
    for i in range(benchmark_runs):
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            if hasattr(jax, 'device_count'):
                jax.devices()[0].synchronize_all_activity()
                
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            times.append(elapsed)
            
        except Exception as e:
            logger.error(f"Benchmark run {i+1} failed: {e}")
            
    if not times:
        return {'error': 'All benchmark runs failed'}
        
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'median_time': np.median(times),
        'total_runs': len(times),
        'throughput': 1.0 / np.mean(times) if times else 0,
    }


class DataloaderOptimizer:
    """Optimize data loading for training."""
    
    def __init__(self, num_workers: int = 4, prefetch_factor: int = 2):
        """Initialize dataloader optimizer.
        
        Args:
            num_workers: Number of worker threads
            prefetch_factor: Number of batches to prefetch
        """
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.logger = get_logger("dataloader_optimizer")
        
    def create_optimized_dataloader(
        self,
        dataset: Dict[str, np.ndarray],
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> Callable:
        """Create optimized dataloader with prefetching and caching.
        
        Args:
            dataset: Dataset dictionary
            batch_size: Batch size
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
            
        Returns:
            Optimized dataloader generator
        """
        def dataloader_generator():
            n_samples = len(dataset['observations'])
            indices = np.arange(n_samples)
            
            if shuffle:
                np.random.shuffle(indices)
                
            n_batches = n_samples // batch_size
            if not drop_last and n_samples % batch_size != 0:
                n_batches += 1
                
            # Prefetch batches
            def prepare_batch(batch_idx):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch = {}
                for key, values in dataset.items():
                    batch[key] = values[batch_indices]
                    
                return batch
                
            # Submit initial prefetch jobs
            futures = []
            for i in range(min(self.prefetch_factor, n_batches)):
                future = self.executor.submit(prepare_batch, i)
                futures.append(future)
                
            # Yield batches and maintain prefetch queue
            for batch_idx in range(n_batches):
                # Get next batch
                if futures:
                    current_future = futures.pop(0)
                    batch = current_future.result()
                else:
                    batch = prepare_batch(batch_idx)
                    
                # Submit next prefetch job
                next_batch_idx = batch_idx + len(futures) + 1
                if next_batch_idx < n_batches:
                    future = self.executor.submit(prepare_batch, next_batch_idx)
                    futures.append(future)
                    
                yield batch
                
        return dataloader_generator
        
    def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)


# Global optimizer instance
_performance_optimizer = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


def auto_optimize_function(
    func: Callable,
    enable_jit: bool = True,
    enable_vectorization: bool = True,
    enable_caching: bool = True,
    enable_memory_opt: bool = True,
) -> Callable:
    """Automatically apply all relevant optimizations to a function.
    
    Args:
        func: Function to optimize
        enable_jit: Enable JIT compilation
        enable_vectorization: Enable vectorization
        enable_caching: Enable result caching
        enable_memory_opt: Enable memory optimizations
        
    Returns:
        Optimized function
    """
    optimizer = get_performance_optimizer()
    
    optimized_func = func
    
    if enable_caching:
        optimized_func = cached_computation()(optimized_func)
        
    if enable_vectorization:
        optimized_func = optimizer.vectorize_computation(optimized_func)
        
    if enable_memory_opt:
        optimized_func = optimizer.optimize_memory_usage(optimized_func)
        
    if enable_jit:
        optimized_func = optimizer.enable_jit_compilation(optimized_func)
        
    return optimized_func