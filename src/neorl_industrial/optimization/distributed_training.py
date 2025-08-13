"""Distributed training system with JAX for large-scale industrial RL."""

import jax
import jax.numpy as jnp
from jax import random, pmap, device_put
from jax.experimental import PartitionSpec as P
from jax.experimental.pjit import pjit
import optax
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import time
import threading
from collections import deque
import logging
from functools import partial

from ..core.types import Array, StateArray, ActionArray
from ..monitoring.dashboard import record_metric

@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    num_devices: int
    batch_size_per_device: int
    data_parallel: bool = True
    model_parallel: bool = False
    pipeline_parallel: bool = False
    gradient_accumulation_steps: int = 1
    use_mixed_precision: bool = True
    synchronous: bool = True

class DistributedDataLoader:
    """High-performance data loader for distributed training."""
    
    def __init__(
        self,
        dataset: Dict[str, Array],
        batch_size: int,
        num_devices: int,
        shuffle: bool = True,
        prefetch_batches: int = 4,
        drop_last: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_devices = num_devices
        self.shuffle = shuffle
        self.prefetch_batches = prefetch_batches
        self.drop_last = drop_last
        
        # Calculate effective batch size per device
        self.batch_size_per_device = batch_size // num_devices
        if batch_size % num_devices != 0 and not drop_last:
            raise ValueError(f"Batch size {batch_size} not divisible by num_devices {num_devices}")
        
        # Prepare data
        self.data_size = len(next(iter(dataset.values())))
        self.indices = np.arange(self.data_size)
        
        # Prefetch queue
        self.prefetch_queue = deque(maxlen=prefetch_batches)
        self.prefetch_thread = None
        self.stop_prefetch = threading.Event()
        
        self.logger = logging.getLogger("distributed_dataloader")
    
    def __iter__(self):
        """Iterate over batches."""
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Start prefetching
        self.stop_prefetch.clear()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self.prefetch_thread.start()
        
        batch_idx = 0
        while batch_idx < len(self.indices) - self.batch_size + 1:
            # Get batch from prefetch queue
            while len(self.prefetch_queue) == 0 and not self.stop_prefetch.is_set():
                time.sleep(0.001)  # Short sleep
            
            if len(self.prefetch_queue) > 0:
                yield self.prefetch_queue.popleft()
            
            batch_idx += self.batch_size
        
        # Stop prefetching
        self.stop_prefetch.set()
        if self.prefetch_thread:
            self.prefetch_thread.join()
    
    def _prefetch_worker(self):
        """Background worker for prefetching batches."""
        batch_idx = 0
        
        while batch_idx < len(self.indices) - self.batch_size + 1 and not self.stop_prefetch.is_set():
            if len(self.prefetch_queue) >= self.prefetch_batches:
                time.sleep(0.01)  # Queue is full, wait
                continue
            
            # Create batch
            batch_indices = self.indices[batch_idx:batch_idx + self.batch_size]
            batch = self._create_batch(batch_indices)
            
            self.prefetch_queue.append(batch)
            batch_idx += self.batch_size
    
    def _create_batch(self, indices: np.ndarray) -> Dict[str, Array]:
        """Create a batch from indices."""
        batch = {}
        
        for key, data in self.dataset.items():
            batch_data = data[indices]
            
            # Reshape for multi-device (batch_size_per_device, num_devices, ...)
            reshaped = batch_data.reshape(
                self.num_devices, 
                self.batch_size_per_device, 
                *batch_data.shape[1:]
            )
            
            # Convert to JAX array and distribute across devices
            batch[key] = device_put(reshaped, jax.devices())
        
        return batch

class DistributedTrainer:
    """Distributed trainer with advanced optimizations."""
    
    def __init__(
        self,
        config: DistributedConfig,
        model_init_fn: Callable,
        loss_fn: Callable,
        optimizer: optax.GradientTransformation,
    ):
        self.config = config
        self.model_init_fn = model_init_fn
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        
        # Setup devices
        self.devices = jax.devices()[:config.num_devices]
        self.logger = logging.getLogger("distributed_trainer")
        
        # Initialize distributed training functions
        self._setup_distributed_functions()
        
        # Performance tracking
        self.training_metrics = {
            "step_times": deque(maxlen=100),
            "throughput": deque(maxlen=100),
            "gradient_norms": deque(maxlen=100),
            "memory_usage": deque(maxlen=100),
        }
    
    def _setup_distributed_functions(self):
        """Setup distributed training functions."""
        
        if self.config.data_parallel:
            # Data parallel training step
            @pmap
            def parallel_train_step(state, batch):
                def loss_fn_wrapper(params):
                    predictions = state.apply_fn(params, batch["observations"], training=True)
                    loss, aux = self.loss_fn(predictions, batch)
                    return loss, aux
                
                grad_fn = jax.value_and_grad(loss_fn_wrapper, has_aux=True)
                (loss, aux), grads = grad_fn(state.params)
                
                # Apply gradient clipping
                grads = self._clip_gradients(grads)
                
                # Update parameters
                new_state = state.apply_gradients(grads=grads)
                
                return new_state, {"loss": loss, **aux}
            
            self.train_step_fn = parallel_train_step
        
        else:
            # Single device training (fallback)
            @jax.jit
            def single_train_step(state, batch):
                def loss_fn_wrapper(params):
                    predictions = state.apply_fn(params, batch["observations"], training=True)
                    loss, aux = self.loss_fn(predictions, batch)
                    return loss, aux
                
                grad_fn = jax.value_and_grad(loss_fn_wrapper, has_aux=True)
                (loss, aux), grads = grad_fn(state.params)
                
                grads = self._clip_gradients(grads)
                new_state = state.apply_gradients(grads=grads)
                
                return new_state, {"loss": loss, **aux}
            
            self.train_step_fn = single_train_step
    
    def _clip_gradients(self, grads, max_norm: float = 1.0):
        """Clip gradients by global norm."""
        grad_norm = jnp.sqrt(sum(
            jnp.sum(jnp.square(g)) for g in jax.tree_leaves(grads)
        ))
        
        clip_factor = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))
        clipped_grads = jax.tree_map(lambda g: g * clip_factor, grads)
        
        return clipped_grads
    
    def train_epoch(
        self,
        state,
        dataloader: DistributedDataLoader,
        metrics_callback: Optional[Callable] = None,
    ) -> Tuple[Any, Dict[str, float]]:
        """Train for one epoch."""
        
        epoch_metrics = []
        total_samples = 0
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            step_start_time = time.time()
            
            # Training step
            if self.config.data_parallel:
                # Replicate state across devices if needed
                if not hasattr(state, '_replicated'):
                    state = self._replicate_state(state)
                    state._replicated = True
                
                new_state, step_metrics = self.train_step_fn(state, batch)
                
                # Average metrics across devices
                step_metrics = self._average_metrics(step_metrics)
            else:
                # Single device training
                # Flatten batch for single device
                flat_batch = self._flatten_batch(batch)
                new_state, step_metrics = self.train_step_fn(state, flat_batch)
            
            # Update state
            state = new_state
            
            # Record timing
            step_time = time.time() - step_start_time
            batch_size = self.config.batch_size_per_device * self.config.num_devices
            throughput = batch_size / step_time
            
            self.training_metrics["step_times"].append(step_time)
            self.training_metrics["throughput"].append(throughput)
            
            # Record metrics
            record_metric("training_step_time", step_time, tags={"component": "training"})
            record_metric("training_throughput", throughput, tags={"component": "training"})
            record_metric("training_loss", step_metrics["loss"], tags={"component": "training"})
            
            epoch_metrics.append(step_metrics)
            total_samples += batch_size
            
            # Call metrics callback if provided
            if metrics_callback:
                metrics_callback(batch_idx, step_metrics, step_time, throughput)
            
            # Log progress periodically
            if batch_idx % 50 == 0:
                self.logger.info(
                    f"Batch {batch_idx}: loss={step_metrics['loss']:.4f}, "
                    f"throughput={throughput:.1f} samples/s"
                )
        
        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start_time
        avg_metrics = self._average_epoch_metrics(epoch_metrics)
        avg_metrics["epoch_time"] = epoch_time
        avg_metrics["samples_per_second"] = total_samples / epoch_time
        
        return state, avg_metrics
    
    def _replicate_state(self, state):
        """Replicate training state across devices."""
        return jax.device_put_replicated(state, self.devices)
    
    def _average_metrics(self, metrics: Dict[str, Array]) -> Dict[str, float]:
        """Average metrics across devices."""
        averaged = {}
        for key, values in metrics.items():
            if hasattr(values, 'mean'):
                averaged[key] = float(jnp.mean(values))
            else:
                averaged[key] = float(values)
        return averaged
    
    def _flatten_batch(self, batch: Dict[str, Array]) -> Dict[str, Array]:
        """Flatten batch for single device training."""
        flattened = {}
        for key, values in batch.items():
            # Reshape from (num_devices, batch_per_device, ...) to (total_batch, ...)
            flattened[key] = values.reshape(-1, *values.shape[2:])
        return flattened
    
    def _average_epoch_metrics(self, epoch_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """Average metrics over the epoch."""
        if not epoch_metrics:
            return {}
        
        averaged = {}
        for key in epoch_metrics[0].keys():
            values = [m[key] for m in epoch_metrics if key in m]
            if values:
                averaged[key] = float(np.mean(values))
        
        return averaged
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "avg_step_time": np.mean(self.training_metrics["step_times"]) if self.training_metrics["step_times"] else 0.0,
            "avg_throughput": np.mean(self.training_metrics["throughput"]) if self.training_metrics["throughput"] else 0.0,
            "max_throughput": np.max(self.training_metrics["throughput"]) if self.training_metrics["throughput"] else 0.0,
            "device_count": self.config.num_devices,
            "effective_batch_size": self.config.batch_size_per_device * self.config.num_devices,
        }

class AdaptiveBatchSizeOptimizer:
    """Automatically optimize batch size for maximum throughput."""
    
    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 8,
        max_batch_size: int = 512,
        target_memory_utilization: float = 0.85,
        measurement_window: int = 10,
    ):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_memory_utilization = target_memory_utilization
        self.measurement_window = measurement_window
        
        self.throughput_history = deque(maxlen=measurement_window)
        self.memory_history = deque(maxlen=measurement_window)
        self.batch_size_history = deque(maxlen=measurement_window)
        
        self.best_batch_size = initial_batch_size
        self.best_throughput = 0.0
        
        self.logger = logging.getLogger("adaptive_batch_optimizer")
    
    def update(self, throughput: float, memory_utilization: float) -> int:
        """Update batch size based on performance metrics."""
        
        self.throughput_history.append(throughput)
        self.memory_history.append(memory_utilization)
        self.batch_size_history.append(self.current_batch_size)
        
        # Need enough measurements to make decisions
        if len(self.throughput_history) < self.measurement_window:
            return self.current_batch_size
        
        # Calculate recent averages
        recent_throughput = np.mean(list(self.throughput_history)[-5:])
        recent_memory = np.mean(list(self.memory_history)[-5:])
        
        # Update best known configuration
        if recent_throughput > self.best_throughput:
            self.best_throughput = recent_throughput
            self.best_batch_size = self.current_batch_size
        
        # Decision logic
        if recent_memory > 0.95:  # Memory pressure
            # Reduce batch size
            new_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
            self.logger.info(f"Reducing batch size due to memory pressure: {self.current_batch_size} -> {new_batch_size}")
        
        elif recent_memory < self.target_memory_utilization and recent_throughput >= self.best_throughput * 0.95:
            # Increase batch size to utilize more memory
            new_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
            self.logger.info(f"Increasing batch size for better utilization: {self.current_batch_size} -> {new_batch_size}")
        
        elif recent_throughput < self.best_throughput * 0.9:
            # Performance degradation, revert to best known
            new_batch_size = self.best_batch_size
            self.logger.info(f"Reverting to best known batch size: {self.current_batch_size} -> {new_batch_size}")
        
        else:
            # Stay with current batch size
            new_batch_size = self.current_batch_size
        
        self.current_batch_size = new_batch_size
        return new_batch_size
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return {
            "current_batch_size": self.current_batch_size,
            "best_batch_size": self.best_batch_size,
            "best_throughput": self.best_throughput,
            "recent_throughput": np.mean(list(self.throughput_history)[-5:]) if len(self.throughput_history) >= 5 else 0.0,
            "recent_memory": np.mean(list(self.memory_history)[-5:]) if len(self.memory_history) >= 5 else 0.0,
            "measurements": len(self.throughput_history),
        }

def create_distributed_config(
    target_throughput: Optional[float] = None,
    available_memory_gb: Optional[float] = None,
) -> DistributedConfig:
    """Create optimized distributed configuration based on hardware."""
    
    num_devices = len(jax.devices())
    
    # Estimate optimal batch size based on memory
    if available_memory_gb:
        # Rough heuristic: 1GB can handle ~1000 samples for typical RL data
        estimated_batch_size = int(available_memory_gb * 1000 / num_devices)
        batch_size_per_device = max(8, min(128, estimated_batch_size))
    else:
        batch_size_per_device = 32  # Conservative default
    
    return DistributedConfig(
        num_devices=num_devices,
        batch_size_per_device=batch_size_per_device,
        data_parallel=num_devices > 1,
        use_mixed_precision=True,
        gradient_accumulation_steps=1,
        synchronous=True,
    )

def estimate_memory_usage(model_params, batch_size: int) -> float:
    """Estimate memory usage in GB."""
    
    # Count parameters
    param_count = sum(
        p.size for p in jax.tree_leaves(model_params)
    )
    
    # Estimate memory usage (parameters + gradients + activations)
    # Rough heuristic: 4 bytes per float32, 3x for params+grads+activations, batch factor
    memory_bytes = param_count * 4 * 3 * (1 + batch_size / 100)
    memory_gb = memory_bytes / (1024 ** 3)
    
    return memory_gb