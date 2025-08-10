"""Distributed training utilities for scalable RL."""

import os
import json
import time
import threading
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from .monitoring.logger import get_logger
from .core.types import Array


@dataclass
class TrainingConfig:
    """Configuration for distributed training."""
    n_workers: int = 4
    batch_size_per_worker: int = 64
    sync_frequency: int = 10
    use_gpu: bool = False
    worker_type: str = "thread"  # "thread", "process", "ray"
    max_workers: Optional[int] = None
    timeout: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DistributedTrainingManager:
    """Manager for distributed RL training across multiple workers."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = get_logger(f"distributed_manager_{id(self)}")
        self.workers = []
        self.is_initialized = False
        self.shared_state = {}
        self._lock = threading.Lock()
        
        # Initialize distributed backend
        self._init_backend()
    
    def _init_backend(self):
        """Initialize the distributed backend."""
        if self.config.worker_type == "ray" and RAY_AVAILABLE:
            try:
                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True)
                self.logger.info("Ray backend initialized")
            except Exception as e:
                self.logger.warning(f"Ray initialization failed: {e}, falling back to threading")
                self.config.worker_type = "thread"
        elif self.config.worker_type == "ray" and not RAY_AVAILABLE:
            self.logger.warning("Ray not available, falling back to threading")
            self.config.worker_type = "thread"
        
        self.is_initialized = True
        self.logger.info(f"Distributed training initialized with {self.config.worker_type} backend")
    
    def create_worker_pool(self, worker_fn: Callable, worker_args: List[Any]):
        """Create a pool of workers for distributed training.
        
        Args:
            worker_fn: Function to run on each worker
            worker_args: List of arguments for each worker
        """
        max_workers = self.config.max_workers or self.config.n_workers
        
        if self.config.worker_type == "thread":
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        elif self.config.worker_type == "process":
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        elif self.config.worker_type == "ray":
            # Ray workers are created differently
            self.executor = None
        else:
            raise ValueError(f"Unknown worker type: {self.config.worker_type}")
        
        self.logger.info(f"Created worker pool with {max_workers} workers")
    
    def train_distributed(
        self,
        agent_class,
        agent_configs: List[Dict[str, Any]],
        dataset: Dict[str, Array],
        training_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute distributed training.
        
        Args:
            agent_class: Agent class to train
            agent_configs: List of agent configurations
            dataset: Training dataset
            training_params: Training parameters
            
        Returns:
            Dictionary with training results
        """
        if not self.is_initialized:
            raise RuntimeError("Distributed manager not initialized")
        
        self.logger.info(f"Starting distributed training with {len(agent_configs)} agents")
        
        start_time = time.time()
        
        # Split dataset for workers
        worker_datasets = self._split_dataset(dataset, len(agent_configs))
        
        # Create training tasks
        training_tasks = []
        for i, (config, worker_dataset) in enumerate(zip(agent_configs, worker_datasets)):
            task_config = {
                "worker_id": i,
                "agent_class": agent_class,
                "agent_config": config,
                "dataset": worker_dataset,
                "training_params": training_params,
            }
            training_tasks.append(task_config)
        
        # Execute training
        if self.config.worker_type == "ray":
            results = self._train_with_ray(training_tasks)
        else:
            results = self._train_with_executor(training_tasks)
        
        # Aggregate results
        training_time = time.time() - start_time
        final_results = self._aggregate_results(results, training_time)
        
        self.logger.info(f"Distributed training completed in {training_time:.2f}s")
        
        return final_results
    
    def _split_dataset(self, dataset: Dict[str, Array], n_splits: int) -> List[Dict[str, Array]]:
        """Split dataset into chunks for workers."""
        import numpy as np
        
        n_samples = len(dataset["observations"])
        chunk_size = n_samples // n_splits
        
        worker_datasets = []
        
        for i in range(n_splits):
            start_idx = i * chunk_size
            if i == n_splits - 1:  # Last worker gets remaining samples
                end_idx = n_samples
            else:
                end_idx = (i + 1) * chunk_size
            
            worker_dataset = {}
            for key, values in dataset.items():
                worker_dataset[key] = values[start_idx:end_idx]
            
            worker_datasets.append(worker_dataset)
            
            self.logger.debug(f"Worker {i} dataset: {len(worker_dataset['observations'])} samples")
        
        return worker_datasets
    
    def _train_with_executor(self, training_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Train using ThreadPoolExecutor or ProcessPoolExecutor."""
        futures = []
        
        for task in training_tasks:
            future = self.executor.submit(self._train_worker, task)
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures, timeout=self.config.timeout):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                self.logger.error(f"Worker training failed: {e}")
                # Add dummy result to maintain list length
                results.append({"success": False, "error": str(e)})
        
        return results
    
    def _train_with_ray(self, training_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Train using Ray distributed computing."""
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray not available")
        
        @ray.remote
        def train_worker_ray(task_config):
            return self._train_worker(task_config)
        
        # Submit tasks
        futures = [train_worker_ray.remote(task) for task in training_tasks]
        
        # Collect results
        results = ray.get(futures)
        
        return results
    
    def _train_worker(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train a single worker (to be called in separate thread/process)."""
        worker_id = task_config["worker_id"]
        agent_class = task_config["agent_class"]
        agent_config = task_config["agent_config"]
        dataset = task_config["dataset"]
        training_params = task_config["training_params"]
        
        try:
            # Create agent
            agent = agent_class(**agent_config)
            
            # Train agent
            start_time = time.time()
            training_results = agent.train(dataset, **training_params)
            training_time = time.time() - start_time
            
            return {
                "worker_id": worker_id,
                "success": True,
                "training_time": training_time,
                "results": training_results,
                "agent_state": getattr(agent, 'state', None),
            }
            
        except Exception as e:
            return {
                "worker_id": worker_id,
                "success": False,
                "error": str(e),
                "training_time": 0,
            }
    
    def _aggregate_results(
        self, 
        worker_results: List[Dict[str, Any]], 
        total_time: float
    ) -> Dict[str, Any]:
        """Aggregate results from all workers."""
        successful_workers = [r for r in worker_results if r.get("success", False)]
        failed_workers = [r for r in worker_results if not r.get("success", False)]
        
        total_training_time = sum(r.get("training_time", 0) for r in successful_workers)
        avg_training_time = total_training_time / len(successful_workers) if successful_workers else 0
        
        # Aggregate training metrics
        all_metrics = []
        for result in successful_workers:
            if "results" in result and "training_metrics" in result["results"]:
                all_metrics.extend(result["results"]["training_metrics"])
        
        # Calculate speedup
        estimated_sequential_time = sum(r.get("training_time", 0) for r in successful_workers)
        speedup = estimated_sequential_time / total_time if total_time > 0 else 1.0
        
        aggregated_results = {
            "total_workers": len(worker_results),
            "successful_workers": len(successful_workers),
            "failed_workers": len(failed_workers),
            "total_time": total_time,
            "avg_worker_time": avg_training_time,
            "speedup": speedup,
            "efficiency": speedup / self.config.n_workers,
            "aggregated_metrics": all_metrics,
            "worker_results": worker_results,
        }
        
        if failed_workers:
            aggregated_results["failures"] = [
                {"worker_id": r["worker_id"], "error": r.get("error", "Unknown error")}
                for r in failed_workers
            ]
        
        return aggregated_results
    
    def cleanup(self):
        """Clean up distributed resources."""
        if hasattr(self, 'executor') and self.executor is not None:
            self.executor.shutdown(wait=True)
        
        if self.config.worker_type == "ray" and RAY_AVAILABLE:
            try:
                ray.shutdown()
            except Exception as e:
                self.logger.warning(f"Ray shutdown failed: {e}")
        
        self.logger.info("Distributed training manager cleaned up")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class ParameterServer:
    """Simple parameter server for distributed training synchronization."""
    
    def __init__(self):
        self.parameters = {}
        self._lock = threading.Lock()
        self.version = 0
        self.logger = get_logger("parameter_server")
    
    def put(self, key: str, value: Any) -> None:
        """Store parameters."""
        with self._lock:
            self.parameters[key] = value
            self.version += 1
            self.logger.debug(f"Updated parameter {key} (version {self.version})")
    
    def get(self, key: str) -> Any:
        """Retrieve parameters."""
        with self._lock:
            return self.parameters.get(key)
    
    def get_all(self) -> Dict[str, Any]:
        """Get all parameters."""
        with self._lock:
            return self.parameters.copy()
    
    def get_version(self) -> int:
        """Get current version number."""
        with self._lock:
            return self.version
    
    def average_parameters(self, param_list: List[Dict[str, Any]], weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """Average parameters from multiple workers."""
        if not param_list:
            return {}
        
        if weights is None:
            weights = [1.0 / len(param_list)] * len(param_list)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        averaged_params = {}
        
        # Get all parameter keys
        all_keys = set()
        for params in param_list:
            all_keys.update(params.keys())
        
        for key in all_keys:
            values = []
            valid_weights = []
            
            for params, weight in zip(param_list, weights):
                if key in params:
                    values.append(params[key])
                    valid_weights.append(weight)
            
            if values:
                # Simple averaging for now (would need more sophisticated logic for JAX params)
                if isinstance(values[0], (int, float)):
                    averaged_params[key] = sum(v * w for v, w in zip(values, valid_weights))
                else:
                    # For complex parameters, just use the first valid value
                    averaged_params[key] = values[0]
        
        return averaged_params


def create_distributed_training_config(
    n_workers: int = 4,
    worker_type: str = "thread",
    **kwargs
) -> TrainingConfig:
    """Create a distributed training configuration.
    
    Args:
        n_workers: Number of workers
        worker_type: Type of workers ("thread", "process", "ray")
        **kwargs: Additional configuration parameters
        
    Returns:
        TrainingConfig instance
    """
    return TrainingConfig(
        n_workers=n_workers,
        worker_type=worker_type,
        **kwargs
    )


def train_agents_distributed(
    agent_class,
    agent_configs: List[Dict[str, Any]],
    dataset: Dict[str, Array],
    training_params: Dict[str, Any],
    distributed_config: Optional[TrainingConfig] = None,
) -> Dict[str, Any]:
    """Convenience function for distributed training.
    
    Args:
        agent_class: Agent class to train
        agent_configs: List of agent configurations
        dataset: Training dataset
        training_params: Training parameters
        distributed_config: Distributed training configuration
        
    Returns:
        Training results dictionary
    """
    if distributed_config is None:
        distributed_config = create_distributed_training_config(
            n_workers=len(agent_configs)
        )
    
    with DistributedTrainingManager(distributed_config) as manager:
        manager.create_worker_pool(None, agent_configs)
        results = manager.train_distributed(
            agent_class,
            agent_configs,
            dataset,
            training_params
        )
    
    return results
