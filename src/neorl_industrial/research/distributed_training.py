"""Distributed training framework for large-scale industrial RL research."""

import jax
import jax.numpy as jnp
from jax import sharding
from jax.experimental import multihost_utils
import flax.linen as nn
import optax
from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass
import time
import threading
import queue
import concurrent.futures
from pathlib import Path

from ..agents.base import OfflineAgent
from ..core.types import Array
from ..monitoring.logger import get_logger
from ..validation.input_validator import validate_array_input
from ..resilience.error_recovery import ErrorRecoveryManager


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    num_devices: int
    batch_size_per_device: int
    gradient_accumulation_steps: int = 1
    data_parallel: bool = True
    model_parallel: bool = False
    pipeline_parallel: bool = False
    async_updates: bool = False
    communication_backend: str = "nccl"
    
    @property
    def global_batch_size(self) -> int:
        return self.num_devices * self.batch_size_per_device * self.gradient_accumulation_steps


class DataParallelTrainer:
    """Data parallel trainer for distributed RL training."""
    
    def __init__(
        self,
        agent_class: type,
        agent_kwargs: Dict[str, Any],
        config: DistributedConfig
    ):
        self.agent_class = agent_class
        self.agent_kwargs = agent_kwargs
        self.config = config
        
        self.logger = get_logger(self.__class__.__name__)
        self.error_recovery = ErrorRecoveryManager()
        
        # Initialize JAX distributed
        self._setup_distributed()
        
        # Create sharded agents
        self.agents = self._create_sharded_agents()
        
    def _setup_distributed(self):
        """Setup JAX distributed computing."""
        
        try:
            # Get available devices
            self.devices = jax.devices()
            actual_num_devices = len(self.devices)
            
            if actual_num_devices < self.config.num_devices:
                self.logger.warning(
                    f"Requested {self.config.num_devices} devices, "
                    f"but only {actual_num_devices} available"
                )
                self.config.num_devices = actual_num_devices
            
            # Create device mesh
            self.device_mesh = np.array(self.devices[:self.config.num_devices])
            
            # Setup sharding
            self.data_sharding = sharding.PositionalSharding(self.device_mesh)
            self.replicated_sharding = sharding.PositionalSharding(self.device_mesh).replicate()
            
            self.logger.info(f"Distributed setup complete: {self.config.num_devices} devices")
            
        except Exception as e:
            self.logger.error(f"Distributed setup failed: {e}")
            raise RuntimeError(f"Failed to setup distributed training: {e}") from e
    
    def _create_sharded_agents(self) -> List[OfflineAgent]:
        """Create agents for each device."""
        
        agents = []
        
        for device_id in range(self.config.num_devices):
            try:
                # Create agent for this device
                agent = self.agent_class(**self.agent_kwargs)
                
                # Initialize agent
                dummy_dataset = self._create_dummy_dataset()
                agent.state = agent._init_networks()
                agent.train_step = agent._create_train_step()
                
                agents.append(agent)
                
            except Exception as e:
                self.logger.error(f"Failed to create agent for device {device_id}: {e}")
                raise
        
        return agents
    
    def _create_dummy_dataset(self) -> Dict[str, Array]:
        """Create dummy dataset for initialization."""
        
        state_dim = self.agent_kwargs.get("state_dim", 20)
        action_dim = self.agent_kwargs.get("action_dim", 6)
        
        return {
            "observations": jnp.ones((32, state_dim)),
            "actions": jnp.ones((32, action_dim)),
            "rewards": jnp.ones((32,)),
            "terminals": jnp.zeros((32,), dtype=bool)
        }
    
    def distributed_train(
        self,
        dataset: Dict[str, Array],
        n_epochs: int = 100,
        checkpoint_freq: int = 10,
        evaluation_freq: int = 5,
        eval_env: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Train with data parallelism across devices."""
        
        self.logger.info(f"Starting distributed training: {n_epochs} epochs")
        
        try:
            # Validate dataset
            validate_array_input(dataset["observations"], "observations")
            validate_array_input(dataset["actions"], "actions")
            
            # Shard dataset across devices
            sharded_dataset = self._shard_dataset(dataset)
            
            # Training metrics
            training_metrics = []
            
            # Create distributed training step
            distributed_step = self._create_distributed_step()
            
            for epoch in range(n_epochs):
                epoch_start = time.time()
                
                # Create epoch batches
                epoch_batches = self._create_epoch_batches(sharded_dataset)
                
                epoch_losses = []
                
                for batch_idx, batches in enumerate(epoch_batches):
                    # Distributed training step
                    step_metrics = distributed_step(batches)
                    epoch_losses.append(step_metrics)
                
                # Aggregate metrics across devices
                avg_metrics = self._aggregate_metrics(epoch_losses)
                training_metrics.append(avg_metrics)
                
                epoch_time = time.time() - epoch_start
                
                # Log progress
                if (epoch + 1) % max(1, n_epochs // 10) == 0:
                    self.logger.info(
                        f"Epoch {epoch + 1}/{n_epochs}: {avg_metrics}, "
                        f"Time: {epoch_time:.2f}s"
                    )
                
                # Evaluation
                if eval_env is not None and (epoch + 1) % evaluation_freq == 0:
                    eval_metrics = self._distributed_evaluation(eval_env)
                    avg_metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
                
                # Checkpointing
                if (epoch + 1) % checkpoint_freq == 0:
                    self._save_distributed_checkpoint(epoch + 1)
            
            self.logger.info("Distributed training completed successfully")
            
            return {
                "training_metrics": training_metrics,
                "final_metrics": training_metrics[-1] if training_metrics else {},
                "distributed_config": self.config.__dict__,
                "num_devices_used": self.config.num_devices
            }
            
        except Exception as e:
            self.logger.error(f"Distributed training failed: {e}")
            raise RuntimeError(f"Distributed training failed: {e}") from e
    
    def _shard_dataset(self, dataset: Dict[str, Array]) -> Dict[str, Array]:
        """Shard dataset across devices."""
        
        try:
            sharded_data = {}
            
            for key, data in dataset.items():
                # Shard along batch dimension
                sharded_data[key] = jax.device_put(
                    data, self.data_sharding
                )
            
            return sharded_data
            
        except Exception as e:
            self.logger.error(f"Dataset sharding failed: {e}")
            raise
    
    def _create_epoch_batches(self, sharded_dataset: Dict[str, Array]) -> List[List[Dict[str, Array]]]:
        """Create batches for each device for the epoch."""
        
        try:
            n_samples = len(sharded_dataset["observations"])
            device_batch_size = self.config.batch_size_per_device
            
            # Create batches for each device
            epoch_batches = []
            
            n_batches = max(1, n_samples // (device_batch_size * self.config.num_devices))
            
            for batch_idx in range(n_batches):
                device_batches = []
                
                for device_id in range(self.config.num_devices):
                    start_idx = (batch_idx * self.config.num_devices + device_id) * device_batch_size
                    end_idx = start_idx + device_batch_size
                    
                    if start_idx < n_samples:
                        end_idx = min(end_idx, n_samples)
                        
                        device_batch = {
                            key: data[start_idx:end_idx]
                            for key, data in sharded_dataset.items()
                        }
                        device_batches.append(device_batch)
                    else:
                        # Pad with dummy data if needed
                        device_batch = {
                            key: data[:device_batch_size]
                            for key, data in sharded_dataset.items()
                        }
                        device_batches.append(device_batch)
                
                epoch_batches.append(device_batches)
            
            return epoch_batches
            
        except Exception as e:
            self.logger.error(f"Batch creation failed: {e}")
            raise
    
    def _create_distributed_step(self) -> Callable:
        """Create distributed training step function."""
        
        def distributed_step(device_batches: List[Dict[str, Array]]) -> Dict[str, float]:
            try:
                # Execute training step on each device
                step_metrics = []
                
                for device_id, batch in enumerate(device_batches):
                    if device_id < len(self.agents):
                        agent = self.agents[device_id]
                        
                        # Training step on device
                        agent.state, metrics = agent.train_step(agent.state, batch)
                        step_metrics.append(metrics)
                
                # Synchronize parameters across devices (simplified)
                if len(self.agents) > 1:
                    self._synchronize_parameters()
                
                return step_metrics
                
            except Exception as e:
                self.logger.error(f"Distributed step failed: {e}")
                return [{"error": str(e)}] * len(device_batches)
        
        return distributed_step
    
    def _synchronize_parameters(self):
        """Synchronize parameters across devices using all-reduce."""
        
        try:
            if len(self.agents) <= 1:
                return
            
            # Get parameters from all agents
            all_params = [agent.state for agent in self.agents]
            
            # Average parameters (simplified all-reduce)
            def average_params(params_list):
                if not params_list:
                    return {}
                
                averaged = {}
                for key in params_list[0].keys():
                    if key.endswith("_params"):
                        # Average parameter values
                        param_values = [p[key] for p in params_list]
                        averaged[key] = jax.tree_map(
                            lambda *args: jnp.mean(jnp.stack(args), axis=0),
                            *param_values
                        )
                    else:
                        # Keep other state as-is from first agent
                        averaged[key] = params_list[0][key]
                
                return averaged
            
            # Compute averaged parameters
            averaged_state = average_params(all_params)
            
            # Update all agents with averaged parameters
            for agent in self.agents:
                for key, value in averaged_state.items():
                    if key.endswith("_params"):
                        agent.state[key] = value
            
        except Exception as e:
            self.logger.error(f"Parameter synchronization failed: {e}")
            # Continue without synchronization
    
    def _aggregate_metrics(self, epoch_losses: List[List[Dict[str, Any]]]) -> Dict[str, float]:
        """Aggregate metrics across devices and batches."""
        
        try:
            all_metrics = []
            
            for batch_metrics in epoch_losses:
                for device_metrics in batch_metrics:
                    if "error" not in device_metrics:
                        all_metrics.append(device_metrics)
            
            if not all_metrics:
                return {"error": "No valid metrics"}
            
            # Average across all devices and batches
            aggregated = {}
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m and not jnp.isnan(m[key])]
                if values:
                    aggregated[key] = float(np.mean(values))
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Metric aggregation failed: {e}")
            return {"aggregation_error": str(e)}
    
    def _distributed_evaluation(self, eval_env: Any) -> Dict[str, float]:
        """Perform distributed evaluation."""
        
        try:
            # Use first agent for evaluation (they should be synchronized)
            if self.agents:
                agent = self.agents[0]
                return agent.evaluate(eval_env, n_episodes=10)
            else:
                return {"error": "No agents available for evaluation"}
                
        except Exception as e:
            self.logger.error(f"Distributed evaluation failed: {e}")
            return {"eval_error": str(e)}
    
    def _save_distributed_checkpoint(self, epoch: int):
        """Save distributed training checkpoint."""
        
        try:
            checkpoint_dir = Path(f"distributed_checkpoints/epoch_{epoch}")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save state from each device
            for device_id, agent in enumerate(self.agents):
                checkpoint_file = checkpoint_dir / f"device_{device_id}.pkl"
                agent.save(str(checkpoint_file))
            
            # Save distributed config
            config_file = checkpoint_dir / "distributed_config.json"
            import json
            with open(config_file, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
            
            self.logger.info(f"Distributed checkpoint saved: {checkpoint_dir}")
            
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}")


class AsyncParameterServer:
    """Asynchronous parameter server for distributed training."""
    
    def __init__(self, model_shape: Dict[str, Any]):
        self.model_shape = model_shape
        self.parameters = None
        self.version = 0
        self.lock = threading.Lock()
        self.update_queue = queue.Queue()
        
        self.logger = get_logger(self.__class__.__name__)
        
        # Start parameter server thread
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
    
    def _server_loop(self):
        """Main parameter server loop."""
        
        while True:
            try:
                # Get gradient update from queue
                update = self.update_queue.get(timeout=1.0)
                
                if update is None:  # Shutdown signal
                    break
                
                # Apply update
                self._apply_update(update)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Parameter server error: {e}")
    
    def _apply_update(self, update: Dict[str, Any]):
        """Apply gradient update to parameters."""
        
        try:
            with self.lock:
                if self.parameters is None:
                    # Initialize parameters
                    self.parameters = update["gradients"]
                else:
                    # Apply gradients (simplified SGD)
                    learning_rate = update.get("learning_rate", 0.001)
                    
                    self.parameters = jax.tree_map(
                        lambda p, g: p - learning_rate * g,
                        self.parameters,
                        update["gradients"]
                    )
                
                self.version += 1
            
        except Exception as e:
            self.logger.error(f"Parameter update failed: {e}")
    
    def get_parameters(self) -> Tuple[Dict[str, Any], int]:
        """Get current parameters and version."""
        
        with self.lock:
            return self.parameters, self.version
    
    def push_gradients(self, gradients: Dict[str, Any], learning_rate: float = 0.001):
        """Push gradients to parameter server."""
        
        update = {
            "gradients": gradients,
            "learning_rate": learning_rate
        }
        
        try:
            self.update_queue.put(update, timeout=1.0)
        except queue.Full:
            self.logger.warning("Parameter server queue full, dropping update")
    
    def shutdown(self):
        """Shutdown parameter server."""
        self.update_queue.put(None)
        self.server_thread.join(timeout=5.0)


class DistributedResearchFramework:
    """High-level framework for distributed RL research."""
    
    def __init__(
        self,
        num_workers: int = 4,
        devices_per_worker: int = 1,
        use_async_updates: bool = False
    ):
        self.num_workers = num_workers
        self.devices_per_worker = devices_per_worker
        self.use_async_updates = use_async_updates
        
        self.logger = get_logger(self.__class__.__name__)
        
        # Setup distributed configuration
        self.distributed_config = DistributedConfig(
            num_devices=num_workers * devices_per_worker,
            batch_size_per_device=64,
            async_updates=use_async_updates
        )
        
        self.parameter_server = None
        if use_async_updates:
            self.parameter_server = AsyncParameterServer({})
    
    def distributed_hyperparameter_search(
        self,
        agent_class: type,
        base_agent_kwargs: Dict[str, Any],
        hyperparameter_space: Dict[str, List[Any]],
        dataset: Dict[str, Array],
        n_trials_per_worker: int = 5
    ) -> Dict[str, Any]:
        """Perform distributed hyperparameter search."""
        
        self.logger.info(f"Starting distributed hyperparameter search with {self.num_workers} workers")
        
        try:
            # Generate hyperparameter combinations
            import itertools
            
            param_names = list(hyperparameter_space.keys())
            param_values = list(hyperparameter_space.values())
            
            # Limit combinations for practical execution
            max_combinations = self.num_workers * n_trials_per_worker
            all_combinations = list(itertools.product(*param_values))
            
            if len(all_combinations) > max_combinations:
                # Sample random combinations
                np.random.shuffle(all_combinations)
                combinations = all_combinations[:max_combinations]
            else:
                combinations = all_combinations
            
            # Distribute work across workers
            work_chunks = self._chunk_work(combinations, self.num_workers)
            
            # Run distributed search
            search_results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                
                for worker_id, chunk in enumerate(work_chunks):
                    future = executor.submit(
                        self._worker_hyperparameter_search,
                        worker_id,
                        agent_class,
                        base_agent_kwargs,
                        param_names,
                        chunk,
                        dataset
                    )
                    futures.append(future)
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        worker_results = future.result()
                        search_results.extend(worker_results)
                    except Exception as e:
                        self.logger.error(f"Worker failed: {e}")
            
            # Find best hyperparameters
            best_result = self._find_best_hyperparameters(search_results)
            
            return {
                "search_results": search_results,
                "best_hyperparameters": best_result,
                "total_trials": len(search_results),
                "num_workers": self.num_workers
            }
            
        except Exception as e:
            self.logger.error(f"Distributed hyperparameter search failed: {e}")
            raise RuntimeError(f"Distributed search failed: {e}") from e
    
    def _chunk_work(self, work_items: List[Any], num_chunks: int) -> List[List[Any]]:
        """Split work items into chunks for workers."""
        
        chunk_size = max(1, len(work_items) // num_chunks)
        chunks = []
        
        for i in range(0, len(work_items), chunk_size):
            chunk = work_items[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _worker_hyperparameter_search(
        self,
        worker_id: int,
        agent_class: type,
        base_agent_kwargs: Dict[str, Any],
        param_names: List[str],
        param_combinations: List[Tuple],
        dataset: Dict[str, Array]
    ) -> List[Dict[str, Any]]:
        """Worker function for hyperparameter search."""
        
        self.logger.info(f"Worker {worker_id} processing {len(param_combinations)} combinations")
        
        worker_results = []
        
        for combination in param_combinations:
            try:
                # Create agent kwargs with hyperparameters
                agent_kwargs = base_agent_kwargs.copy()
                
                for param_name, param_value in zip(param_names, combination):
                    agent_kwargs[param_name] = param_value
                
                # Create and train agent
                agent = agent_class(**agent_kwargs)
                
                # Quick training for hyperparameter evaluation
                training_result = agent.train(
                    dataset,
                    n_epochs=20,  # Reduced for speed
                    batch_size=64
                )
                
                # Extract performance metric
                final_metrics = training_result.get("final_metrics", {})
                performance_score = self._compute_performance_score(final_metrics)
                
                result = {
                    "worker_id": worker_id,
                    "hyperparameters": dict(zip(param_names, combination)),
                    "performance_score": performance_score,
                    "training_metrics": final_metrics,
                    "success": True
                }
                
                worker_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} trial failed: {e}")
                
                result = {
                    "worker_id": worker_id,
                    "hyperparameters": dict(zip(param_names, combination)),
                    "performance_score": float('-inf'),
                    "error": str(e),
                    "success": False
                }
                
                worker_results.append(result)
        
        return worker_results
    
    def _compute_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Compute performance score from metrics."""
        
        try:
            score = 0.0
            
            # Primary: minimize loss
            if "critic_loss" in metrics:
                loss = float(metrics["critic_loss"])
                if not (jnp.isnan(loss) or jnp.isinf(loss)):
                    score += max(0.0, 1.0 - loss / 10.0)
            
            # Stability bonus
            if not any(jnp.isnan(v) or jnp.isinf(v) for v in metrics.values() if isinstance(v, (int, float))):
                score += 0.5
            
            return float(score)
            
        except Exception as e:
            self.logger.error(f"Performance scoring failed: {e}")
            return 0.0
    
    def _find_best_hyperparameters(self, search_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find best hyperparameter configuration."""
        
        try:
            successful_results = [r for r in search_results if r.get("success", False)]
            
            if not successful_results:
                return None
            
            # Sort by performance score
            best_result = max(
                successful_results,
                key=lambda x: x.get("performance_score", float('-inf'))
            )
            
            return best_result
            
        except Exception as e:
            self.logger.error(f"Best hyperparameter search failed: {e}")
            return None
    
    def distributed_ensemble_training(
        self,
        agent_class: type,
        agent_kwargs: Dict[str, Any],
        dataset: Dict[str, Array],
        ensemble_size: int = 4,
        n_epochs: int = 100
    ) -> Dict[str, Any]:
        """Train ensemble of agents in parallel."""
        
        self.logger.info(f"Training ensemble of {ensemble_size} agents")
        
        try:
            # Create ensemble training tasks
            ensemble_tasks = []
            
            for i in range(ensemble_size):
                # Add randomization to each ensemble member
                task_kwargs = agent_kwargs.copy()
                task_kwargs["seed"] = task_kwargs.get("seed", 42) + i
                
                ensemble_tasks.append((i, task_kwargs))
            
            # Train ensemble members in parallel
            ensemble_results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = []
                
                for member_id, task_kwargs in ensemble_tasks:
                    future = executor.submit(
                        self._train_ensemble_member,
                        member_id,
                        agent_class,
                        task_kwargs,
                        dataset,
                        n_epochs
                    )
                    futures.append(future)
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        member_result = future.result()
                        ensemble_results.append(member_result)
                    except Exception as e:
                        self.logger.error(f"Ensemble member failed: {e}")
            
            # Aggregate ensemble results
            successful_members = [r for r in ensemble_results if r.get("success", False)]
            
            return {
                "ensemble_results": ensemble_results,
                "successful_members": len(successful_members),
                "total_members": ensemble_size,
                "ensemble_performance": self._aggregate_ensemble_performance(successful_members)
            }
            
        except Exception as e:
            self.logger.error(f"Distributed ensemble training failed: {e}")
            raise RuntimeError(f"Ensemble training failed: {e}") from e
    
    def _train_ensemble_member(
        self,
        member_id: int,
        agent_class: type,
        agent_kwargs: Dict[str, Any],
        dataset: Dict[str, Array],
        n_epochs: int
    ) -> Dict[str, Any]:
        """Train single ensemble member."""
        
        try:
            self.logger.info(f"Training ensemble member {member_id}")
            
            # Create agent
            agent = agent_class(**agent_kwargs)
            
            # Train agent
            training_result = agent.train(
                dataset,
                n_epochs=n_epochs,
                batch_size=128
            )
            
            return {
                "member_id": member_id,
                "training_result": training_result,
                "agent_config": agent_kwargs,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble member {member_id} training failed: {e}")
            return {
                "member_id": member_id,
                "error": str(e),
                "success": False
            }
    
    def _aggregate_ensemble_performance(self, successful_members: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate performance across ensemble members."""
        
        try:
            if not successful_members:
                return {"error": "No successful ensemble members"}
            
            # Extract final metrics from each member
            all_metrics = []
            for member in successful_members:
                final_metrics = member.get("training_result", {}).get("final_metrics", {})
                if final_metrics:
                    all_metrics.append(final_metrics)
            
            if not all_metrics:
                return {"error": "No metrics available"}
            
            # Compute ensemble statistics
            ensemble_stats = {}
            
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m and not jnp.isnan(m[key])]
                if values:
                    ensemble_stats[f"{key}_mean"] = float(np.mean(values))
                    ensemble_stats[f"{key}_std"] = float(np.std(values))
                    ensemble_stats[f"{key}_min"] = float(np.min(values))
                    ensemble_stats[f"{key}_max"] = float(np.max(values))
            
            return ensemble_stats
            
        except Exception as e:
            self.logger.error(f"Ensemble aggregation failed: {e}")
            return {"aggregation_error": str(e)}
    
    def cleanup(self):
        """Cleanup distributed resources."""
        
        if self.parameter_server is not None:
            self.parameter_server.shutdown()
        
        self.logger.info("Distributed framework cleanup complete")


def create_distributed_trainer(
    agent_class: type,
    agent_kwargs: Dict[str, Any],
    num_devices: int = None
) -> DataParallelTrainer:
    """Factory function for distributed trainer."""
    
    if num_devices is None:
        num_devices = len(jax.devices())
    
    config = DistributedConfig(
        num_devices=num_devices,
        batch_size_per_device=64,
        data_parallel=True
    )
    
    return DataParallelTrainer(
        agent_class=agent_class,
        agent_kwargs=agent_kwargs,
        config=config
    )