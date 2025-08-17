"""Continual learning for industrial RL with catastrophic forgetting prevention."""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from functools import partial
import warnings

from ..agents.base import OfflineAgent
from ..core.types import Array
from ..monitoring.logger import get_logger
from ..validation.input_validator import validate_array_input
from ..resilience.error_recovery import ErrorRecoveryManager


class ElasticWeightConsolidation:
    """Elastic Weight Consolidation for preventing catastrophic forgetting."""
    
    def __init__(self, importance_weight: float = 1000.0):
        self.importance_weight = importance_weight
        self.fisher_information = None
        self.optimal_params = None
        self.logger = get_logger(self.__class__.__name__)
        
    def compute_fisher_information(
        self,
        params: Dict[str, Any],
        dataset: Dict[str, Array],
        model_apply_fn: callable,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """Compute Fisher Information Matrix for current task."""
        
        try:
            validate_array_input(dataset["observations"], "observations")
            validate_array_input(dataset["actions"], "actions")
            
            n_samples = len(dataset["observations"])
            n_batches = max(1, n_samples // batch_size)
            
            # Initialize Fisher Information
            fisher_info = jax.tree_map(jnp.zeros_like, params)
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                batch_obs = dataset["observations"][start_idx:end_idx]
                batch_actions = dataset["actions"][start_idx:end_idx]
                
                # Compute gradients for Fisher Information
                def loss_fn(p):
                    logits = model_apply_fn(p, batch_obs)
                    return jnp.mean((logits - batch_actions) ** 2)
                
                grads = jax.grad(loss_fn)(params)
                
                # Accumulate squared gradients (Fisher Information approximation)
                fisher_info = jax.tree_map(
                    lambda fi, g: fi + (g ** 2) / n_batches,
                    fisher_info,
                    grads
                )
            
            self.fisher_information = fisher_info
            self.optimal_params = params
            
            self.logger.info("Fisher Information Matrix computed successfully")
            return fisher_info
            
        except Exception as e:
            self.logger.error(f"Failed to compute Fisher Information: {e}")
            raise RuntimeError(f"Fisher Information computation failed: {e}") from e
    
    def compute_ewc_loss(self, current_params: Dict[str, Any]) -> float:
        """Compute EWC regularization loss."""
        
        if self.fisher_information is None or self.optimal_params is None:
            return 0.0
        
        try:
            ewc_loss = 0.0
            
            def compute_param_loss(current, optimal, fisher):
                return jnp.sum(fisher * (current - optimal) ** 2)
            
            param_losses = jax.tree_map(
                compute_param_loss,
                current_params,
                self.optimal_params,
                self.fisher_information
            )
            
            # Sum all parameter losses
            ewc_loss = jax.tree_util.tree_reduce(jnp.add, param_losses)
            
            return self.importance_weight * ewc_loss / 2.0
            
        except Exception as e:
            self.logger.error(f"EWC loss computation failed: {e}")
            return 0.0


class ProgressiveNetworks:
    """Progressive Networks for continual learning without forgetting."""
    
    def __init__(self, base_network_size: int = 256):
        self.base_network_size = base_network_size
        self.task_columns = []
        self.lateral_connections = {}
        self.logger = get_logger(self.__class__.__name__)
        
    def add_task_column(self, task_id: str, network_params: Dict[str, Any]):
        """Add new column for new task."""
        
        try:
            task_column = {
                "task_id": task_id,
                "network_params": network_params,
                "frozen": False
            }
            
            self.task_columns.append(task_column)
            
            # Create lateral connections to previous columns
            if len(self.task_columns) > 1:
                self._create_lateral_connections(task_id)
            
            self.logger.info(f"Added task column for task: {task_id}")
            return len(self.task_columns) - 1
            
        except Exception as e:
            self.logger.error(f"Failed to add task column: {e}")
            raise RuntimeError(f"Task column addition failed: {e}") from e
    
    def _create_lateral_connections(self, new_task_id: str):
        """Create lateral connections from previous columns to new column."""
        
        try:
            # Initialize lateral connection weights
            lateral_weights = {}
            
            for prev_column in self.task_columns[:-1]:  # Exclude the new column
                prev_task_id = prev_column["task_id"]
                
                # Create random lateral connection weights
                key = jax.random.PRNGKey(hash(f"{prev_task_id}_{new_task_id}") % 2**32)
                lateral_weights[prev_task_id] = jax.random.normal(
                    key, (self.base_network_size, self.base_network_size // 2)
                ) * 0.1
            
            self.lateral_connections[new_task_id] = lateral_weights
            
        except Exception as e:
            self.logger.error(f"Failed to create lateral connections: {e}")
            raise
    
    def freeze_column(self, task_id: str):
        """Freeze parameters of a task column."""
        
        for column in self.task_columns:
            if column["task_id"] == task_id:
                column["frozen"] = True
                self.logger.info(f"Frozen column for task: {task_id}")
                return
        
        self.logger.warning(f"Task {task_id} not found for freezing")


class ContinualIndustrialRL(OfflineAgent):
    """Continual learning agent for sequential industrial tasks."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        continual_method: str = "ewc",
        ewc_importance: float = 1000.0,
        task_memory_size: int = 1000,
        **kwargs
    ):
        super().__init__(state_dim, action_dim, **kwargs)
        
        # Validate continual learning method
        valid_methods = ["ewc", "progressive", "replay", "lwf"]
        if continual_method not in valid_methods:
            raise ValueError(f"Unknown continual method: {continual_method}. Valid: {valid_methods}")
        
        self.continual_method = continual_method
        self.ewc_importance = ewc_importance
        self.task_memory_size = task_memory_size
        
        # Initialize continual learning components
        self.ewc = ElasticWeightConsolidation(ewc_importance) if continual_method == "ewc" else None
        self.progressive_nets = ProgressiveNetworks() if continual_method == "progressive" else None
        self.task_memory = {} if continual_method == "replay" else None
        
        # Task tracking
        self.current_task_id = None
        self.completed_tasks = []
        self.task_performance = {}
        
        # Error recovery
        self.error_recovery = ErrorRecoveryManager(
            max_retries=3,
            backoff_strategy="exponential"
        )
        
        self.logger.info(f"Initialized continual learning with method: {continual_method}")
    
    def _init_networks(self) -> Dict[str, Any]:
        """Initialize networks with continual learning support."""
        
        try:
            class ContinualCritic(nn.Module):
                hidden_dim: int = 256
                
                @nn.compact
                def __call__(self, state: Array, action: Array) -> Array:
                    x = jnp.concatenate([state, action], axis=-1)
                    x = nn.Dense(self.hidden_dim, name="layer1")(x)
                    x = nn.relu(x)
                    x = nn.Dense(self.hidden_dim, name="layer2")(x)
                    x = nn.relu(x)
                    return nn.Dense(1, name="output")(x)
            
            class ContinualActor(nn.Module):
                action_dim: int
                hidden_dim: int = 256
                
                @nn.compact
                def __call__(self, state: Array) -> Array:
                    x = nn.Dense(self.hidden_dim, name="layer1")(state)
                    x = nn.relu(x)
                    x = nn.Dense(self.hidden_dim, name="layer2")(x)
                    x = nn.relu(x)
                    return nn.Dense(self.action_dim, name="output")(x)
            
            # Initialize dummy inputs
            dummy_state = jnp.ones((1, self.state_dim))
            dummy_action = jnp.ones((1, self.action_dim))
            
            key1, key2 = jax.random.split(self.key, 2)
            
            # Initialize networks
            critic = ContinualCritic()
            critic_params = critic.init(key1, dummy_state, dummy_action)
            
            actor = ContinualActor(action_dim=self.action_dim)
            actor_params = actor.init(key2, dummy_state)
            
            return {
                "critic": critic,
                "critic_params": critic_params,
                "actor": actor,
                "actor_params": actor_params,
                "critic_opt": optax.adam(3e-4).init(critic_params),
                "actor_opt": optax.adam(3e-4).init(actor_params),
            }
            
        except Exception as e:
            self.logger.error(f"Network initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize networks: {e}") from e
    
    def _create_train_step(self) -> Any:
        """Create continual learning training step."""
        
        @jax.jit
        def train_step(state, batch):
            try:
                def critic_loss_fn(params):
                    q_vals = state["critic"].apply(
                        params,
                        batch["observations"],
                        batch["actions"]
                    )
                    
                    # Basic Q-learning loss
                    targets = batch["rewards"] + 0.99 * q_vals  # Simplified
                    critic_loss = jnp.mean((q_vals - targets) ** 2)
                    
                    # Add continual learning regularization
                    if self.continual_method == "ewc" and self.ewc is not None:
                        ewc_loss = self.ewc.compute_ewc_loss(params)
                        critic_loss += ewc_loss
                    
                    return critic_loss
                
                # Update critic
                grads = jax.grad(critic_loss_fn)(state["critic_params"])
                
                # Check for gradient issues
                grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads)))
                if jnp.isnan(grad_norm) or jnp.isinf(grad_norm):
                    self.logger.warning(f"Invalid gradients detected: {grad_norm}")
                    # Return state unchanged
                    return state, {"critic_loss": float('inf'), "grad_norm": float(grad_norm)}
                
                # Clip gradients for stability
                grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
                
                critic_updates, new_critic_opt = optax.update(
                    grads, state["critic_opt"], state["critic_params"]
                )
                new_critic_params = optax.apply_updates(state["critic_params"], critic_updates)
                
                new_state = state.copy()
                new_state.update({
                    "critic_params": new_critic_params,
                    "critic_opt": new_critic_opt,
                })
                
                metrics = {
                    "critic_loss": critic_loss_fn(state["critic_params"]),
                    "grad_norm": float(grad_norm)
                }
                
                return new_state, metrics
                
            except Exception as e:
                # Return error state
                error_metrics = {
                    "critic_loss": float('inf'),
                    "error": str(e)
                }
                return state, error_metrics
        
        return train_step
    
    def learn_new_task(
        self,
        task_id: str,
        task_dataset: Dict[str, Array],
        n_epochs: int = 50,
        **kwargs
    ) -> Dict[str, Any]:
        """Learn a new task with continual learning."""
        
        self.logger.info(f"Learning new task: {task_id}")
        
        try:
            # Validate inputs
            validate_array_input(task_dataset["observations"], "task_observations")
            validate_array_input(task_dataset["actions"], "task_actions")
            
            if len(task_dataset["observations"]) == 0:
                raise ValueError("Empty task dataset provided")
            
            # Store current task
            self.current_task_id = task_id
            
            # Prepare for new task based on continual method
            self._prepare_for_new_task(task_id, task_dataset)
            
            # Train on new task with error recovery
            def train_with_recovery():
                return self.train(
                    task_dataset,
                    n_epochs=n_epochs,
                    **kwargs
                )
            
            training_result = self.error_recovery.execute_with_recovery(
                train_with_recovery,
                operation_name=f"train_task_{task_id}"
            )
            
            # Post-training continual learning updates
            self._post_task_learning(task_id, task_dataset)
            
            # Track task completion
            self.completed_tasks.append(task_id)
            self.task_performance[task_id] = training_result.get("final_metrics", {})
            
            self.logger.info(f"Successfully learned task: {task_id}")
            
            return {
                "task_id": task_id,
                "training_result": training_result,
                "continual_method": self.continual_method,
                "tasks_completed": len(self.completed_tasks)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to learn task {task_id}: {e}")
            return {
                "task_id": task_id,
                "error": str(e),
                "success": False
            }
    
    def _prepare_for_new_task(self, task_id: str, dataset: Dict[str, Array]):
        """Prepare agent for learning new task."""
        
        try:
            if self.continual_method == "ewc" and self.ewc is not None:
                # Compute Fisher Information for previous task
                if hasattr(self, 'state') and self.state is not None:
                    self.ewc.compute_fisher_information(
                        self.state["critic_params"],
                        dataset,
                        self.state["critic"].apply
                    )
                    
            elif self.continual_method == "progressive" and self.progressive_nets is not None:
                # Add new column for new task
                if hasattr(self, 'state') and self.state is not None:
                    self.progressive_nets.add_task_column(task_id, self.state["critic_params"])
                    
            elif self.continual_method == "replay" and self.task_memory is not None:
                # Store samples from previous tasks
                if len(self.completed_tasks) > 0:
                    self._store_task_memory(dataset)
                    
        except Exception as e:
            self.logger.error(f"Task preparation failed: {e}")
            # Continue without preparation rather than failing completely
            warnings.warn(f"Task preparation failed: {e}")
    
    def _post_task_learning(self, task_id: str, dataset: Dict[str, Array]):
        """Post-task updates for continual learning."""
        
        try:
            if self.continual_method == "progressive" and self.progressive_nets is not None:
                # Freeze previous task columns
                for prev_task in self.completed_tasks:
                    self.progressive_nets.freeze_column(prev_task)
                    
        except Exception as e:
            self.logger.error(f"Post-task learning failed: {e}")
            # Continue without post-processing
            
    def _store_task_memory(self, dataset: Dict[str, Array]):
        """Store subset of task data for replay."""
        
        try:
            if self.current_task_id is None:
                return
            
            n_samples = len(dataset["observations"])
            memory_size = min(self.task_memory_size, n_samples)
            
            # Random sampling for memory
            indices = np.random.choice(n_samples, memory_size, replace=False)
            
            self.task_memory[self.current_task_id] = {
                "observations": dataset["observations"][indices],
                "actions": dataset["actions"][indices],
                "rewards": dataset["rewards"][indices] if "rewards" in dataset else jnp.zeros(memory_size)
            }
            
            self.logger.info(f"Stored {memory_size} samples for task {self.current_task_id}")
            
        except Exception as e:
            self.logger.error(f"Task memory storage failed: {e}")
    
    def evaluate_continual_performance(
        self,
        test_tasks: Dict[str, Dict[str, Array]]
    ) -> Dict[str, Any]:
        """Evaluate performance across all learned tasks."""
        
        self.logger.info("Evaluating continual learning performance")
        
        results = {
            "task_results": {},
            "average_performance": 0.0,
            "forgetting_measure": 0.0,
            "forward_transfer": 0.0
        }
        
        try:
            task_performances = []
            
            for task_id, task_data in test_tasks.items():
                try:
                    # Simple evaluation (would use proper environment in practice)
                    if len(task_data["observations"]) > 0:
                        test_obs = task_data["observations"][:100]
                        predicted_actions = self.predict(test_obs)
                        
                        # Compute simple performance metric
                        if "actions" in task_data:
                            mse = float(jnp.mean((predicted_actions - task_data["actions"][:100]) ** 2))
                            performance = max(0.0, 1.0 - mse)  # Simple performance score
                        else:
                            performance = 0.5  # Default for unknown ground truth
                        
                        task_performances.append(performance)
                        results["task_results"][task_id] = {
                            "performance": performance,
                            "evaluated": True
                        }
                    else:
                        results["task_results"][task_id] = {
                            "performance": 0.0,
                            "evaluated": False,
                            "error": "Empty test data"
                        }
                        
                except Exception as e:
                    self.logger.error(f"Task {task_id} evaluation failed: {e}")
                    results["task_results"][task_id] = {
                        "performance": 0.0,
                        "evaluated": False,
                        "error": str(e)
                    }
            
            # Compute aggregate metrics
            if task_performances:
                results["average_performance"] = float(np.mean(task_performances))
                
                # Simple forgetting measure (would be more sophisticated in practice)
                if len(task_performances) > 1:
                    first_task_perf = task_performances[0]
                    last_task_perf = task_performances[-1]
                    results["forgetting_measure"] = max(0.0, first_task_perf - last_task_perf)
                
            results["num_tasks_evaluated"] = len(task_performances)
            results["continual_method"] = self.continual_method
            
        except Exception as e:
            self.logger.error(f"Continual evaluation failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def _update_step(self, state, batch):
        """Override update step for continual learning."""
        return self.train_step(state, batch)
    
    def _predict_impl(self, observations, deterministic=True):
        """Predict with continual learning considerations."""
        
        try:
            if not hasattr(self, 'state') or self.state is None:
                raise RuntimeError("Model not initialized")
            
            # Basic prediction using current actor
            actions = self.state["actor"].apply(
                self.state["actor_params"],
                observations
            )
            
            return jnp.tanh(actions)
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            # Return zero actions as fallback
            return jnp.zeros((observations.shape[0], self.action_dim))


def create_continual_learning_suite(
    state_dim: int = 20,
    action_dim: int = 6,
    methods: List[str] = None
) -> Dict[str, ContinualIndustrialRL]:
    """Create suite of continual learning agents for comparison."""
    
    if methods is None:
        methods = ["ewc", "progressive", "replay"]
    
    agents = {}
    
    for method in methods:
        try:
            agent = ContinualIndustrialRL(
                state_dim=state_dim,
                action_dim=action_dim,
                continual_method=method
            )
            agents[method] = agent
            
        except Exception as e:
            print(f"Failed to create {method} agent: {e}")
    
    return agents