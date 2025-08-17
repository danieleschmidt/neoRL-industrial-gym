"""Industrial meta-learning for rapid adaptation across environments."""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Dict, List, Any, Tuple
import numpy as np
from functools import partial

from ..agents.base import OfflineAgent
from ..core.types import Array


class MAMLIndustrialAgent(OfflineAgent):
    """Model-Agnostic Meta-Learning for industrial environments."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        **kwargs
    ):
        super().__init__(state_dim, action_dim, **kwargs)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        
    def _init_networks(self) -> Dict[str, Any]:
        """Initialize meta-learning networks."""
        
        class MetaPolicy(nn.Module):
            hidden_dim: int = 256
            
            @nn.compact
            def __call__(self, state: Array) -> Array:
                x = nn.Dense(self.hidden_dim)(state)
                x = nn.relu(x)
                x = nn.Dense(self.hidden_dim)(x)
                x = nn.relu(x)
                return nn.Dense(self.hidden_dim)(x)  # Feature representation
        
        class TaskHead(nn.Module):
            action_dim: int
            
            @nn.compact
            def __call__(self, features: Array) -> Array:
                return nn.Dense(self.action_dim)(features)
        
        dummy_state = jnp.ones((1, self.state_dim))
        
        key1, key2 = jax.random.split(self.key, 2)
        
        meta_policy = MetaPolicy()
        meta_params = meta_policy.init(key1, dummy_state)
        
        task_head = TaskHead(action_dim=self.action_dim)
        head_params = task_head.init(key2, jnp.ones((1, 256)))
        
        return {
            "meta_policy": meta_policy,
            "meta_params": meta_params,
            "task_head": task_head,
            "head_params": head_params,
            "meta_opt": optax.adam(self.outer_lr).init(meta_params),
            "head_opt": optax.adam(self.inner_lr).init(head_params),
        }
    
    def _create_train_step(self) -> Any:
        """Create MAML training step."""
        
        @jax.jit
        def inner_update(params, task_batch):
            """Single inner loop update for task adaptation."""
            
            def task_loss(head_params):
                features = self.state["meta_policy"].apply(
                    params["meta_params"],
                    task_batch["observations"]
                )
                actions = self.state["task_head"].apply(head_params, features)
                
                # Behavioral cloning loss for simplicity
                loss = jnp.mean((actions - task_batch["actions"]) ** 2)
                return loss
            
            grads = jax.grad(task_loss)(params["head_params"])
            updated_head_params = jax.tree_map(
                lambda p, g: p - self.inner_lr * g,
                params["head_params"],
                grads
            )
            
            return {**params, "head_params": updated_head_params}
        
        @jax.jit
        def meta_update(state, task_batches):
            """Meta-learning update across multiple tasks."""
            
            def meta_loss(meta_params):
                total_loss = 0.0
                
                for task_batch in task_batches:
                    # Create task-specific parameters
                    task_params = {
                        "meta_params": meta_params,
                        "head_params": state["head_params"]
                    }
                    
                    # Perform inner updates
                    for _ in range(self.inner_steps):
                        task_params = inner_update(task_params, task_batch)
                    
                    # Compute loss on adapted parameters
                    features = state["meta_policy"].apply(
                        task_params["meta_params"],
                        task_batch["observations"]
                    )
                    actions = state["task_head"].apply(
                        task_params["head_params"],
                        features
                    )
                    
                    task_loss = jnp.mean((actions - task_batch["actions"]) ** 2)
                    total_loss += task_loss
                
                return total_loss / len(task_batches)
            
            # Update meta parameters
            grads = jax.grad(meta_loss)(state["meta_params"])
            meta_updates, new_meta_opt = optax.update(
                grads, state["meta_opt"], state["meta_params"]
            )
            new_meta_params = optax.apply_updates(state["meta_params"], meta_updates)
            
            new_state = state.copy()
            new_state.update({
                "meta_params": new_meta_params,
                "meta_opt": new_meta_opt,
            })
            
            return new_state, {"meta_loss": meta_loss(state["meta_params"])}
        
        return meta_update
    
    def adapt_to_task(
        self,
        task_dataset: Dict[str, Array],
        adaptation_steps: int = None
    ) -> Dict[str, Any]:
        """Adapt meta-learned policy to new task."""
        
        if adaptation_steps is None:
            adaptation_steps = self.inner_steps
        
        # Initialize with current head parameters
        adapted_params = self.state["head_params"]
        
        # Create batches for adaptation
        n_samples = len(task_dataset["observations"])
        batch_size = min(32, n_samples)
        
        for step in range(adaptation_steps):
            # Sample batch
            indices = jax.random.randint(
                self.key, (batch_size,), 0, n_samples
            )
            
            batch = {
                "observations": task_dataset["observations"][indices],
                "actions": task_dataset["actions"][indices]
            }
            
            # Inner update
            def task_loss(params):
                features = self.state["meta_policy"].apply(
                    self.state["meta_params"],
                    batch["observations"]
                )
                actions = self.state["task_head"].apply(params, features)
                return jnp.mean((actions - batch["actions"]) ** 2)
            
            grads = jax.grad(task_loss)(adapted_params)
            adapted_params = jax.tree_map(
                lambda p, g: p - self.inner_lr * g,
                adapted_params,
                grads
            )
        
        return {
            "adapted_head_params": adapted_params,
            "adaptation_loss": task_loss(adapted_params)
        }
    
    def _update_step(self, state, batch):
        # For MAML, we expect batch to contain multiple task batches
        return self.train_step(state, [batch])  # Wrap single batch
    
    def _predict_impl(self, observations, deterministic=True):
        """Predict using meta-learned policy."""
        if not hasattr(self, 'state'):
            raise RuntimeError("Model not initialized")
        
        features = self.state["meta_policy"].apply(
            self.state["meta_params"],
            observations
        )
        actions = self.state["task_head"].apply(
            self.state["head_params"],
            features
        )
        
        return jnp.tanh(actions)


class IndustrialMetaLearning:
    """Meta-learning framework for industrial RL applications."""
    
    def __init__(
        self,
        base_environments: List[str],
        state_dim: int,
        action_dim: int
    ):
        self.base_environments = base_environments
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.meta_agent = None
        
    def setup_meta_agent(self, **kwargs) -> MAMLIndustrialAgent:
        """Initialize meta-learning agent."""
        self.meta_agent = MAMLIndustrialAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            **kwargs
        )
        return self.meta_agent
    
    def meta_train(
        self,
        task_datasets: Dict[str, Dict[str, Array]],
        n_meta_epochs: int = 100,
        tasks_per_batch: int = 4
    ) -> Dict[str, Any]:
        """Train meta-learning agent across multiple tasks."""
        
        if self.meta_agent is None:
            raise RuntimeError("Meta agent not initialized. Call setup_meta_agent() first.")
        
        # Initialize meta agent
        if not hasattr(self.meta_agent, 'state'):
            dummy_batch = {
                "observations": jnp.ones((32, self.state_dim)),
                "actions": jnp.ones((32, self.action_dim)),
                "rewards": jnp.ones((32,))
            }
            self.meta_agent.state = self.meta_agent._init_networks()
            self.meta_agent.train_step = self.meta_agent._create_train_step()
        
        task_names = list(task_datasets.keys())
        meta_metrics = []
        
        for epoch in range(n_meta_epochs):
            # Sample tasks for this meta-batch
            selected_tasks = np.random.choice(
                task_names, 
                size=min(tasks_per_batch, len(task_names)),
                replace=False
            )
            
            # Create task batches
            task_batches = []
            for task_name in selected_tasks:
                dataset = task_datasets[task_name]
                n_samples = len(dataset["observations"])
                
                # Sample batch from task
                batch_size = min(32, n_samples)
                indices = np.random.choice(n_samples, batch_size, replace=False)
                
                task_batch = {
                    "observations": dataset["observations"][indices],
                    "actions": dataset["actions"][indices],
                    "rewards": dataset["rewards"][indices] if "rewards" in dataset else jnp.zeros(batch_size)
                }
                task_batches.append(task_batch)
            
            # Meta update
            self.meta_agent.state, step_metrics = self.meta_agent.train_step(
                self.meta_agent.state,
                task_batches
            )
            
            meta_metrics.append(step_metrics)
            
            if (epoch + 1) % 10 == 0:
                print(f"Meta-epoch {epoch + 1}: {step_metrics}")
        
        return {
            "meta_training_metrics": meta_metrics,
            "final_meta_loss": meta_metrics[-1]["meta_loss"] if meta_metrics else 0.0
        }
    
    def few_shot_adaptation(
        self,
        target_task_dataset: Dict[str, Array],
        n_adaptation_samples: int = 50,
        adaptation_steps: int = 10
    ) -> Dict[str, Any]:
        """Perform few-shot adaptation to new task."""
        
        if self.meta_agent is None:
            raise RuntimeError("Meta agent not trained. Call meta_train() first.")
        
        # Subsample for few-shot setting
        n_samples = len(target_task_dataset["observations"])
        if n_samples > n_adaptation_samples:
            indices = np.random.choice(n_samples, n_adaptation_samples, replace=False)
            few_shot_dataset = {
                key: value[indices] for key, value in target_task_dataset.items()
            }
        else:
            few_shot_dataset = target_task_dataset
        
        # Adapt to target task
        adaptation_result = self.meta_agent.adapt_to_task(
            few_shot_dataset,
            adaptation_steps=adaptation_steps
        )
        
        return adaptation_result
    
    def evaluate_meta_learning(
        self,
        test_tasks: Dict[str, Dict[str, Array]],
        baseline_comparison: bool = True
    ) -> Dict[str, Any]:
        """Evaluate meta-learning performance across test tasks."""
        
        results = {}
        
        for task_name, task_dataset in test_tasks.items():
            # Meta-learning adaptation
            meta_result = self.few_shot_adaptation(task_dataset)
            
            # Evaluate adapted policy (simplified evaluation)
            test_obs = task_dataset["observations"][:100]  # Use first 100 samples
            
            # Get meta-learned predictions
            meta_features = self.meta_agent.state["meta_policy"].apply(
                self.meta_agent.state["meta_params"],
                test_obs
            )
            meta_actions = self.meta_agent.state["task_head"].apply(
                meta_result["adapted_head_params"],
                meta_features
            )
            
            # Simple MSE evaluation (would use proper environment evaluation in practice)
            if "actions" in task_dataset:
                meta_mse = float(jnp.mean((meta_actions - task_dataset["actions"][:100]) ** 2))
            else:
                meta_mse = 0.0
            
            results[task_name] = {
                "meta_mse": meta_mse,
                "adaptation_loss": float(meta_result["adaptation_loss"]),
                "adapted": True
            }
            
            # Baseline comparison (train from scratch)
            if baseline_comparison:
                try:
                    from ..agents.cql import CQLAgent
                    baseline_agent = CQLAgent(self.state_dim, self.action_dim)
                    
                    # Quick baseline training
                    baseline_metrics = baseline_agent.train(
                        task_dataset,
                        n_epochs=20,
                        batch_size=32
                    )
                    
                    baseline_predictions = baseline_agent.predict(test_obs)
                    baseline_mse = float(jnp.mean(
                        (baseline_predictions - task_dataset["actions"][:100]) ** 2
                    )) if "actions" in task_dataset else 0.0
                    
                    results[task_name]["baseline_mse"] = baseline_mse
                    results[task_name]["improvement"] = (baseline_mse - meta_mse) / baseline_mse if baseline_mse > 0 else 0.0
                    
                except Exception as e:
                    results[task_name]["baseline_error"] = str(e)
        
        # Aggregate results
        meta_mses = [r["meta_mse"] for r in results.values()]
        baseline_mses = [r.get("baseline_mse", 0) for r in results.values() if "baseline_mse" in r]
        improvements = [r.get("improvement", 0) for r in results.values() if "improvement" in r]
        
        summary = {
            "task_results": results,
            "summary": {
                "average_meta_mse": float(np.mean(meta_mses)),
                "average_baseline_mse": float(np.mean(baseline_mses)) if baseline_mses else 0.0,
                "average_improvement": float(np.mean(improvements)) if improvements else 0.0,
                "num_tasks": len(test_tasks)
            }
        }
        
        return summary
    
    def transfer_to_new_environment(
        self,
        target_env_name: str,
        target_dataset: Dict[str, Array],
        **adaptation_kwargs
    ) -> Dict[str, Any]:
        """Transfer meta-learned knowledge to completely new environment."""
        
        print(f"Transferring meta-knowledge to environment: {target_env_name}")
        
        # Perform adaptation
        adaptation_result = self.few_shot_adaptation(
            target_dataset,
            **adaptation_kwargs
        )
        
        # Store adapted parameters for the new environment
        transfer_result = {
            "target_environment": target_env_name,
            "adaptation_result": adaptation_result,
            "transfer_successful": True,
            "meta_parameters_transferred": True
        }
        
        return transfer_result


def create_industrial_meta_learner(
    environments: List[str] = None,
    state_dim: int = 20,
    action_dim: int = 6
) -> IndustrialMetaLearning:
    """Factory function for industrial meta-learning."""
    
    if environments is None:
        environments = [
            "ChemicalReactor-v0",
            "PowerGrid-v0", 
            "RobotAssembly-v0",
            "HVACControl-v0"
        ]
    
    meta_learner = IndustrialMetaLearning(
        base_environments=environments,
        state_dim=state_dim,
        action_dim=action_dim
    )
    
    return meta_learner