"""Base offline RL agent with safety awareness."""

import abc
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

from ..core.types import Array, StateArray, ActionArray


class OfflineAgent(abc.ABC):
    """Base class for offline RL agents with safety constraints."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        safety_critic: bool = True,
        constraint_threshold: float = 0.1,
        seed: int = 42,
    ):
        """Initialize offline agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space  
            safety_critic: Whether to use safety critic
            constraint_threshold: Safety constraint threshold
            seed: Random seed for reproducibility
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.safety_critic = safety_critic
        self.constraint_threshold = constraint_threshold
        
        # Initialize JAX random key
        self.key = jax.random.PRNGKey(seed)
        
        # Training state
        self.training_step = 0
        self.is_trained = False
        
        # Metrics tracking
        self.training_metrics = []
        
    @abc.abstractmethod
    def _init_networks(self) -> Dict[str, Any]:
        """Initialize neural networks."""
        pass
    
    @abc.abstractmethod
    def _create_train_step(self) -> Any:
        """Create compiled training step function."""
        pass
    
    @abc.abstractmethod
    def _update_step(
        self, 
        state: Dict[str, Any], 
        batch: Dict[str, Array]
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Single training update step.""" 
        pass
    
    @abc.abstractmethod
    def predict(
        self, 
        observations: StateArray, 
        deterministic: bool = True
    ) -> ActionArray:
        """Predict actions for given observations."""
        pass
    
    def train(
        self,
        dataset: Dict[str, Array],
        n_epochs: int = 100,
        batch_size: int = 256,
        eval_env: Optional[Any] = None,
        eval_freq: int = 10,
        use_mlflow: bool = False,
    ) -> Dict[str, Any]:
        """Train agent on offline dataset.
        
        Args:
            dataset: Offline dataset
            n_epochs: Number of training epochs
            batch_size: Training batch size
            eval_env: Environment for evaluation
            eval_freq: Evaluation frequency (epochs)
            use_mlflow: Whether to log to MLflow
            
        Returns:
            Training metrics
        """
        # Initialize networks if not done
        if not hasattr(self, 'state'):
            self.state = self._init_networks()
            self.train_step = self._create_train_step()
        
        # Prepare dataset
        n_samples = len(dataset['observations'])
        indices = np.arange(n_samples)
        
        epoch_metrics = []
        
        for epoch in range(n_epochs):
            # Shuffle data
            np.random.shuffle(indices)
            
            # Mini-batch training
            epoch_losses = []
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                
                batch = {
                    key: val[batch_indices] for key, val in dataset.items()
                }
                
                # Update parameters
                self.state, step_metrics = self._update_step(self.state, batch)
                epoch_losses.append(step_metrics)
                
                self.training_step += 1
            
            # Aggregate epoch metrics
            avg_metrics = {}
            if epoch_losses:
                for key in epoch_losses[0].keys():
                    avg_metrics[key] = np.mean([m[key] for m in epoch_losses])
            
            epoch_metrics.append(avg_metrics)
            
            # Evaluation
            if eval_env is not None and (epoch + 1) % eval_freq == 0:
                eval_results = self.evaluate(eval_env, n_episodes=10)
                avg_metrics.update({f"eval_{k}": v for k, v in eval_results.items()})
                
            # MLflow logging
            if use_mlflow:
                try:
                    import mlflow
                    for key, value in avg_metrics.items():
                        mlflow.log_metric(key, value, step=epoch)
                except ImportError:
                    pass
        
        self.is_trained = True
        self.training_metrics = epoch_metrics
        
        return {
            "training_metrics": epoch_metrics,
            "final_metrics": epoch_metrics[-1] if epoch_metrics else {},
        }
    
    def evaluate(
        self, 
        env: Any, 
        n_episodes: int = 100,
        render: bool = False
    ) -> Dict[str, float]:
        """Evaluate agent performance.
        
        Args:
            env: Environment to evaluate on
            n_episodes: Number of evaluation episodes
            render: Whether to render episodes
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Agent must be trained before evaluation")
        
        episode_returns = []
        episode_lengths = []
        safety_violations = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0
            episode_violations = 0
            
            done = False
            while not done:
                # Get action
                action = self.predict(obs[None], deterministic=True)[0]
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_return += reward
                episode_length += 1
                
                # Track safety violations
                if 'safety_metrics' in info:
                    episode_violations += info['safety_metrics'].violation_count
                
                obs = next_obs
                
                if render:
                    try:
                        env.render()
                    except:
                        pass
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            safety_violations.append(episode_violations)
        
        return {
            "return_mean": np.mean(episode_returns),
            "return_std": np.std(episode_returns),
            "length_mean": np.mean(episode_lengths),
            "length_std": np.std(episode_lengths),
            "safety_violations": np.mean(safety_violations),
        }
    
    def save(self, path: str) -> None:
        """Save agent parameters."""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained agent")
        
        import pickle
        
        save_data = {
            "state": self.state,
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "safety_critic": self.safety_critic,
                "constraint_threshold": self.constraint_threshold,
            },
            "training_step": self.training_step,
            "training_metrics": self.training_metrics,
        }
        
        with open(path, "wb") as f:
            pickle.dump(save_data, f)
    
    def load(self, path: str) -> None:
        """Load agent parameters."""
        import pickle
        
        with open(path, "rb") as f:
            save_data = pickle.load(f)
        
        self.state = save_data["state"]
        self.training_step = save_data["training_step"] 
        self.training_metrics = save_data["training_metrics"]
        self.is_trained = True