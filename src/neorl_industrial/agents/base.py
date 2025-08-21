"""Base offline RL agent with safety awareness."""

import abc
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

from ..core.types import Array, StateArray, ActionArray
from ..monitoring.logger import get_logger
from ..monitoring.performance import get_performance_monitor
from ..security import get_security_manager, SecurityError
from ..optimization import get_performance_optimizer, DataloaderOptimizer


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
        
        Raises:
            ValueError: If dimensions are invalid
            TypeError: If parameters are wrong type
        """
        # Input validation
        if not isinstance(state_dim, int) or state_dim <= 0:
            raise ValueError(f"state_dim must be positive integer, got {state_dim}")
        if not isinstance(action_dim, int) or action_dim <= 0:
            raise ValueError(f"action_dim must be positive integer, got {action_dim}")
        if not isinstance(safety_critic, bool):
            raise TypeError(f"safety_critic must be bool, got {type(safety_critic)}")
        if not isinstance(constraint_threshold, (int, float)) or constraint_threshold <= 0:
            raise ValueError(f"constraint_threshold must be positive number, got {constraint_threshold}")
        if not isinstance(seed, int):
            raise TypeError(f"seed must be int, got {type(seed)}")
            
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
        
        # Logging, monitoring, security, and optimization
        agent_id = f"agent_{self.__class__.__name__}_{id(self)}"
        self.logger = get_logger(agent_id)
        self.performance_monitor = get_performance_monitor(agent_id)
        self.security_manager = get_security_manager()
        self.performance_optimizer = get_performance_optimizer()
        self.dataloader_optimizer = None  # Initialize on demand
        
        self.logger.info(
            f"Initialized agent: state_dim={state_dim}, "
            f"action_dim={action_dim}, safety_critic={safety_critic}"
        )
        self.performance_monitor.start_monitoring()
        
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
    
    def predict(
        self, 
        observations: StateArray, 
        deterministic: bool = True
    ) -> ActionArray:
        """Predict actions for given observations."""
        # Security validation of inputs
        try:
            # Validate shape more flexibly
            if len(observations.shape) == 1 and observations.shape[0] == self.state_dim:
                # Single observation
                expected_shape = (self.state_dim,)
            elif len(observations.shape) == 2 and observations.shape[1] == self.state_dim:
                # Batch of observations
                expected_shape = None  # Don't validate batch dimension
            else:
                raise SecurityError(
                    f"Invalid observation shape: {observations.shape}, "
                    f"expected (..., {self.state_dim})"
                )
                
            self.security_manager.validate_input_array(
                observations,
                expected_shape=expected_shape,
                expected_dtype=np.float32,
                min_value=-1e6,
                max_value=1e6,
                allow_nan=False,
                allow_inf=False,
            )
        except SecurityError as e:
            self.logger.error(f"Input validation failed: {e}")
            raise ValueError(f"Invalid observations: {e}") from e
            
        with self.performance_monitor.time_operation("inference"):
            return self._predict_impl(observations, deterministic)
            
    @abc.abstractmethod
    def _predict_impl(
        self, 
        observations: StateArray, 
        deterministic: bool = True
    ) -> ActionArray:
        """Implementation of prediction logic."""
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
            
        Raises:
            ValueError: If dataset is invalid
            RuntimeError: If training fails
        """
        # Validate dataset
        required_keys = ['observations', 'actions', 'rewards']
        for key in required_keys:
            if key not in dataset:
                raise ValueError(f"Dataset missing required key: {key}")
        
        # Check dataset consistency
        n_samples = len(dataset['observations'])
        for key in required_keys:
            if len(dataset[key]) != n_samples:
                raise ValueError(f"Inconsistent dataset sizes: {key} has {len(dataset[key])}, expected {n_samples}")
        
        # Validate shapes
        if dataset['observations'].shape[1] != self.state_dim:
            raise ValueError(
                f"Observation dim mismatch: got {dataset['observations'].shape[1]}, "
                f"expected {self.state_dim}"
            )
        if dataset['actions'].shape[1] != self.action_dim:
            raise ValueError(f"Action dim mismatch: got {dataset['actions'].shape[1]}, expected {self.action_dim}")
        
        # Validate and secure hyperparameters
        hyperparams = {
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "eval_freq": eval_freq,
        }
        
        try:
            validated_params = self.security_manager.validate_hyperparameters(hyperparams)
            n_epochs = validated_params["n_epochs"]
            batch_size = validated_params["batch_size"] 
            eval_freq = validated_params["eval_freq"]
        except SecurityError as e:
            self.logger.error(f"Hyperparameter validation failed: {e}")
            raise ValueError(f"Invalid hyperparameters: {e}") from e
        self.logger.info(f"Starting training: {n_epochs} epochs, batch_size={batch_size}")
        
        # Initialize networks if not done
        try:
            if not hasattr(self, 'state'):
                self.logger.info("Initializing neural networks...")
                self.state = self._init_networks()
                self.train_step = self._create_train_step()
                self.logger.info("Neural networks initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize networks: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize networks: {e}") from e
        
        # Prepare optimized dataloader
        n_samples = len(dataset['observations'])
        
        # Initialize dataloader optimizer
        if self.dataloader_optimizer is None:
            self.dataloader_optimizer = DataloaderOptimizer(num_workers=4, prefetch_factor=2)
            
        # Create optimized dataloader
        dataloader = self.dataloader_optimizer.create_optimized_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        epoch_metrics = []
        
        for epoch in range(n_epochs):
            # Use optimized dataloader
            epoch_losses = []
            
            for batch in dataloader():
                # Update parameters with performance monitoring
                with self.performance_monitor.time_operation("training_step"):
                    try:
                        self.state, step_metrics = self._update_step(self.state, batch)
                        epoch_losses.append(step_metrics)
                    except Exception as e:
                        self.performance_monitor.record_event("errors")
                        self.logger.error(f"Training step failed: {e}", exc_info=True)
                        raise
                
                self.training_step += 1
            
            # Aggregate epoch metrics
            avg_metrics = {}
            if epoch_losses:
                for key in epoch_losses[0].keys():
                    avg_metrics[key] = np.mean([m[key] for m in epoch_losses])
            
            epoch_metrics.append(avg_metrics)
            
            # Log progress periodically
            if (epoch + 1) % max(1, n_epochs // 10) == 0 or epoch == 0:
                self.logger.log_training_progress(
                    epoch=epoch + 1,
                    metrics=avg_metrics,
                    agent_id=self.__class__.__name__
                )
            
            # Evaluation (temporarily mark as trained for evaluation)
            if eval_env is not None and (epoch + 1) % eval_freq == 0:
                self.logger.info(f"Running evaluation at epoch {epoch + 1}...")
                
                # Temporarily mark as trained for evaluation
                was_trained = self.is_trained
                self.is_trained = True
                
                try:
                    eval_results = self.evaluate(eval_env, n_episodes=10)
                    avg_metrics.update({f"eval_{k}": v for k, v in eval_results.items()})
                    
                    # Log evaluation results
                    self.logger.log_evaluation_results(
                        results=eval_results,
                        agent_id=self.__class__.__name__,
                        env_id=getattr(eval_env.__class__, '__name__', 'Unknown')
                    )
                finally:
                    # Restore original training state
                    self.is_trained = was_trained
                
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
        
        self.logger.info(f"Training completed successfully after {n_epochs} epochs")
        final_metrics = epoch_metrics[-1] if epoch_metrics else {}
        if final_metrics:
            self.logger.info(f"Final training metrics: {final_metrics}")
        
        # Generate optimization report
        optimization_report = self.performance_optimizer.get_optimization_report()
        self.logger.info(f"Applied optimizations: {optimization_report['total_optimizations']}")
        
        # Cleanup resources
        if self.dataloader_optimizer:
            self.dataloader_optimizer.cleanup()
            
        return {
            "training_metrics": epoch_metrics,
            "final_metrics": epoch_metrics[-1] if epoch_metrics else {},
            "optimization_report": optimization_report,
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