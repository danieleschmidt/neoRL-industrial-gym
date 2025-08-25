"""Autonomous self-improving agent with meta-learning capabilities."""

import abc
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import deque
import time

from .base import OfflineAgent
from ..core.types import Array, StateArray, ActionArray, MetricsDict
from ..monitoring.logger import get_logger
from ..research.meta_learning import MetaLearningModule
from ..optimization.adaptive_caching import AdaptiveCacheManager
from ..quality_gates import QualityGateExecutor, AdaptiveQualityGates


class AutonomousAgent(OfflineAgent):
    """Autonomous agent with self-improvement and adaptation capabilities."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        meta_learning_window: int = 100,
        adaptation_frequency: int = 50,
        quality_gate_threshold: float = 0.85,
        **kwargs
    ):
        """Initialize autonomous agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            meta_learning_window: Window size for meta-learning
            adaptation_frequency: Frequency of self-adaptation
            quality_gate_threshold: Quality gate threshold
            **kwargs: Additional arguments for base agent
        """
        super().__init__(state_dim, action_dim, **kwargs)
        
        self.meta_learning_window = meta_learning_window
        self.adaptation_frequency = adaptation_frequency
        self.quality_gate_threshold = quality_gate_threshold
        
        # Self-improvement components
        self.meta_learner = MetaLearningModule(
            input_dim=state_dim + action_dim,
            hidden_dim=256,
            adaptation_steps=5
        )
        
        self.cache_manager = AdaptiveCacheManager(
            max_size=10000,
            adaptation_rate=0.1
        )
        
        self.quality_executor = QualityGateExecutor(
            threshold=quality_gate_threshold
        )
        
        self.adaptive_gates = AdaptiveQualityGates(
            initial_thresholds={
                'performance': 0.8,
                'safety': 0.95,
                'efficiency': 0.7
            }
        )
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.adaptation_history = []
        self.last_adaptation_step = 0
        
        # Autonomous learning state
        self.autonomous_learning_enabled = True
        self.learning_rate_schedule = self._create_adaptive_schedule()
        
        self.logger.info("Initialized AutonomousAgent with self-improvement capabilities")
    
    def _create_adaptive_schedule(self) -> Dict[str, Any]:
        """Create adaptive learning rate schedule."""
        return {
            'initial_lr': 3e-4,
            'decay_factor': 0.95,
            'adaptation_boost': 1.2,
            'min_lr': 1e-6,
            'warmup_steps': 1000
        }
    
    def _init_networks(self) -> Dict[str, Any]:
        """Initialize neural networks with adaptive architecture."""
        # Enhanced network architecture with meta-learning
        key = jax.random.PRNGKey(42)
        
        # Main policy network
        policy_params = self._init_policy_network(key)
        
        # Value networks
        value_params = self._init_value_networks(key)
        
        # Meta-learning components
        meta_params = self.meta_learner.initialize_parameters(key)
        
        return {
            'policy': policy_params,
            'value': value_params,
            'meta': meta_params,
            'step': 0,
            'learning_stats': {
                'total_updates': 0,
                'adaptation_count': 0,
                'performance_trend': 0.0
            }
        }
    
    def _init_policy_network(self, key: jax.Array) -> Dict[str, Any]:
        """Initialize adaptive policy network."""
        # Dynamic architecture based on problem complexity
        layers = [
            ('dense', self.state_dim, 256),
            ('relu',),
            ('dense', 256, 256),
            ('relu',),
            ('dense', 256, 128),
            ('relu',),
            ('dense', 128, self.action_dim),
            ('tanh',)
        ]
        
        return self._create_network_params(key, layers)
    
    def _init_value_networks(self, key: jax.Array) -> Dict[str, Any]:
        """Initialize value networks with safety critic."""
        # Q-network
        q_layers = [
            ('dense', self.state_dim + self.action_dim, 256),
            ('relu',),
            ('dense', 256, 256),
            ('relu',),
            ('dense', 256, 1)
        ]
        
        # Safety critic
        safety_layers = [
            ('dense', self.state_dim + self.action_dim, 128),
            ('relu',),
            ('dense', 128, 128),
            ('relu',),
            ('dense', 128, 1),
            ('sigmoid',)
        ]
        
        key1, key2 = jax.random.split(key)
        
        return {
            'q_network': self._create_network_params(key1, q_layers),
            'safety_critic': self._create_network_params(key2, safety_layers) if self.safety_critic else None
        }
    
    def _create_network_params(self, key: jax.Array, layers: List[Tuple]) -> Dict[str, Any]:
        """Create network parameters from layer specification."""
        params = {}
        layer_key = key
        
        for i, layer_spec in enumerate(layers):
            if layer_spec[0] == 'dense':
                input_dim, output_dim = layer_spec[1], layer_spec[2]
                layer_key, sub_key = jax.random.split(layer_key)
                
                params[f'layer_{i}'] = {
                    'weights': jax.random.normal(sub_key, (input_dim, output_dim)) * 0.1,
                    'bias': jnp.zeros(output_dim)
                }
        
        return params
    
    def _create_train_step(self) -> Any:
        """Create compiled training step with meta-learning."""
        
        @jax.jit
        def train_step(state, batch):
            # Extract components
            policy_params = state['policy']
            value_params = state['value']
            meta_params = state['meta']
            
            # Compute losses
            policy_loss = self._compute_policy_loss(policy_params, batch)
            value_loss = self._compute_value_loss(value_params, batch)
            
            # Meta-learning adaptation
            adapted_params, meta_loss = self.meta_learner.adapt_parameters(
                meta_params, policy_params, batch
            )
            
            # Combined loss
            total_loss = policy_loss + value_loss + 0.1 * meta_loss
            
            # Update parameters
            new_state = {
                **state,
                'policy': adapted_params,
                'step': state['step'] + 1
            }
            
            metrics = {
                'policy_loss': policy_loss,
                'value_loss': value_loss,
                'meta_loss': meta_loss,
                'total_loss': total_loss
            }
            
            return new_state, metrics
        
        return train_step
    
    def _compute_policy_loss(self, params: Dict[str, Any], batch: Dict[str, Array]) -> float:
        """Compute policy loss with behavioral cloning."""
        # Simple behavioral cloning loss for demonstration
        states = batch['observations']
        actions = batch['actions']
        
        # Forward pass through policy network
        predicted_actions = self._forward_policy(params, states)
        
        # MSE loss
        loss = jnp.mean((predicted_actions - actions) ** 2)
        
        return loss
    
    def _compute_value_loss(self, params: Dict[str, Any], batch: Dict[str, Array]) -> float:
        """Compute value network loss."""
        # Simplified TD learning for demonstration
        states = batch['observations']
        actions = batch['actions']
        rewards = batch['rewards']
        
        # Q-value prediction
        state_actions = jnp.concatenate([states, actions], axis=-1)
        q_values = self._forward_q_network(params['q_network'], state_actions)
        
        # Target values (simplified)
        targets = rewards.reshape(-1, 1)
        
        # MSE loss
        loss = jnp.mean((q_values - targets) ** 2)
        
        return loss
    
    def _forward_policy(self, params: Dict[str, Any], states: Array) -> Array:
        """Forward pass through policy network."""
        x = states
        
        for layer_name, layer_params in params.items():
            if 'layer_' in layer_name:
                x = jnp.dot(x, layer_params['weights']) + layer_params['bias']
                x = jnp.tanh(x)  # Simplified activation
        
        return x
    
    def _forward_q_network(self, params: Dict[str, Any], state_actions: Array) -> Array:
        """Forward pass through Q-network."""
        x = state_actions
        
        for layer_name, layer_params in params.items():
            if 'layer_' in layer_name:
                x = jnp.dot(x, layer_params['weights']) + layer_params['bias']
                x = jax.nn.relu(x)  # ReLU for Q-network
        
        return x
    
    def _update_step(
        self, 
        state: Dict[str, Any], 
        batch: Dict[str, Array]
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Enhanced update step with autonomous adaptation."""
        
        # Standard training step
        new_state, metrics = self.train_step(state, batch)
        
        # Track performance
        self.performance_history.append(metrics['total_loss'])
        
        # Autonomous adaptation check
        if (
            self.autonomous_learning_enabled and
            (self.training_step - self.last_adaptation_step) >= self.adaptation_frequency
        ):
            self._perform_autonomous_adaptation(new_state, metrics)
            self.last_adaptation_step = self.training_step
        
        # Update quality gates
        gate_results = self.adaptive_gates.evaluate_quality({
            'performance': 1.0 / (1.0 + metrics['total_loss']),
            'safety': 0.9,  # Placeholder
            'efficiency': self._compute_efficiency_score()
        })
        
        metrics.update(gate_results)
        
        # Cache management
        self.cache_manager.update_cache(
            key=f"batch_{self.training_step}",
            data=batch,
            performance_score=1.0 / (1.0 + metrics['total_loss'])
        )
        
        return new_state, metrics
    
    def _perform_autonomous_adaptation(
        self, 
        state: Dict[str, Any], 
        metrics: Dict[str, float]
    ) -> None:
        """Perform autonomous adaptation based on performance."""
        
        # Analyze performance trend
        if len(self.performance_history) >= 20:
            recent_performance = list(self.performance_history)[-20:]
            trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
            
            adaptation_applied = False
            
            # If performance is degrading, adapt learning rate
            if trend > 0.01:  # Loss is increasing
                self._adapt_learning_rate(factor=0.8)
                adaptation_applied = True
                
            # If performance is stagnating, try architecture adaptation
            elif abs(trend) < 0.001:  # Loss is flat
                self._adapt_network_architecture(state)
                adaptation_applied = True
            
            if adaptation_applied:
                self.adaptation_history.append({
                    'step': self.training_step,
                    'trend': trend,
                    'action': 'lr_adapt' if trend > 0.01 else 'arch_adapt',
                    'performance': metrics['total_loss']
                })
                
                self.logger.info(f"Autonomous adaptation applied at step {self.training_step}")
    
    def _adapt_learning_rate(self, factor: float = 0.8) -> None:
        """Adapt learning rate based on performance."""
        self.learning_rate_schedule['initial_lr'] *= factor
        self.learning_rate_schedule['initial_lr'] = max(
            self.learning_rate_schedule['initial_lr'],
            self.learning_rate_schedule['min_lr']
        )
        
        self.logger.info(f"Adapted learning rate to {self.learning_rate_schedule['initial_lr']}")
    
    def _adapt_network_architecture(self, state: Dict[str, Any]) -> None:
        """Adapt network architecture dynamically."""
        # Simple architecture adaptation - could be more sophisticated
        self.logger.info("Triggering network architecture adaptation")
        
        # For now, just log the adaptation - in practice, this would
        # modify network structure based on performance analysis
        state['learning_stats']['adaptation_count'] += 1
    
    def _compute_efficiency_score(self) -> float:
        """Compute efficiency score based on cache hits and computation time."""
        cache_stats = self.cache_manager.get_statistics()
        hit_rate = cache_stats.get('hit_rate', 0.0)
        
        # Simple efficiency metric
        return min(1.0, hit_rate + 0.5)
    
    def _predict_impl(
        self, 
        observations: StateArray, 
        deterministic: bool = True
    ) -> ActionArray:
        """Enhanced prediction with caching and adaptation."""
        
        # Check cache first
        cache_key = hash(observations.tobytes())
        cached_result = self.cache_manager.get_cached_result(cache_key)
        
        if cached_result is not None:
            return cached_result
        
        # Forward pass through policy network
        actions = self._forward_policy(self.state['policy'], observations)
        
        # Add exploration noise if not deterministic
        if not deterministic:
            key, self.key = jax.random.split(self.key)
            noise = jax.random.normal(key, actions.shape) * 0.1
            actions = actions + noise
        
        # Clip actions to valid range
        actions = jnp.clip(actions, -1.0, 1.0)
        
        # Cache result
        self.cache_manager.cache_result(cache_key, actions, performance_score=1.0)
        
        return actions
    
    def get_autonomous_status(self) -> Dict[str, Any]:
        """Get status of autonomous learning components."""
        cache_stats = self.cache_manager.get_statistics()
        gate_stats = self.adaptive_gates.get_adaptation_history()
        
        return {
            'autonomous_learning_enabled': self.autonomous_learning_enabled,
            'adaptations_performed': len(self.adaptation_history),
            'last_adaptation_step': self.last_adaptation_step,
            'cache_hit_rate': cache_stats.get('hit_rate', 0.0),
            'quality_gate_adaptations': len(gate_stats),
            'performance_trend': self._compute_performance_trend(),
            'learning_rate': self.learning_rate_schedule['initial_lr']
        }
    
    def _compute_performance_trend(self) -> float:
        """Compute recent performance trend."""
        if len(self.performance_history) < 10:
            return 0.0
        
        recent = list(self.performance_history)[-10:]
        return float(np.polyfit(range(len(recent)), recent, 1)[0])
    
    def enable_autonomous_learning(self) -> None:
        """Enable autonomous learning and adaptation."""
        self.autonomous_learning_enabled = True
        self.logger.info("Autonomous learning enabled")
    
    def disable_autonomous_learning(self) -> None:
        """Disable autonomous learning and adaptation."""
        self.autonomous_learning_enabled = False
        self.logger.info("Autonomous learning disabled")
    
    def get_adaptation_report(self) -> Dict[str, Any]:
        """Generate comprehensive adaptation report."""
        return {
            'total_adaptations': len(self.adaptation_history),
            'adaptation_history': self.adaptation_history[-10:],  # Last 10
            'cache_performance': self.cache_manager.get_statistics(),
            'quality_gates': self.adaptive_gates.get_current_thresholds(),
            'meta_learning_stats': self.meta_learner.get_adaptation_stats() if hasattr(self.meta_learner, 'get_adaptation_stats') else {},
            'autonomous_status': self.get_autonomous_status()
        }