"""Conservative Q-Learning (CQL) agent for offline RL."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, Tuple
from functools import partial

from .base import OfflineAgent
from .networks import Actor, DoubleCritic, SafetyCritic, create_train_state, update_target_network
from ..core.types import Array, StateArray, ActionArray


class CQLAgent(OfflineAgent):
    """Conservative Q-Learning agent with safety constraints.
    
    Implements CQL (Kumar et al., 2020) with additional safety critic
    for industrial control applications.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        safety_critic: bool = True,
        constraint_threshold: float = 0.1,
        hidden_dims: Tuple[int, ...] = (256, 256),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        cql_alpha: float = 1.0,
        safety_penalty: float = 100.0,
        seed: int = 42,
    ):
        """Initialize CQL agent.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            safety_critic: Use safety critic
            constraint_threshold: Safety constraint threshold
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate
            gamma: Discount factor
            tau: Target network update rate
            alpha: Temperature parameter
            cql_alpha: CQL regularization strength
            safety_penalty: Penalty for safety violations
            seed: Random seed
        """
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            safety_critic=safety_critic,
            constraint_threshold=constraint_threshold,
            seed=seed,
        )
        
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.cql_alpha = cql_alpha
        self.safety_penalty = safety_penalty
        
        # Initialize networks
        self.state = self._init_networks()
        self.train_step = self._create_train_step()
    
    def _init_networks(self) -> Dict[str, Any]:
        """Initialize actor, critic, and optional safety critic."""
        key = self.key
        
        # Split keys for different networks
        actor_key, critic_key, safety_key = jax.random.split(key, 3)
        
        # Create dummy inputs
        dummy_obs = jnp.ones((1, self.state_dim))
        dummy_action = jnp.ones((1, self.action_dim))
        dummy_critic_input = jnp.concatenate([dummy_obs, dummy_action], axis=-1)
        
        # Initialize actor
        actor = Actor(
            action_dim=self.action_dim,
            hidden_dims=self.hidden_dims,
        )
        
        actor_state = create_train_state(
            network=actor,
            key=actor_key,
            dummy_input=dummy_obs,
            learning_rate=self.learning_rate,
        )
        
        # Initialize double critic
        critic = DoubleCritic(hidden_dims=self.hidden_dims)
        critic_state = create_train_state(
            network=critic,
            key=critic_key,
            dummy_input=(dummy_obs, dummy_action),
            learning_rate=self.learning_rate,
        )
        
        # Initialize safety critic if enabled
        safety_state = None
        if self.safety_critic:
            safety_critic = SafetyCritic(hidden_dims=self.hidden_dims)
            safety_state = create_train_state(
                network=safety_critic,
                key=safety_key,
                dummy_input=(dummy_obs, dummy_action),
                learning_rate=self.learning_rate,
            )
        
        return {
            "actor": actor_state,
            "critic": critic_state,
            "safety": safety_state,
        }
    
    def _create_train_step(self):
        """Create compiled training step function."""
        
        @jax.jit
        def train_step(state: Dict[str, Any], batch: Dict[str, Array]) -> Tuple[Dict[str, Any], Dict[str, float]]:
            
            observations = batch["observations"]
            actions = batch["actions"]
            rewards = batch["rewards"]
            next_observations = batch["next_observations"]
            dones = batch["terminals"]
            
            batch_size = observations.shape[0]
            key = jax.random.PRNGKey(self.training_step)
            
            # Update critic
            def critic_loss_fn(critic_params):
                # Compute target Q-values
                next_actions = state["actor"].apply_fn(
                    state["actor"].target_params, 
                    next_observations,
                    training=False
                )
                
                q1_next, q2_next = state["critic"].apply_fn(
                    state["critic"].target_params,
                    next_observations,
                    next_actions,
                    training=False
                )
                
                target_q = jnp.minimum(q1_next, q2_next)
                target_q = rewards + self.gamma * (1 - dones) * target_q
                target_q = jax.lax.stop_gradient(target_q)
                
                # Current Q-values
                q1, q2 = state["critic"].apply_fn(
                    critic_params, observations, actions, training=True
                )
                
                # TD loss
                td_loss1 = jnp.mean((q1 - target_q) ** 2)
                td_loss2 = jnp.mean((q2 - target_q) ** 2)
                td_loss = td_loss1 + td_loss2
                
                # CQL regularization
                # Sample random actions for CQL
                random_actions = jax.random.uniform(
                    key, (batch_size * 10, self.action_dim), minval=-1, maxval=1
                )
                
                # Tile observations to match random actions
                tiled_obs = jnp.tile(observations, (10, 1))
                
                # Q-values for random actions
                q1_rand, q2_rand = state["critic"].apply_fn(
                    critic_params, tiled_obs, random_actions, training=True
                )
                
                # Current policy actions
                current_actions = state["actor"].apply_fn(
                    state["actor"].params, observations, training=False
                )
                
                q1_curr, q2_curr = state["critic"].apply_fn(
                    critic_params, observations, current_actions, training=True
                )
                
                # CQL loss: encourage low Q-values for out-of-distribution actions
                cql_loss1 = (
                    jax.scipy.special.logsumexp(q1_rand.reshape(10, batch_size), axis=0).mean() -
                    q1_curr.mean()
                )
                cql_loss2 = (
                    jax.scipy.special.logsumexp(q2_rand.reshape(10, batch_size), axis=0).mean() -
                    q2_curr.mean()
                )
                
                cql_loss = cql_loss1 + cql_loss2
                
                total_loss = td_loss + self.cql_alpha * cql_loss
                
                return total_loss, {
                    "td_loss": td_loss,
                    "cql_loss": cql_loss,
                    "q1_mean": q1.mean(),
                    "q2_mean": q2.mean(),
                }
            
            # Update critic
            grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
            (critic_loss, critic_info), critic_grads = grad_fn(state["critic"].params)
            
            new_critic_state = state["critic"].apply_gradients(grads=critic_grads)
            
            # Update actor
            def actor_loss_fn(actor_params):
                actions_pred = state["actor"].apply_fn(
                    actor_params, observations, training=True
                )
                
                q1, q2 = state["critic"].apply_fn(
                    new_critic_state.params, observations, actions_pred, training=False
                )
                
                q_pred = jnp.minimum(q1, q2)
                
                # Safety penalty if safety critic is enabled
                safety_penalty = 0.0
                if state["safety"] is not None:
                    safety_pred = state["safety"].apply_fn(
                        state["safety"].params, observations, actions_pred, training=False
                    )
                    # Penalize actions with high safety violation probability
                    safety_penalty = self.safety_penalty * jnp.mean(
                        jnp.maximum(0, safety_pred - self.constraint_threshold)
                    )
                
                # Actor loss: maximize Q-values while respecting safety
                actor_loss = -jnp.mean(q_pred) + safety_penalty
                
                return actor_loss, {
                    "actor_loss": actor_loss,
                    "q_pred_mean": q_pred.mean(),
                    "safety_penalty": safety_penalty,
                }
            
            grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
            (actor_loss, actor_info), actor_grads = grad_fn(state["actor"].params)
            
            new_actor_state = state["actor"].apply_gradients(grads=actor_grads)
            
            # Update safety critic if enabled
            new_safety_state = state["safety"]
            safety_info = {}
            
            if state["safety"] is not None:
                # Create safety labels (simplified: based on reward)
                # In practice, this should be based on actual safety violations
                safety_labels = (rewards < -50).astype(jnp.float32)
                
                def safety_loss_fn(safety_params):
                    safety_pred = state["safety"].apply_fn(
                        safety_params, observations, actions, training=True
                    )
                    
                    # Binary cross-entropy loss
                    safety_loss = -jnp.mean(
                        safety_labels * jnp.log(safety_pred + 1e-8) +
                        (1 - safety_labels) * jnp.log(1 - safety_pred + 1e-8)
                    )
                    
                    return safety_loss, {
                        "safety_loss": safety_loss,
                        "safety_pred_mean": safety_pred.mean(),
                        "safety_accuracy": jnp.mean(
                            (safety_pred > 0.5) == safety_labels
                        ),
                    }
                
                grad_fn = jax.value_and_grad(safety_loss_fn, has_aux=True)
                (safety_loss, safety_info), safety_grads = grad_fn(state["safety"].params)
                
                new_safety_state = state["safety"].apply_gradients(grads=safety_grads)
            
            # Update target networks
            new_actor_state = update_target_network(new_actor_state, self.tau)
            new_critic_state = update_target_network(new_critic_state, self.tau)
            
            # Combine metrics
            metrics = {**critic_info, **actor_info, **safety_info}
            
            new_state = {
                "actor": new_actor_state,
                "critic": new_critic_state, 
                "safety": new_safety_state,
            }
            
            return new_state, metrics
        
        return train_step
    
    def _update_step(
        self, 
        state: Dict[str, Any], 
        batch: Dict[str, Array]
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Single training update step."""
        
        # Add next observations if not present
        if "next_observations" not in batch:
            # Assume sequential data, shift observations
            batch["next_observations"] = np.roll(batch["observations"], -1, axis=0)
            batch["next_observations"][-1] = batch["observations"][-1]  # Handle last element
        
        return self.train_step(state, batch)
    
    def predict(
        self, 
        observations: StateArray, 
        deterministic: bool = True
    ) -> ActionArray:
        """Predict actions for given observations."""
        if not self.is_trained:
            raise RuntimeError("Agent must be trained before prediction")
        
        # Ensure observations are in correct format
        if len(observations.shape) == 1:
            observations = observations[None]
        
        # Get actions from actor network
        actions = self.state["actor"].apply_fn(
            self.state["actor"].params,
            observations,
            training=False
        )
        
        # Add noise if not deterministic
        if not deterministic:
            key = jax.random.PRNGKey(self.training_step)
            noise = jax.random.normal(key, actions.shape) * 0.1
            actions = actions + noise
            actions = jnp.clip(actions, -1, 1)
        
        return np.array(actions)
    
    def predict_with_safety(
        self,
        observations: StateArray,
        safety_threshold: Optional[float] = None,
    ) -> Tuple[ActionArray, Array]:
        """Predict actions with safety violation probabilities.
        
        Args:
            observations: Input observations
            safety_threshold: Override default safety threshold
            
        Returns:
            Tuple of (actions, safety_probabilities)
        """
        if not self.is_trained or self.state["safety"] is None:
            raise RuntimeError("Safety critic must be trained")
        
        # Get actions
        actions = self.predict(observations, deterministic=True)
        
        # Get safety predictions
        safety_probs = self.state["safety"].apply_fn(
            self.state["safety"].params,
            observations,
            actions,
            training=False
        )
        
        threshold = safety_threshold or self.constraint_threshold
        
        # Filter actions based on safety threshold
        safe_mask = safety_probs < threshold
        
        # For unsafe actions, take more conservative approach
        actions = np.where(
            safe_mask[..., None],
            actions,
            actions * 0.5  # More conservative actions
        )
        
        return actions, np.array(safety_probs)