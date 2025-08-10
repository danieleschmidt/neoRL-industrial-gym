"""Neural network definitions for industrial RL agents."""

import jax
import jax.numpy as jnp
import optax
from typing import Callable, Sequence

import flax.linen as nn
from flax.training import train_state


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""
    
    features: Sequence[int]
    activation: Callable = nn.relu
    dropout_rate: float = 0.0
    use_layer_norm: bool = False
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        for i, feat in enumerate(self.features[:-1]):
            x = nn.Dense(feat)(x)
            
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
                
            x = self.activation(x)
            
            if self.dropout_rate > 0:
                x = nn.Dropout(self.dropout_rate, deterministic=not training)(x)
        
        # Output layer
        x = nn.Dense(self.features[-1])(x)
        return x


class Critic(nn.Module):
    """Critic network for value function approximation."""
    
    hidden_dims: Sequence[int] = (256, 256)
    activation: Callable = nn.relu
    use_layer_norm: bool = False
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, observations, actions, training: bool = False):
        # Concatenate observations and actions
        x = jnp.concatenate([observations, actions], axis=-1)
        
        # Pass through MLP
        x = MLP(
            features=(*self.hidden_dims, 1),
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
        )(x, training=training)
        
        return x.squeeze(-1)


class DoubleCritic(nn.Module):
    """Double critic for reduced overestimation bias."""
    
    hidden_dims: Sequence[int] = (256, 256)
    activation: Callable = nn.relu
    use_layer_norm: bool = False
    dropout_rate: float = 0.0
    
    def setup(self):
        self.critic1 = Critic(
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
        )
        self.critic2 = Critic(
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
        )
    
    def __call__(self, observations, actions, training: bool = False):
        q1 = self.critic1(observations, actions, training=training)
        q2 = self.critic2(observations, actions, training=training)
        return q1, q2


class Actor(nn.Module):
    """Actor network for policy approximation."""
    
    action_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    activation: Callable = nn.relu
    use_layer_norm: bool = False
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, observations, training: bool = False):
        x = MLP(
            features=(*self.hidden_dims, self.action_dim),
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
        )(observations, training=training)
        
        # Tanh activation for bounded actions
        return nn.tanh(x)


class SafetyCritic(nn.Module):
    """Safety critic for constraint violation prediction."""
    
    hidden_dims: Sequence[int] = (256, 256)
    activation: Callable = nn.relu
    use_layer_norm: bool = False
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, observations, actions, training: bool = False):
        # Concatenate observations and actions
        x = jnp.concatenate([observations, actions], axis=-1)
        
        # Pass through MLP
        x = MLP(
            features=(*self.hidden_dims, 1),
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
        )(x, training=training)
        
        # Sigmoid for probability output
        return nn.sigmoid(x.squeeze(-1))


class ValueFunction(nn.Module):
    """Value function network for IQL."""
    
    hidden_dims: Sequence[int] = (256, 256)
    activation: Callable = nn.relu
    use_layer_norm: bool = False
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, observations, training: bool = False):
        x = MLP(
            features=(*self.hidden_dims, 1),
            activation=self.activation,
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
        )(observations, training=training)
        
        return x.squeeze(-1)


class TrainState(train_state.TrainState):
    """Extended train state with target parameters."""
    target_params: dict


def create_train_state(
    network: nn.Module,
    key: jax.random.PRNGKey,
    dummy_input,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.0,
) -> TrainState:
    """Create training state with optimizer."""
    
    # Handle tuple inputs for multi-input networks
    if isinstance(dummy_input, tuple):
        params = network.init(key, *dummy_input, training=False)
    else:
        params = network.init(key, dummy_input, training=False)
    
    # Create optimizer with optional weight decay
    if weight_decay > 0:
        optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optax.adam(learning_rate)
    
    return TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=optimizer,
        target_params=params,
    )


def update_target_network(
    train_state: TrainState, 
    tau: float = 0.005
) -> TrainState:
    """Soft update of target network parameters."""
    
    new_target_params = jax.tree.map(
        lambda target, online: tau * online + (1 - tau) * target,
        train_state.target_params,
        train_state.params,
    )
    
    return train_state.replace(target_params=new_target_params)