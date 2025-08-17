"""Foundation models for industrial RL with pre-training and fine-tuning."""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
import pickle
from pathlib import Path

from ..agents.base import OfflineAgent
from ..core.types import Array
from ..monitoring.logger import get_logger
from ..validation.input_validator import validate_array_input
from ..resilience.error_recovery import ErrorRecoveryManager


@dataclass
class FoundationModelConfig:
    """Configuration for foundation model architecture."""
    embed_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    mlp_dim: int = 2048
    dropout_rate: float = 0.1
    max_sequence_length: int = 1024
    vocab_size: int = 10000  # For tokenized inputs
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout_rate": self.dropout_rate,
            "max_sequence_length": self.max_sequence_length,
            "vocab_size": self.vocab_size
        }


class TransformerBlock(nn.Module):
    """Transformer block for foundation model."""
    
    embed_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float
    
    @nn.compact
    def __call__(self, x: Array, training: bool = True) -> Array:
        # Multi-head attention
        attention_out = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.embed_dim,
            dropout_rate=self.dropout_rate,
            name="attention"
        )(x, x, deterministic=not training)
        
        # Add & Norm
        x = nn.LayerNorm(name="norm1")(x + attention_out)
        
        # MLP
        mlp_out = nn.Dense(self.mlp_dim, name="mlp_dense1")(x)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dropout(self.dropout_rate, name="mlp_dropout")(
            mlp_out, deterministic=not training
        )
        mlp_out = nn.Dense(self.embed_dim, name="mlp_dense2")(mlp_out)
        
        # Add & Norm
        x = nn.LayerNorm(name="norm2")(x + mlp_out)
        
        return x


class IndustrialFoundationTransformer(nn.Module):
    """Foundation transformer model for industrial data."""
    
    config: FoundationModelConfig
    
    def setup(self):
        # Input embedding
        self.input_projection = nn.Dense(
            self.config.embed_dim,
            name="input_projection"
        )
        
        # Positional encoding
        self.pos_encoding = self.param(
            "pos_encoding",
            nn.initializers.normal(0.02),
            (self.config.max_sequence_length, self.config.embed_dim)
        )
        
        # Transformer layers
        self.transformer_layers = [
            TransformerBlock(
                embed_dim=self.config.embed_dim,
                num_heads=self.config.num_heads,
                mlp_dim=self.config.mlp_dim,
                dropout_rate=self.config.dropout_rate,
                name=f"transformer_{i}"
            )
            for i in range(self.config.num_layers)
        ]
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(name="final_norm")
    
    def __call__(
        self,
        x: Array,
        mask: Optional[Array] = None,
        training: bool = True
    ) -> Array:
        batch_size, seq_len, input_dim = x.shape
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_embed = self.pos_encoding[:seq_len]
        x = x + pos_embed[None, :, :]  # Broadcast over batch
        
        # Apply dropout to embeddings
        x = nn.Dropout(self.config.dropout_rate, name="embed_dropout")(
            x, deterministic=not training
        )
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, training=training)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x


class IndustrialFoundationModel(OfflineAgent):
    """Foundation model for industrial RL with pre-training capabilities."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: FoundationModelConfig = None,
        pre_training_objectives: List[str] = None,
        **kwargs
    ):
        super().__init__(state_dim, action_dim, **kwargs)
        
        self.config = config or FoundationModelConfig()
        self.pre_training_objectives = pre_training_objectives or [
            "masked_language_modeling",
            "next_token_prediction",
            "contrastive_learning"
        ]
        
        self.is_pre_trained = False
        self.error_recovery = ErrorRecoveryManager()
        
        self.logger.info(f"Initialized foundation model with config: {self.config.to_dict()}")
    
    def _init_networks(self) -> Dict[str, Any]:
        """Initialize foundation model networks."""
        
        try:
            # Foundation transformer
            foundation_model = IndustrialFoundationTransformer(config=self.config)
            
            # Task-specific heads
            class TaskHead(nn.Module):
                output_dim: int
                
                @nn.compact
                def __call__(self, x: Array) -> Array:
                    # Pool sequence dimension (take last token or mean)
                    if len(x.shape) == 3:  # (batch, seq, embed)
                        x = jnp.mean(x, axis=1)  # Mean pooling
                    
                    # Task-specific layers
                    x = nn.Dense(self.config.embed_dim // 2, name="task_dense1")(x)
                    x = nn.gelu(x)
                    x = nn.Dense(self.output_dim, name="task_output")(x)
                    return x
            
            # Pre-training heads
            class PreTrainingHead(nn.Module):
                vocab_size: int
                
                @nn.compact
                def __call__(self, x: Array) -> Array:
                    return nn.Dense(self.vocab_size, name="pre_train_head")(x)
            
            # Initialize parameters
            dummy_input = jnp.ones((1, 10, self.state_dim + self.action_dim))
            dummy_sequence = jnp.ones((1, 10, self.config.embed_dim))
            
            key1, key2, key3, key4 = jax.random.split(self.key, 4)
            
            # Foundation model
            foundation_params = foundation_model.init(
                key1, dummy_input, training=False
            )
            
            # Task heads
            critic_head = TaskHead(output_dim=1)
            critic_head_params = critic_head.init(key2, dummy_sequence)
            
            actor_head = TaskHead(output_dim=self.action_dim)
            actor_head_params = actor_head.init(key3, dummy_sequence)
            
            # Pre-training head
            pretrain_head = PreTrainingHead(vocab_size=self.config.vocab_size)
            pretrain_head_params = pretrain_head.init(key4, dummy_sequence)
            
            return {
                "foundation_model": foundation_model,
                "foundation_params": foundation_params,
                "critic_head": critic_head,
                "critic_head_params": critic_head_params,
                "actor_head": actor_head,
                "actor_head_params": actor_head_params,
                "pretrain_head": pretrain_head,
                "pretrain_head_params": pretrain_head_params,
                "foundation_opt": optax.adam(1e-4).init(foundation_params),
                "critic_opt": optax.adam(3e-4).init(critic_head_params),
                "actor_opt": optax.adam(3e-4).init(actor_head_params),
                "pretrain_opt": optax.adam(1e-4).init(pretrain_head_params),
            }
            
        except Exception as e:
            self.logger.error(f"Foundation model initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize foundation model: {e}") from e
    
    def pre_train(
        self,
        pre_training_data: Dict[str, Array],
        n_epochs: int = 100,
        batch_size: int = 32,
        save_checkpoints: bool = True,
        checkpoint_dir: str = "foundation_checkpoints"
    ) -> Dict[str, Any]:
        """Pre-train foundation model on large-scale industrial data."""
        
        self.logger.info(f"Starting pre-training for {n_epochs} epochs")
        
        try:
            # Validate pre-training data
            validate_array_input(pre_training_data["sequences"], "pre_training_sequences")
            
            if not hasattr(self, 'state'):
                self.state = self._init_networks()
                self.pretrain_step = self._create_pretrain_step()
            
            # Create checkpoint directory
            if save_checkpoints:
                checkpoint_path = Path(checkpoint_dir)
                checkpoint_path.mkdir(exist_ok=True)
            
            # Pre-training loop
            pretrain_metrics = []
            n_samples = len(pre_training_data["sequences"])
            
            for epoch in range(n_epochs):
                epoch_losses = []
                
                # Create batches
                n_batches = max(1, n_samples // batch_size)
                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, n_samples)
                    
                    batch = {
                        "sequences": pre_training_data["sequences"][start_idx:end_idx]
                    }
                    
                    # Add targets for different objectives
                    if "masked_language_modeling" in self.pre_training_objectives:
                        batch["mlm_targets"] = self._create_mlm_targets(batch["sequences"])
                    
                    # Pre-training step
                    self.state, step_metrics = self.pretrain_step(self.state, batch)
                    epoch_losses.append(step_metrics)
                
                # Aggregate epoch metrics
                avg_metrics = {}
                if epoch_losses:
                    for key in epoch_losses[0].keys():
                        avg_metrics[key] = np.mean([m[key] for m in epoch_losses])
                
                pretrain_metrics.append(avg_metrics)
                
                # Log progress
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Pre-training epoch {epoch + 1}: {avg_metrics}")
                
                # Save checkpoint
                if save_checkpoints and (epoch + 1) % 20 == 0:
                    checkpoint_file = checkpoint_path / f"checkpoint_epoch_{epoch + 1}.pkl"
                    self._save_checkpoint(checkpoint_file)
            
            self.is_pre_trained = True
            self.logger.info("Pre-training completed successfully")
            
            return {
                "pretrain_metrics": pretrain_metrics,
                "final_metrics": pretrain_metrics[-1] if pretrain_metrics else {},
                "pre_trained": True,
                "checkpoint_dir": checkpoint_dir if save_checkpoints else None
            }
            
        except Exception as e:
            self.logger.error(f"Pre-training failed: {e}")
            return {
                "error": str(e),
                "pre_trained": False
            }
    
    def _create_pretrain_step(self) -> Any:
        """Create pre-training step function."""
        
        @jax.jit
        def pretrain_step(state, batch):
            try:
                def pretrain_loss_fn(foundation_params, pretrain_params):
                    sequences = batch["sequences"]
                    
                    # Forward pass through foundation model
                    hidden_states = state["foundation_model"].apply(
                        foundation_params,
                        sequences,
                        training=True,
                        rngs={"dropout": jax.random.PRNGKey(0)}
                    )
                    
                    total_loss = 0.0
                    
                    # Masked Language Modeling loss
                    if "mlm_targets" in batch:
                        mlm_logits = state["pretrain_head"].apply(
                            pretrain_params,
                            hidden_states
                        )
                        
                        # Flatten for loss computation
                        mlm_logits_flat = mlm_logits.reshape(-1, self.config.vocab_size)
                        mlm_targets_flat = batch["mlm_targets"].reshape(-1)
                        
                        # Cross-entropy loss
                        mlm_loss = optax.softmax_cross_entropy_with_integer_labels(
                            mlm_logits_flat, mlm_targets_flat
                        )
                        total_loss += jnp.mean(mlm_loss)
                    
                    # Next token prediction loss
                    if "next_token_prediction" in self.pre_training_objectives:
                        # Shift targets by one position
                        shifted_targets = jnp.roll(sequences, -1, axis=1)
                        
                        next_token_logits = state["pretrain_head"].apply(
                            pretrain_params,
                            hidden_states
                        )
                        
                        # Use continuous values instead of discrete tokens
                        next_token_loss = jnp.mean(
                            (next_token_logits[..., :sequences.shape[-1]] - shifted_targets) ** 2
                        )
                        total_loss += next_token_loss * 0.1  # Scale down
                    
                    return total_loss
                
                # Compute gradients
                loss_val, grads = jax.value_and_grad(
                    pretrain_loss_fn,
                    argnums=(0, 1)
                )(state["foundation_params"], state["pretrain_head_params"])
                
                foundation_grads, pretrain_grads = grads
                
                # Update foundation model
                foundation_updates, new_foundation_opt = optax.update(
                    foundation_grads, state["foundation_opt"], state["foundation_params"]
                )
                new_foundation_params = optax.apply_updates(
                    state["foundation_params"], foundation_updates
                )
                
                # Update pre-training head
                pretrain_updates, new_pretrain_opt = optax.update(
                    pretrain_grads, state["pretrain_opt"], state["pretrain_head_params"]
                )
                new_pretrain_params = optax.apply_updates(
                    state["pretrain_head_params"], pretrain_updates
                )
                
                new_state = state.copy()
                new_state.update({
                    "foundation_params": new_foundation_params,
                    "foundation_opt": new_foundation_opt,
                    "pretrain_head_params": new_pretrain_params,
                    "pretrain_opt": new_pretrain_opt,
                })
                
                metrics = {"pretrain_loss": loss_val}
                
                return new_state, metrics
                
            except Exception as e:
                # Return error state
                error_metrics = {"pretrain_loss": float('inf'), "error": str(e)}
                return state, error_metrics
        
        return pretrain_step
    
    def fine_tune(
        self,
        downstream_dataset: Dict[str, Array],
        n_epochs: int = 50,
        freeze_foundation: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Fine-tune pre-trained model on downstream task."""
        
        self.logger.info(f"Fine-tuning on downstream task (freeze_foundation={freeze_foundation})")
        
        try:
            if not self.is_pre_trained:
                self.logger.warning("Model not pre-trained. Consider pre-training first.")
            
            # Prepare sequences from state-action pairs
            sequences = self._prepare_sequences_for_finetuning(downstream_dataset)
            
            # Create fine-tuning step
            self.finetune_step = self._create_finetune_step(freeze_foundation)
            
            # Fine-tuning loop
            finetune_metrics = []
            n_samples = len(sequences)
            batch_size = kwargs.get("batch_size", 32)
            
            for epoch in range(n_epochs):
                epoch_losses = []
                n_batches = max(1, n_samples // batch_size)
                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, n_samples)
                    
                    batch = {
                        "sequences": sequences[start_idx:end_idx],
                        "observations": downstream_dataset["observations"][start_idx:end_idx],
                        "actions": downstream_dataset["actions"][start_idx:end_idx],
                        "rewards": downstream_dataset["rewards"][start_idx:end_idx]
                    }
                    
                    # Fine-tuning step
                    self.state, step_metrics = self.finetune_step(self.state, batch)
                    epoch_losses.append(step_metrics)
                
                # Aggregate metrics
                avg_metrics = {}
                if epoch_losses:
                    for key in epoch_losses[0].keys():
                        avg_metrics[key] = np.mean([m[key] for m in epoch_losses])
                
                finetune_metrics.append(avg_metrics)
                
                # Log progress
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Fine-tuning epoch {epoch + 1}: {avg_metrics}")
            
            self.is_trained = True
            
            return {
                "finetune_metrics": finetune_metrics,
                "final_metrics": finetune_metrics[-1] if finetune_metrics else {},
                "fine_tuned": True
            }
            
        except Exception as e:
            self.logger.error(f"Fine-tuning failed: {e}")
            return {
                "error": str(e),
                "fine_tuned": False
            }
    
    def _create_finetune_step(self, freeze_foundation: bool = False) -> Any:
        """Create fine-tuning step function."""
        
        @jax.jit
        def finetune_step(state, batch):
            try:
                def finetune_loss_fn(params):
                    foundation_params = params["foundation_params"]
                    critic_params = params["critic_head_params"]
                    actor_params = params["actor_head_params"]
                    
                    sequences = batch["sequences"]
                    
                    # Forward pass through foundation model
                    if freeze_foundation:
                        # Stop gradients to foundation model
                        hidden_states = jax.lax.stop_gradient(
                            state["foundation_model"].apply(
                                foundation_params,
                                sequences,
                                training=False
                            )
                        )
                    else:
                        hidden_states = state["foundation_model"].apply(
                            foundation_params,
                            sequences,
                            training=True,
                            rngs={"dropout": jax.random.PRNGKey(0)}
                        )
                    
                    # Critic loss
                    q_values = state["critic_head"].apply(critic_params, hidden_states)
                    targets = batch["rewards"].reshape(-1, 1)
                    critic_loss = jnp.mean((q_values - targets) ** 2)
                    
                    # Actor loss (behavioral cloning for now)
                    predicted_actions = state["actor_head"].apply(actor_params, hidden_states)
                    actor_loss = jnp.mean((predicted_actions - batch["actions"]) ** 2)
                    
                    return critic_loss + actor_loss
                
                # Compute gradients
                all_params = {
                    "foundation_params": state["foundation_params"],
                    "critic_head_params": state["critic_head_params"],
                    "actor_head_params": state["actor_head_params"]
                }
                
                grads = jax.grad(finetune_loss_fn)(all_params)
                
                new_state = state.copy()
                
                # Update foundation model (if not frozen)
                if not freeze_foundation:
                    foundation_updates, new_foundation_opt = optax.update(
                        grads["foundation_params"],
                        state["foundation_opt"],
                        state["foundation_params"]
                    )
                    new_foundation_params = optax.apply_updates(
                        state["foundation_params"], foundation_updates
                    )
                    new_state.update({
                        "foundation_params": new_foundation_params,
                        "foundation_opt": new_foundation_opt,
                    })
                
                # Update critic head
                critic_updates, new_critic_opt = optax.update(
                    grads["critic_head_params"],
                    state["critic_opt"],
                    state["critic_head_params"]
                )
                new_critic_params = optax.apply_updates(
                    state["critic_head_params"], critic_updates
                )
                
                # Update actor head
                actor_updates, new_actor_opt = optax.update(
                    grads["actor_head_params"],
                    state["actor_opt"],
                    state["actor_head_params"]
                )
                new_actor_params = optax.apply_updates(
                    state["actor_head_params"], actor_updates
                )
                
                new_state.update({
                    "critic_head_params": new_critic_params,
                    "critic_opt": new_critic_opt,
                    "actor_head_params": new_actor_params,
                    "actor_opt": new_actor_opt,
                })
                
                metrics = {"finetune_loss": finetune_loss_fn(all_params)}
                
                return new_state, metrics
                
            except Exception as e:
                # Return error state
                error_metrics = {"finetune_loss": float('inf'), "error": str(e)}
                return state, error_metrics
        
        return finetune_step
    
    def _prepare_sequences_for_finetuning(
        self,
        dataset: Dict[str, Array]
    ) -> Array:
        """Prepare sequential data for fine-tuning."""
        
        try:
            observations = dataset["observations"]
            actions = dataset["actions"]
            
            # Combine state and action
            state_action = jnp.concatenate([observations, actions], axis=-1)
            
            # Create sequences (sliding window)
            sequence_length = min(10, len(state_action))  # Short sequences for speed
            
            sequences = []
            for i in range(len(state_action) - sequence_length + 1):
                seq = state_action[i:i + sequence_length]
                sequences.append(seq)
            
            if not sequences:
                # Fallback: just repeat the data
                sequences = [state_action[:sequence_length]]
            
            return jnp.array(sequences)
            
        except Exception as e:
            self.logger.error(f"Sequence preparation failed: {e}")
            # Return dummy sequences
            dummy_seq = jnp.ones((1, 10, self.state_dim + self.action_dim))
            return dummy_seq
    
    def _create_mlm_targets(self, sequences: Array) -> Array:
        """Create masked language modeling targets."""
        
        try:
            # Simple masking strategy: mask 15% of tokens
            mask_prob = 0.15
            
            # Create random mask
            key = jax.random.PRNGKey(42)
            mask = jax.random.uniform(key, sequences.shape[:2]) < mask_prob
            
            # Create targets (discretized values for simplicity)
            targets = jnp.clip(sequences * 100, 0, self.config.vocab_size - 1).astype(jnp.int32)
            
            return targets
            
        except Exception as e:
            self.logger.error(f"MLM target creation failed: {e}")
            # Return dummy targets
            return jnp.zeros(sequences.shape[:2], dtype=jnp.int32)
    
    def _save_checkpoint(self, checkpoint_path: Path):
        """Save model checkpoint."""
        
        try:
            checkpoint_data = {
                "state": self.state,
                "config": self.config.to_dict(),
                "is_pre_trained": self.is_pre_trained,
                "is_trained": self.is_trained
            }
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.state = checkpoint_data["state"]
            self.is_pre_trained = checkpoint_data.get("is_pre_trained", False)
            self.is_trained = checkpoint_data.get("is_trained", False)
            
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            
        except Exception as e:
            self.logger.error(f"Checkpoint load failed: {e}")
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e
    
    def _create_train_step(self) -> Any:
        """Create standard training step (delegates to fine-tuning)."""
        return self._create_finetune_step(freeze_foundation=False)
    
    def _update_step(self, state, batch):
        """Update step for foundation model."""
        if not hasattr(self, 'finetune_step'):
            self.finetune_step = self._create_finetune_step()
        
        # Prepare sequences from batch
        sequences = self._prepare_sequences_for_finetuning(batch)
        
        # Add sequences to batch
        enhanced_batch = {**batch, "sequences": sequences}
        
        return self.finetune_step(state, enhanced_batch)
    
    def _predict_impl(self, observations, deterministic=True):
        """Predict using foundation model."""
        
        try:
            if not hasattr(self, 'state') or self.state is None:
                raise RuntimeError("Model not initialized")
            
            # Prepare sequences for prediction
            dummy_actions = jnp.zeros((observations.shape[0], self.action_dim))
            state_action = jnp.concatenate([observations, dummy_actions], axis=-1)
            
            # Create sequences (just repeat for simplicity)
            sequences = jnp.repeat(state_action[:, None, :], 10, axis=1)
            
            # Forward pass
            hidden_states = self.state["foundation_model"].apply(
                self.state["foundation_params"],
                sequences,
                training=False
            )
            
            # Get actions from actor head
            actions = self.state["actor_head"].apply(
                self.state["actor_head_params"],
                hidden_states
            )
            
            return jnp.tanh(actions)
            
        except Exception as e:
            self.logger.error(f"Foundation model prediction failed: {e}")
            # Return zero actions as fallback
            return jnp.zeros((observations.shape[0], self.action_dim))


def create_foundation_model_suite(
    state_dim: int = 20,
    action_dim: int = 6,
    model_sizes: List[str] = None
) -> Dict[str, IndustrialFoundationModel]:
    """Create suite of foundation models with different sizes."""
    
    if model_sizes is None:
        model_sizes = ["small", "medium", "large"]
    
    size_configs = {
        "small": FoundationModelConfig(
            embed_dim=256,
            num_layers=4,
            num_heads=4,
            mlp_dim=1024
        ),
        "medium": FoundationModelConfig(
            embed_dim=512,
            num_layers=6,
            num_heads=8,
            mlp_dim=2048
        ),
        "large": FoundationModelConfig(
            embed_dim=768,
            num_layers=12,
            num_heads=12,
            mlp_dim=3072
        )
    }
    
    models = {}
    
    for size in model_sizes:
        if size in size_configs:
            try:
                model = IndustrialFoundationModel(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    config=size_configs[size]
                )
                models[size] = model
                
            except Exception as e:
                print(f"Failed to create {size} foundation model: {e}")
    
    return models