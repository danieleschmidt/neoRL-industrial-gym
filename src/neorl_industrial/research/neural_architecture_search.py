"""Neural Architecture Search (NAS) for industrial RL optimization."""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Dict, List, Any, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
import itertools
import time
import concurrent.futures

from ..agents.base import OfflineAgent
from ..core.types import Array
from ..monitoring.logger import get_logger
from ..validation.input_validator import validate_array_input
from ..resilience.error_recovery import ErrorRecoveryManager


@dataclass
class ArchitectureConfig:
    """Configuration for neural architecture."""
    num_layers: int
    hidden_dims: List[int]
    activation: str
    dropout_rate: float
    normalization: str
    skip_connections: bool
    attention_mechanism: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_layers": self.num_layers,
            "hidden_dims": self.hidden_dims,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "normalization": self.normalization,
            "skip_connections": self.skip_connections,
            "attention_mechanism": self.attention_mechanism
        }


class NASSearchSpace:
    """Define search space for neural architecture search."""
    
    def __init__(self):
        self.search_space = {
            "num_layers": [2, 3, 4, 5],
            "hidden_dims": [[128], [256], [512], [128, 256], [256, 512], [128, 256, 128]],
            "activation": ["relu", "swish", "gelu"],
            "dropout_rate": [0.0, 0.1, 0.2],
            "normalization": ["none", "batch_norm", "layer_norm"],
            "skip_connections": [False, True],
            "attention_mechanism": [False, True]
        }
        self.logger = get_logger(self.__class__.__name__)
    
    def sample_architecture(self, random_key: jax.random.PRNGKey) -> ArchitectureConfig:
        """Sample random architecture from search space."""
        
        try:
            # Use JAX random for reproducibility
            keys = jax.random.split(random_key, len(self.search_space))
            
            config = {}
            for i, (param_name, param_values) in enumerate(self.search_space.items()):
                idx = jax.random.randint(keys[i], (), 0, len(param_values))
                config[param_name] = param_values[int(idx)]
            
            return ArchitectureConfig(**config)
            
        except Exception as e:
            self.logger.error(f"Architecture sampling failed: {e}")
            # Return default architecture
            return ArchitectureConfig(
                num_layers=3,
                hidden_dims=[256],
                activation="relu",
                dropout_rate=0.1,
                normalization="none",
                skip_connections=False,
                attention_mechanism=False
            )
    
    def get_all_architectures(self) -> List[ArchitectureConfig]:
        """Generate all possible architectures in search space."""
        
        try:
            # Limit search space for computational feasibility
            limited_space = {
                "num_layers": [2, 3, 4],
                "hidden_dims": [[128], [256], [128, 256]],
                "activation": ["relu", "swish"],
                "dropout_rate": [0.0, 0.1],
                "normalization": ["none", "layer_norm"],
                "skip_connections": [False, True],
                "attention_mechanism": [False, True]
            }
            
            architectures = []
            
            # Generate all combinations
            param_names = list(limited_space.keys())
            param_values = list(limited_space.values())
            
            for combination in itertools.product(*param_values):
                config_dict = dict(zip(param_names, combination))
                architectures.append(ArchitectureConfig(**config_dict))
            
            self.logger.info(f"Generated {len(architectures)} architecture configurations")
            return architectures
            
        except Exception as e:
            self.logger.error(f"Architecture generation failed: {e}")
            return [self.sample_architecture(jax.random.PRNGKey(42))]


class DynamicNeuralNetwork(nn.Module):
    """Dynamic neural network that can be configured at runtime."""
    
    config: ArchitectureConfig
    output_dim: int
    
    def setup(self):
        # Build layers based on configuration
        self.layers = []
        
        # Input projection layer
        first_hidden_dim = self.config.hidden_dims[0] if self.config.hidden_dims else 256
        self.input_layer = nn.Dense(first_hidden_dim, name="input_layer")
        
        # Hidden layers
        for i in range(self.config.num_layers):
            layer_dim = (self.config.hidden_dims[i] 
                        if i < len(self.config.hidden_dims) 
                        else self.config.hidden_dims[-1])
            
            # Dense layer
            dense_layer = nn.Dense(layer_dim, name=f"dense_{i}")
            self.layers.append(dense_layer)
            
            # Normalization
            if self.config.normalization == "batch_norm":
                norm_layer = nn.BatchNorm(name=f"batch_norm_{i}")
                self.layers.append(norm_layer)
            elif self.config.normalization == "layer_norm":
                norm_layer = nn.LayerNorm(name=f"layer_norm_{i}")
                self.layers.append(norm_layer)
            
            # Dropout
            if self.config.dropout_rate > 0:
                dropout_layer = nn.Dropout(self.config.dropout_rate, name=f"dropout_{i}")
                self.layers.append(dropout_layer)
        
        # Attention mechanism
        if self.config.attention_mechanism:
            self.attention = nn.MultiHeadDotProductAttention(
                num_heads=4,
                qkv_features=first_hidden_dim,
                name="attention"
            )
        
        # Output layer
        self.output_layer = nn.Dense(self.output_dim, name="output_layer")
    
    @nn.compact
    def __call__(self, x: Array, training: bool = True) -> Array:
        # Input projection
        x = self.input_layer(x)
        
        # Store for skip connections
        skip_connections = [x] if self.config.skip_connections else []
        
        # Apply layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Dropout):
                x = layer(x, deterministic=not training)
            elif isinstance(layer, nn.BatchNorm):
                x = layer(x, use_running_average=not training)
            else:
                x = layer(x)
            
            # Apply activation
            if isinstance(layer, nn.Dense):
                x = self._apply_activation(x)
                
                # Store for skip connections
                if self.config.skip_connections and len(skip_connections) < 3:
                    skip_connections.append(x)
        
        # Skip connections
        if self.config.skip_connections and len(skip_connections) > 1:
            # Add residual connections
            for skip in skip_connections[:-1]:
                if skip.shape == x.shape:
                    x = x + skip
                    break
        
        # Attention mechanism
        if self.config.attention_mechanism:
            # Reshape for attention (add sequence dimension)
            batch_size = x.shape[0]
            x_reshaped = x.reshape(batch_size, 1, -1)
            
            attended = self.attention(x_reshaped, x_reshaped)
            x = attended.reshape(batch_size, -1)
        
        # Output projection
        return self.output_layer(x)
    
    def _apply_activation(self, x: Array) -> Array:
        """Apply activation function based on configuration."""
        
        if self.config.activation == "relu":
            return nn.relu(x)
        elif self.config.activation == "swish":
            return nn.swish(x)
        elif self.config.activation == "gelu":
            return nn.gelu(x)
        elif self.config.activation == "tanh":
            return nn.tanh(x)
        else:
            return nn.relu(x)  # Default


class NASAgent(OfflineAgent):
    """RL Agent with Neural Architecture Search capabilities."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        architecture_config: ArchitectureConfig = None,
        **kwargs
    ):
        super().__init__(state_dim, action_dim, **kwargs)
        
        self.architecture_config = architecture_config or ArchitectureConfig(
            num_layers=3,
            hidden_dims=[256],
            activation="relu",
            dropout_rate=0.1,
            normalization="none",
            skip_connections=False,
            attention_mechanism=False
        )
        
        self.error_recovery = ErrorRecoveryManager()
        
    def _init_networks(self) -> Dict[str, Any]:
        """Initialize networks with NAS architecture."""
        
        try:
            # Create dynamic networks
            critic = DynamicNeuralNetwork(
                config=self.architecture_config,
                output_dim=1,
                name="critic_network"
            )
            
            actor = DynamicNeuralNetwork(
                config=self.architecture_config,
                output_dim=self.action_dim,
                name="actor_network"
            )
            
            # Initialize parameters
            dummy_state = jnp.ones((1, self.state_dim))
            dummy_action = jnp.ones((1, self.action_dim))
            dummy_combined = jnp.concatenate([dummy_state, dummy_action], axis=-1)
            
            key1, key2 = jax.random.split(self.key, 2)
            
            critic_params = critic.init(key1, dummy_combined, training=False)
            actor_params = actor.init(key2, dummy_state, training=False)
            
            return {
                "critic": critic,
                "critic_params": critic_params,
                "actor": actor,
                "actor_params": actor_params,
                "critic_opt": optax.adam(3e-4).init(critic_params),
                "actor_opt": optax.adam(3e-4).init(actor_params),
            }
            
        except Exception as e:
            self.logger.error(f"NAS network initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize NAS networks: {e}") from e
    
    def _create_train_step(self) -> Any:
        """Create training step for NAS agent."""
        
        @jax.jit
        def train_step(state, batch):
            try:
                def critic_loss_fn(params):
                    # Combine state and action for critic input
                    critic_input = jnp.concatenate([
                        batch["observations"], 
                        batch["actions"]
                    ], axis=-1)
                    
                    q_vals = state["critic"].apply(
                        params,
                        critic_input,
                        training=True,
                        rngs={"dropout": jax.random.PRNGKey(0)}
                    )
                    
                    # Simple Q-learning loss
                    targets = batch["rewards"].reshape(-1, 1) + 0.99 * q_vals
                    loss = jnp.mean((q_vals - jax.lax.stop_gradient(targets)) ** 2)
                    
                    return loss
                
                # Update critic
                grads = jax.grad(critic_loss_fn)(state["critic_params"])
                
                # Gradient clipping for stability
                grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
                
                critic_updates, new_critic_opt = optax.update(
                    grads, state["critic_opt"], state["critic_params"]
                )
                new_critic_params = optax.apply_updates(
                    state["critic_params"], critic_updates
                )
                
                new_state = state.copy()
                new_state.update({
                    "critic_params": new_critic_params,
                    "critic_opt": new_critic_opt,
                })
                
                metrics = {"critic_loss": critic_loss_fn(state["critic_params"])}
                
                return new_state, metrics
                
            except Exception as e:
                # Return error state
                error_metrics = {"critic_loss": float('inf'), "error": str(e)}
                return state, error_metrics
        
        return train_step
    
    def _update_step(self, state, batch):
        """Update step for NAS agent."""
        return self.train_step(state, batch)
    
    def _predict_impl(self, observations, deterministic=True):
        """Predict using NAS architecture."""
        
        try:
            if not hasattr(self, 'state') or self.state is None:
                raise RuntimeError("Model not initialized")
            
            actions = self.state["actor"].apply(
                self.state["actor_params"],
                observations,
                training=not deterministic,
                rngs={"dropout": jax.random.PRNGKey(0)} if not deterministic else {}
            )
            
            return jnp.tanh(actions)
            
        except Exception as e:
            self.logger.error(f"NAS prediction failed: {e}")
            # Return zero actions as fallback
            return jnp.zeros((observations.shape[0], self.action_dim))


class AutoMLForIndustrialRL:
    """AutoML framework for industrial RL with NAS and hyperparameter optimization."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        search_budget: int = 50,
        parallel_evaluations: int = 4
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.search_budget = search_budget
        self.parallel_evaluations = parallel_evaluations
        
        self.search_space = NASSearchSpace()
        self.evaluation_history = []
        self.best_architecture = None
        self.best_performance = float('-inf')
        
        self.logger = get_logger(self.__class__.__name__)
        self.error_recovery = ErrorRecoveryManager()
        
    def evaluate_architecture(
        self,
        architecture_config: ArchitectureConfig,
        dataset: Dict[str, Array],
        evaluation_epochs: int = 20
    ) -> Dict[str, Any]:
        """Evaluate single architecture configuration."""
        
        start_time = time.time()
        
        try:
            # Validate inputs
            validate_array_input(dataset["observations"], "observations")
            validate_array_input(dataset["actions"], "actions")
            
            # Create NAS agent with architecture
            agent = NASAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                architecture_config=architecture_config
            )
            
            # Train agent with error recovery
            def train_with_recovery():
                return agent.train(
                    dataset,
                    n_epochs=evaluation_epochs,
                    batch_size=64  # Smaller batch for faster evaluation
                )
            
            training_result = self.error_recovery.execute_with_recovery(
                train_with_recovery,
                operation_name="nas_architecture_evaluation"
            )
            
            # Extract performance metrics
            final_metrics = training_result.get("final_metrics", {})
            performance_score = self._compute_performance_score(final_metrics)
            
            evaluation_time = time.time() - start_time
            
            result = {
                "architecture_config": architecture_config.to_dict(),
                "performance_score": performance_score,
                "training_metrics": final_metrics,
                "evaluation_time": evaluation_time,
                "success": True
            }
            
            # Update best architecture
            if performance_score > self.best_performance:
                self.best_performance = performance_score
                self.best_architecture = architecture_config
                self.logger.info(f"New best architecture found: {performance_score:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Architecture evaluation failed: {e}")
            return {
                "architecture_config": architecture_config.to_dict(),
                "performance_score": float('-inf'),
                "error": str(e),
                "evaluation_time": time.time() - start_time,
                "success": False
            }
    
    def random_search(
        self,
        dataset: Dict[str, Array],
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """Perform random architecture search."""
        
        self.logger.info(f"Starting random search with budget {self.search_budget}")
        
        key = jax.random.PRNGKey(random_seed)
        search_results = []
        
        for iteration in range(self.search_budget):
            key, subkey = jax.random.split(key)
            
            # Sample random architecture
            architecture = self.search_space.sample_architecture(subkey)
            
            # Evaluate architecture
            result = self.evaluate_architecture(architecture, dataset)
            search_results.append(result)
            
            # Log progress
            if (iteration + 1) % 10 == 0:
                successful = sum(1 for r in search_results if r["success"])
                self.logger.info(
                    f"Random search progress: {iteration + 1}/{self.search_budget}, "
                    f"Success rate: {successful}/{len(search_results)}"
                )
        
        self.evaluation_history.extend(search_results)
        
        return {
            "search_method": "random",
            "search_results": search_results,
            "best_architecture": self.best_architecture.to_dict() if self.best_architecture else None,
            "best_performance": self.best_performance,
            "total_evaluations": len(search_results)
        }
    
    def evolutionary_search(
        self,
        dataset: Dict[str, Array],
        population_size: int = 20,
        generations: int = 10,
        mutation_rate: float = 0.3
    ) -> Dict[str, Any]:
        """Perform evolutionary architecture search."""
        
        self.logger.info(f"Starting evolutionary search: {population_size} pop, {generations} gen")
        
        key = jax.random.PRNGKey(42)
        search_results = []
        
        # Initialize population
        population = []
        for _ in range(population_size):
            key, subkey = jax.random.split(key)
            architecture = self.search_space.sample_architecture(subkey)
            population.append(architecture)
        
        for generation in range(generations):
            self.logger.info(f"Generation {generation + 1}/{generations}")
            
            # Evaluate population (parallel if possible)
            generation_results = []
            
            if self.parallel_evaluations > 1:
                # Parallel evaluation
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.parallel_evaluations
                ) as executor:
                    futures = [
                        executor.submit(self.evaluate_architecture, arch, dataset, 15)
                        for arch in population
                    ]
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            generation_results.append(result)
                        except Exception as e:
                            self.logger.error(f"Parallel evaluation failed: {e}")
            else:
                # Sequential evaluation
                for architecture in population:
                    result = self.evaluate_architecture(architecture, dataset, 15)
                    generation_results.append(result)
            
            search_results.extend(generation_results)
            
            # Selection and reproduction
            if generation < generations - 1:  # Not last generation
                population = self._evolve_population(
                    population, generation_results, mutation_rate, key
                )
                key, _ = jax.random.split(key)
        
        self.evaluation_history.extend(search_results)
        
        return {
            "search_method": "evolutionary",
            "search_results": search_results,
            "best_architecture": self.best_architecture.to_dict() if self.best_architecture else None,
            "best_performance": self.best_performance,
            "total_evaluations": len(search_results),
            "population_size": population_size,
            "generations": generations
        }
    
    def _evolve_population(
        self,
        population: List[ArchitectureConfig],
        results: List[Dict[str, Any]],
        mutation_rate: float,
        random_key: jax.random.PRNGKey
    ) -> List[ArchitectureConfig]:
        """Evolve population for next generation."""
        
        try:
            # Sort by performance
            combined = list(zip(population, results))
            combined.sort(key=lambda x: x[1].get("performance_score", float('-inf')), reverse=True)
            
            # Select top 50% as parents
            elite_size = len(population) // 2
            parents = [arch for arch, _ in combined[:elite_size]]
            
            # Generate offspring
            new_population = parents.copy()  # Keep elite
            
            while len(new_population) < len(population):
                key1, key2, random_key = jax.random.split(random_key, 3)
                
                # Select two parents
                parent1_idx = jax.random.randint(key1, (), 0, len(parents))
                parent2_idx = jax.random.randint(key2, (), 0, len(parents))
                
                parent1 = parents[int(parent1_idx)]
                parent2 = parents[int(parent2_idx)]
                
                # Crossover and mutation
                offspring = self._crossover_and_mutate(
                    parent1, parent2, mutation_rate, random_key
                )
                new_population.append(offspring)
                
                random_key, _ = jax.random.split(random_key)
            
            return new_population[:len(population)]
            
        except Exception as e:
            self.logger.error(f"Population evolution failed: {e}")
            # Return original population
            return population
    
    def _crossover_and_mutate(
        self,
        parent1: ArchitectureConfig,
        parent2: ArchitectureConfig,
        mutation_rate: float,
        random_key: jax.random.PRNGKey
    ) -> ArchitectureConfig:
        """Perform crossover and mutation to create offspring."""
        
        try:
            keys = jax.random.split(random_key, 7)
            
            # Crossover: randomly choose from each parent
            offspring_config = {}
            
            # Number of layers - crossover
            offspring_config["num_layers"] = (
                parent1.num_layers if jax.random.uniform(keys[0]) < 0.5 
                else parent2.num_layers
            )
            
            # Hidden dims - crossover
            offspring_config["hidden_dims"] = (
                parent1.hidden_dims if jax.random.uniform(keys[1]) < 0.5
                else parent2.hidden_dims
            )
            
            # Activation - crossover
            offspring_config["activation"] = (
                parent1.activation if jax.random.uniform(keys[2]) < 0.5
                else parent2.activation
            )
            
            # Dropout rate - crossover
            offspring_config["dropout_rate"] = (
                parent1.dropout_rate if jax.random.uniform(keys[3]) < 0.5
                else parent2.dropout_rate
            )
            
            # Normalization - crossover
            offspring_config["normalization"] = (
                parent1.normalization if jax.random.uniform(keys[4]) < 0.5
                else parent2.normalization
            )
            
            # Skip connections - crossover
            offspring_config["skip_connections"] = (
                parent1.skip_connections if jax.random.uniform(keys[5]) < 0.5
                else parent2.skip_connections
            )
            
            # Attention - crossover
            offspring_config["attention_mechanism"] = (
                parent1.attention_mechanism if jax.random.uniform(keys[6]) < 0.5
                else parent2.attention_mechanism
            )
            
            # Mutation
            if jax.random.uniform(random_key) < mutation_rate:
                # Randomly mutate one component
                mutated_offspring = self.search_space.sample_architecture(random_key)
                
                # Replace one random component
                components = list(offspring_config.keys())
                mutate_idx = jax.random.randint(random_key, (), 0, len(components))
                mutate_component = components[int(mutate_idx)]
                
                offspring_config[mutate_component] = getattr(mutated_offspring, mutate_component)
            
            return ArchitectureConfig(**offspring_config)
            
        except Exception as e:
            self.logger.error(f"Crossover/mutation failed: {e}")
            # Return parent1 as fallback
            return parent1
    
    def _compute_performance_score(self, metrics: Dict[str, Any]) -> float:
        """Compute performance score from training metrics."""
        
        try:
            # Multi-objective scoring
            score = 0.0
            
            # Primary: minimize loss
            if "critic_loss" in metrics:
                loss = float(metrics["critic_loss"])
                if not (jnp.isnan(loss) or jnp.isinf(loss)):
                    score += max(0.0, 1.0 - loss / 10.0)  # Normalize loss
            
            # Secondary: training stability
            if "grad_norm" in metrics:
                grad_norm = float(metrics.get("grad_norm", 1.0))
                if not (jnp.isnan(grad_norm) or jnp.isinf(grad_norm)):
                    score += max(0.0, 1.0 - grad_norm / 5.0)  # Penalize large gradients
            
            # Bonus for successful training
            if not any(jnp.isnan(v) or jnp.isinf(v) for v in metrics.values() if isinstance(v, (int, float))):
                score += 0.5  # Stability bonus
            
            return float(score)
            
        except Exception as e:
            self.logger.error(f"Performance scoring failed: {e}")
            return 0.0
    
    def get_best_architecture(self) -> Optional[ArchitectureConfig]:
        """Get the best architecture found during search."""
        return self.best_architecture
    
    def get_search_summary(self) -> Dict[str, Any]:
        """Get summary of architecture search results."""
        
        successful_evals = [r for r in self.evaluation_history if r["success"]]
        
        if not successful_evals:
            return {
                "total_evaluations": len(self.evaluation_history),
                "successful_evaluations": 0,
                "success_rate": 0.0,
                "best_performance": self.best_performance,
                "best_architecture": None
            }
        
        performance_scores = [r["performance_score"] for r in successful_evals]
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "successful_evaluations": len(successful_evals),
            "success_rate": len(successful_evals) / len(self.evaluation_history),
            "best_performance": self.best_performance,
            "average_performance": float(np.mean(performance_scores)),
            "std_performance": float(np.std(performance_scores)),
            "best_architecture": self.best_architecture.to_dict() if self.best_architecture else None
        }