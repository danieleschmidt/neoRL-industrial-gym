"""Ensemble methods for uncertainty quantification and robustness."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union
from functools import partial

from .base import OfflineAgent
from .cql import CQLAgent
from .iql import IQLAgent
from .td3bc import TD3BCAgent
from ..core.types import Array, StateArray, ActionArray


class EnsembleAgent(OfflineAgent):
    """Ensemble of offline RL agents for uncertainty quantification.
    
    Combines multiple agents to provide robust predictions with uncertainty estimates.
    Useful for safety-critical industrial applications where understanding
    prediction confidence is crucial.
    """
    
    def __init__(
        self,
        agent_configs: List[Dict[str, Any]],
        state_dim: int,
        action_dim: int,
        ensemble_method: str = "mean",
        uncertainty_threshold: float = 0.2,
        safety_critic: bool = True,
        constraint_threshold: float = 0.1,
        seed: int = 42,
    ):
        """Initialize ensemble agent.
        
        Args:
            agent_configs: List of configurations for individual agents
            state_dim: State space dimension
            action_dim: Action space dimension
            ensemble_method: Method for combining predictions ("mean", "voting", "weighted")
            uncertainty_threshold: Threshold for high uncertainty detection
            safety_critic: Use safety critic
            constraint_threshold: Safety constraint threshold
            seed: Random seed
        """
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            safety_critic=safety_critic,
            constraint_threshold=constraint_threshold,
            seed=seed,
        )
        
        self.ensemble_method = ensemble_method
        self.uncertainty_threshold = uncertainty_threshold
        self.agent_configs = agent_configs
        
        # Initialize individual agents
        self.agents = self._init_agents()
        self.weights = np.ones(len(self.agents)) / len(self.agents)  # Equal weights initially
        
        self.logger.info(f"Initialized ensemble with {len(self.agents)} agents")
    
    def _init_agents(self) -> List[OfflineAgent]:
        """Initialize individual agents in the ensemble."""
        agents = []
        agent_types = {
            "CQL": CQLAgent,
            "IQL": IQLAgent,
            "TD3BC": TD3BCAgent,
        }
        
        for i, config in enumerate(self.agent_configs):
            agent_type = config.pop("type", "CQL")
            if agent_type not in agent_types:
                raise ValueError(f"Unknown agent type: {agent_type}")
            
            # Add common parameters
            config.update({
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "safety_critic": self.safety_critic,
                "constraint_threshold": self.constraint_threshold,
                "seed": self.key + i,  # Different seed for each agent
            })
            
            agent_class = agent_types[agent_type]
            agent = agent_class(**config)
            agents.append(agent)
            
            self.logger.info(f"Initialized agent {i}: {agent_type}")
        
        return agents
    
    def _init_networks(self) -> Dict[str, Any]:
        """Initialize networks (handled by individual agents)."""
        return {"ensemble": "initialized"}
    
    def _create_train_step(self):
        """Create training step (handled by individual agents)."""
        return lambda state, batch: (state, {})
    
    def _update_step(
        self, 
        state: Dict[str, Any], 
        batch: Dict[str, Array]
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Update all agents in the ensemble."""
        total_metrics = {}
        
        for i, agent in enumerate(self.agents):
            try:
                agent_state, agent_metrics = agent._update_step(agent.state, batch)
                agent.state = agent_state
                
                # Prefix metrics with agent index
                for key, value in agent_metrics.items():
                    total_metrics[f"agent_{i}_{key}"] = value
                    
            except Exception as e:
                self.logger.error(f"Agent {i} update failed: {e}")
                # Continue with other agents
                continue
        
        # Compute ensemble-level metrics
        if total_metrics:
            # Average loss across agents
            losses = [v for k, v in total_metrics.items() if "loss" in k]
            if losses:
                total_metrics["ensemble_avg_loss"] = np.mean(losses)
        
        return state, total_metrics
    
    def train(
        self,
        dataset: Dict[str, Array],
        n_epochs: int = 100,
        batch_size: int = 256,
        eval_env: Optional[Any] = None,
        eval_freq: int = 10,
        use_mlflow: bool = False,
    ) -> Dict[str, Any]:
        """Train all agents in the ensemble."""
        self.logger.info(f"Training ensemble of {len(self.agents)} agents")
        
        all_metrics = []
        
        # Train each agent
        for i, agent in enumerate(self.agents):
            self.logger.info(f"Training agent {i}/{len(self.agents)}")
            
            try:
                agent_metrics = agent.train(
                    dataset=dataset,
                    n_epochs=n_epochs,
                    batch_size=batch_size,
                    eval_env=eval_env,
                    eval_freq=eval_freq,
                    use_mlflow=use_mlflow and i == 0,  # Log only first agent to avoid clutter
                )
                all_metrics.append(agent_metrics)
                
                self.logger.info(f"Agent {i} training completed successfully")
                
            except Exception as e:
                self.logger.error(f"Agent {i} training failed: {e}")
                # Continue with other agents
                continue
        
        # Update training state
        self.is_trained = any(agent.is_trained for agent in self.agents)
        
        # Compute ensemble weights based on performance
        if all_metrics and eval_env is not None:
            self._update_ensemble_weights(all_metrics)
        
        return {
            "individual_metrics": all_metrics,
            "ensemble_weights": self.weights.tolist(),
            "trained_agents": sum(1 for agent in self.agents if agent.is_trained),
        }
    
    def _update_ensemble_weights(self, metrics_list: List[Dict[str, Any]]) -> None:
        """Update ensemble weights based on individual agent performance."""
        if not metrics_list:
            return
        
        performances = []
        for metrics in metrics_list:
            if "final_metrics" in metrics and "eval_return_mean" in metrics["final_metrics"]:
                perf = metrics["final_metrics"]["eval_return_mean"]
                performances.append(perf)
            else:
                performances.append(0.0)  # Default for failed agents
        
        if len(performances) == len(self.agents):
            # Softmax weighting based on performance
            performances = np.array(performances)
            if np.std(performances) > 0:
                exp_perf = np.exp((performances - np.max(performances)) / 0.1)
                self.weights = exp_perf / np.sum(exp_perf)
            
            self.logger.info(f"Updated ensemble weights: {self.weights}")
    
    def _predict_impl(
        self, 
        observations: StateArray, 
        deterministic: bool = True
    ) -> ActionArray:
        """Predict actions using ensemble of agents."""
        if not self.is_trained:
            raise RuntimeError("Ensemble must be trained before prediction")
        
        trained_agents = [agent for agent in self.agents if agent.is_trained]
        if not trained_agents:
            raise RuntimeError("No trained agents in ensemble")
        
        # Get predictions from all trained agents
        predictions = []
        for agent in trained_agents:
            try:
                pred = agent.predict(observations, deterministic=deterministic)
                predictions.append(pred)
            except Exception as e:
                self.logger.warning(f"Agent prediction failed: {e}")
                # Skip this agent
                continue
        
        if not predictions:
            raise RuntimeError("All agent predictions failed")
        
        predictions = np.array(predictions)
        
        # Combine predictions based on ensemble method
        if self.ensemble_method == "mean":
            # Weighted average
            active_weights = self.weights[:len(predictions)]
            active_weights = active_weights / np.sum(active_weights)  # Renormalize
            
            ensemble_action = np.average(predictions, axis=0, weights=active_weights)
            
        elif self.ensemble_method == "voting":
            # Majority voting (for discrete actions, use mean for continuous)
            ensemble_action = np.mean(predictions, axis=0)
            
        elif self.ensemble_method == "weighted":
            # Performance-weighted combination
            ensemble_action = np.average(predictions, axis=0, weights=self.weights[:len(predictions)])
            
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return ensemble_action
    
    def predict_with_uncertainty(
        self,
        observations: StateArray,
        return_individual: bool = False,
    ) -> Union[Tuple[ActionArray, Array], Tuple[ActionArray, Array, List[ActionArray]]]:
        """Predict actions with uncertainty quantification.
        
        Args:
            observations: Input observations
            return_individual: Whether to return individual predictions
            
        Returns:
            Tuple of (ensemble_actions, uncertainties) or 
            (ensemble_actions, uncertainties, individual_predictions)
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble must be trained before prediction")
        
        trained_agents = [agent for agent in self.agents if agent.is_trained]
        if len(trained_agents) < 2:
            self.logger.warning("Need at least 2 trained agents for uncertainty quantification")
        
        # Get predictions from all trained agents
        predictions = []
        for agent in trained_agents:
            try:
                pred = agent.predict(observations, deterministic=True)
                predictions.append(pred)
            except Exception as e:
                self.logger.warning(f"Agent prediction failed: {e}")
                continue
        
        if len(predictions) < 2:
            # Fallback to single agent prediction
            if predictions:
                action = predictions[0]
                uncertainty = np.zeros(action.shape[0]) if len(action.shape) > 1 else 0.0
            else:
                raise RuntimeError("No successful predictions")
        else:
            predictions = np.array(predictions)
            
            # Ensemble prediction (weighted mean)
            active_weights = self.weights[:len(predictions)]
            active_weights = active_weights / np.sum(active_weights)
            
            action = np.average(predictions, axis=0, weights=active_weights)
            
            # Uncertainty as prediction variance
            if len(predictions.shape) == 3:  # Batch of observations
                uncertainty = np.std(predictions, axis=0)
                uncertainty = np.mean(uncertainty, axis=1)  # Average across action dimensions
            else:  # Single observation
                uncertainty = np.std(predictions, axis=0)
                uncertainty = np.mean(uncertainty)  # Average across action dimensions
        
        if return_individual:
            return action, uncertainty, predictions
        else:
            return action, uncertainty
    
    def get_high_uncertainty_mask(
        self,
        observations: StateArray,
        threshold: Optional[float] = None,
    ) -> Array:
        """Identify observations with high prediction uncertainty.
        
        Args:
            observations: Input observations
            threshold: Uncertainty threshold (uses default if None)
            
        Returns:
            Boolean mask indicating high uncertainty observations
        """
        _, uncertainties = self.predict_with_uncertainty(observations)
        threshold = threshold or self.uncertainty_threshold
        
        return uncertainties > threshold
    
    def evaluate_diversity(self, observations: StateArray) -> Dict[str, float]:
        """Evaluate diversity of predictions across ensemble.
        
        Args:
            observations: Input observations
            
        Returns:
            Dictionary with diversity metrics
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble must be trained")
        
        trained_agents = [agent for agent in self.agents if agent.is_trained]
        if len(trained_agents) < 2:
            return {"diversity_score": 0.0, "disagreement": 0.0}
        
        predictions = []
        for agent in trained_agents:
            try:
                pred = agent.predict(observations, deterministic=True)
                predictions.append(pred)
            except Exception:
                continue
        
        if len(predictions) < 2:
            return {"diversity_score": 0.0, "disagreement": 0.0}
        
        predictions = np.array(predictions)
        
        # Calculate pairwise distances between predictions
        n_agents = len(predictions)
        pairwise_distances = []
        
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                dist = np.mean(np.abs(predictions[i] - predictions[j]))
                pairwise_distances.append(dist)
        
        diversity_score = np.mean(pairwise_distances)
        disagreement = np.std(predictions, axis=0).mean()
        
        return {
            "diversity_score": float(diversity_score),
            "disagreement": float(disagreement),
            "n_agents": len(predictions),
        }
