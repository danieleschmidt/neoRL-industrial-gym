"""Safety-critical offline RL agents with advanced constraint handling."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state

from ..core.types import SafetyConstraint, SafetyMetrics
from .base import OfflineAgent
from .cql import CQLAgent
from .iql import IQLAgent
from .networks import create_critic_network, create_actor_network


class SafetyAwareAgent(OfflineAgent, ABC):
    """Abstract base class for safety-aware offline RL agents."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        safety_constraints: List[SafetyConstraint],
        safety_penalty_weight: float = 1.0,
        uncertainty_threshold: float = 0.1,
        **kwargs
    ):
        super().__init__(state_dim, action_dim, **kwargs)
        self.safety_constraints = safety_constraints
        self.safety_penalty_weight = safety_penalty_weight
        self.uncertainty_threshold = uncertainty_threshold
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize safety-specific components
        self._setup_safety_components()
        
    @abstractmethod
    def _setup_safety_components(self) -> None:
        """Setup safety-specific neural networks and components."""
        pass
        
    @abstractmethod
    def compute_safety_violation_probability(
        self, 
        state: jnp.ndarray, 
        action: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute probability of safety constraint violation."""
        pass
        
    @abstractmethod
    def get_safe_action(
        self, 
        state: jnp.ndarray,
        preferred_action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Get safety-corrected action with metadata."""
        pass


class RiskAwareCQLAgent(CQLAgent, SafetyAwareAgent):
    """Conservative Q-Learning with distributional safety risk modeling."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        safety_constraints: List[SafetyConstraint],
        risk_quantile: float = 0.95,
        distributional_atoms: int = 51,
        **kwargs
    ):
        self.risk_quantile = risk_quantile
        self.distributional_atoms = distributional_atoms
        
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            safety_constraints=safety_constraints,
            **kwargs
        )
        
    def _setup_safety_components(self) -> None:
        """Setup distributional safety critic and risk estimator."""
        
        # Distributional safety critic for risk assessment
        class DistributionalSafetyCritic(nn.Module):
            hidden_dims: Tuple[int, ...] = (256, 256)
            atoms: int = 51
            
            def setup(self):
                self.layers = [nn.Dense(dim) for dim in self.hidden_dims]
                self.value_head = nn.Dense(self.atoms)
                
            def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
                x = jnp.concatenate([state, action], axis=-1)
                
                for layer in self.layers:
                    x = nn.relu(layer(x))
                    
                # Output distribution over safety values
                logits = self.value_head(x)
                return nn.softmax(logits, axis=-1)
                
        self.safety_critic = DistributionalSafetyCritic(atoms=self.distributional_atoms)
        
        # Risk estimation network
        class RiskEstimator(nn.Module):
            hidden_dims: Tuple[int, ...] = (128, 128)
            
            def setup(self):
                self.layers = [nn.Dense(dim) for dim in self.hidden_dims]
                self.risk_head = nn.Dense(1)
                
            def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
                x = jnp.concatenate([state, action], axis=-1)
                
                for layer in self.layers:
                    x = nn.relu(layer(x))
                    
                # Output risk probability [0, 1]
                return nn.sigmoid(self.risk_head(x))
                
        self.risk_estimator = RiskEstimator()
        
        # Initialize safety critic parameters
        key = jax.random.PRNGKey(42)
        dummy_state = jnp.ones((1, self.state_dim))
        dummy_action = jnp.ones((1, self.action_dim))
        
        safety_params = self.safety_critic.init(key, dummy_state, dummy_action)
        risk_params = self.risk_estimator.init(key, dummy_state, dummy_action)
        
        self.safety_state = train_state.TrainState.create(
            apply_fn=self.safety_critic.apply,
            params=safety_params,
            tx=optax.adam(3e-4)
        )
        
        self.risk_state = train_state.TrainState.create(
            apply_fn=self.risk_estimator.apply,
            params=risk_params,
            tx=optax.adam(3e-4)
        )
        
    def compute_safety_violation_probability(
        self, 
        state: jnp.ndarray, 
        action: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute probability of safety violation using distributional critic."""
        
        # Get safety value distribution
        safety_dist = self.safety_state.apply_fn(
            self.safety_state.params, state, action
        )
        
        # Define safety threshold (negative values indicate violations)
        safety_threshold = 0.0
        
        # Compute probability mass below threshold
        atoms = jnp.linspace(-1.0, 1.0, self.distributional_atoms)
        violation_mask = atoms < safety_threshold
        violation_prob = jnp.sum(safety_dist * violation_mask, axis=-1)
        
        return violation_prob
        
    def get_safe_action(
        self, 
        state: jnp.ndarray,
        preferred_action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Get safety-corrected action using risk-aware optimization."""
        
        # Compute risk for preferred action
        preferred_risk = self.risk_state.apply_fn(
            self.risk_state.params, state, preferred_action
        )
        
        # If risk is acceptable, return preferred action
        if preferred_risk < self.uncertainty_threshold:
            return preferred_action, {"risk": float(preferred_risk), "corrected": False}
            
        # Otherwise, search for safer action
        key = jax.random.PRNGKey(int(jnp.sum(state) * 1000))
        
        # Sample multiple candidate actions
        n_candidates = 100
        action_candidates = jax.random.uniform(
            key, (n_candidates, self.action_dim), minval=-1.0, maxval=1.0
        )
        
        # Evaluate risk for all candidates
        candidate_risks = jax.vmap(
            lambda a: self.risk_state.apply_fn(self.risk_state.params, state, a)
        )(action_candidates)
        
        # Select action with lowest risk
        safest_idx = jnp.argmin(candidate_risks)
        safe_action = action_candidates[safest_idx]
        final_risk = candidate_risks[safest_idx]
        
        return safe_action, {"risk": float(final_risk), "corrected": True}
        
    def update_safety_critic(
        self, 
        batch: Dict[str, jnp.ndarray],
        safety_labels: jnp.ndarray
    ) -> Dict[str, float]:
        """Update distributional safety critic with safety violation labels."""
        
        def safety_loss_fn(params):
            # Predict safety distribution
            pred_dist = self.safety_critic.apply(
                params, batch["observations"], batch["actions"]
            )
            
            # Convert safety labels to target distribution
            atoms = jnp.linspace(-1.0, 1.0, self.distributional_atoms)
            target_dist = jnp.exp(-jnp.abs(atoms - safety_labels[:, None]) / 0.1)
            target_dist = target_dist / jnp.sum(target_dist, axis=-1, keepdims=True)
            
            # Cross-entropy loss
            loss = -jnp.sum(target_dist * jnp.log(pred_dist + 1e-8))
            return loss
            
        grad_fn = jax.value_and_grad(safety_loss_fn)
        loss, grads = grad_fn(self.safety_state.params)
        
        self.safety_state = self.safety_state.apply_gradients(grads=grads)
        
        return {"safety_critic_loss": float(loss)}


class ConstrainedIQLAgent(IQLAgent, SafetyAwareAgent):
    """Implicit Q-Learning with hard safety constraints via Lagrangian methods."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        safety_constraints: List[SafetyConstraint],
        constraint_tolerance: float = 0.01,
        lagrange_lr: float = 1e-3,
        **kwargs
    ):
        self.constraint_tolerance = constraint_tolerance
        self.lagrange_lr = lagrange_lr
        
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            safety_constraints=safety_constraints,
            **kwargs
        )
        
    def _setup_safety_components(self) -> None:
        """Setup Lagrangian multipliers and constraint networks."""
        
        # Lagrangian multipliers for each constraint
        self.lagrange_multipliers = jnp.ones(len(self.safety_constraints))
        
        # Constraint violation predictor
        class ConstraintPredictor(nn.Module):
            hidden_dims: Tuple[int, ...] = (128, 128)
            n_constraints: int = 1
            
            def setup(self):
                self.layers = [nn.Dense(dim) for dim in self.hidden_dims]
                self.constraint_head = nn.Dense(self.n_constraints)
                
            def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
                x = jnp.concatenate([state, action], axis=-1)
                
                for layer in self.layers:
                    x = nn.relu(layer(x))
                    
                # Output constraint violation predictions
                return self.constraint_head(x)
                
        self.constraint_predictor = ConstraintPredictor(
            n_constraints=len(self.safety_constraints)
        )
        
        # Initialize constraint predictor
        key = jax.random.PRNGKey(42)
        dummy_state = jnp.ones((1, self.state_dim))
        dummy_action = jnp.ones((1, self.action_dim))
        
        constraint_params = self.constraint_predictor.init(key, dummy_state, dummy_action)
        
        self.constraint_state = train_state.TrainState.create(
            apply_fn=self.constraint_predictor.apply,
            params=constraint_params,
            tx=optax.adam(3e-4)
        )
        
    def compute_safety_violation_probability(
        self, 
        state: jnp.ndarray, 
        action: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute constraint violation probabilities."""
        
        violations = self.constraint_state.apply_fn(
            self.constraint_state.params, state, action
        )
        
        # Convert to probabilities using sigmoid
        return nn.sigmoid(violations)
        
    def get_safe_action(
        self, 
        state: jnp.ndarray,
        preferred_action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Get action satisfying hard constraints via projection."""
        
        # Check if preferred action satisfies constraints
        violation_probs = self.compute_safety_violation_probability(state, preferred_action)
        
        if jnp.all(violation_probs < self.uncertainty_threshold):
            return preferred_action, {"violations": violation_probs, "projected": False}
            
        # Project to constraint-satisfying region
        safe_action = self._project_to_safe_region(state, preferred_action)
        final_violations = self.compute_safety_violation_probability(state, safe_action)
        
        return safe_action, {"violations": final_violations, "projected": True}
        
    def _project_to_safe_region(
        self, 
        state: jnp.ndarray, 
        action: jnp.ndarray
    ) -> jnp.ndarray:
        """Project action to constraint-satisfying region using gradient descent."""
        
        current_action = action
        
        for _ in range(10):  # Maximum projection iterations
            # Compute constraint violations
            violations = self.constraint_state.apply_fn(
                self.constraint_state.params, state, current_action
            )
            
            # Check if constraints are satisfied
            if jnp.all(violations < self.constraint_tolerance):
                break
                
            # Compute gradients of constraints w.r.t. action
            grad_fn = jax.grad(
                lambda a: jnp.sum(nn.relu(self.constraint_state.apply_fn(
                    self.constraint_state.params, state, a
                )))
            )
            
            constraint_grad = grad_fn(current_action)
            
            # Gradient descent step
            current_action = current_action - 0.1 * constraint_grad
            
            # Clip to action bounds
            current_action = jnp.clip(current_action, -1.0, 1.0)
            
        return current_action
        
    def update_lagrange_multipliers(
        self, 
        constraint_violations: jnp.ndarray
    ) -> Dict[str, float]:
        """Update Lagrangian multipliers based on constraint violations."""
        
        # Update multipliers
        violation_excess = constraint_violations - self.constraint_tolerance
        self.lagrange_multipliers = jnp.maximum(
            0.0,
            self.lagrange_multipliers + self.lagrange_lr * violation_excess
        )
        
        return {
            "avg_multiplier": float(jnp.mean(self.lagrange_multipliers)),
            "max_violation": float(jnp.max(constraint_violations))
        }


class SafeEnsembleAgent(SafetyAwareAgent):
    """Ensemble agent with uncertainty-calibrated safety predictions."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        safety_constraints: List[SafetyConstraint],
        n_models: int = 5,
        uncertainty_calibration: str = "temperature_scaling",
        **kwargs
    ):
        self.n_models = n_models
        self.uncertainty_calibration = uncertainty_calibration
        
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            safety_constraints=safety_constraints,
            **kwargs
        )
        
    def _setup_safety_components(self) -> None:
        """Setup ensemble of safety predictors with uncertainty calibration."""
        
        class SafetyEnsembleMember(nn.Module):
            hidden_dims: Tuple[int, ...] = (128, 128)
            n_constraints: int = 1
            
            def setup(self):
                self.layers = [nn.Dense(dim) for dim in self.hidden_dims]
                self.safety_head = nn.Dense(self.n_constraints)
                
            def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
                x = jnp.concatenate([state, action], axis=-1)
                
                for layer in self.layers:
                    x = nn.relu(layer(x))
                    
                return self.safety_head(x)
                
        # Create ensemble of safety predictors
        self.ensemble_members = [
            SafetyEnsembleMember(n_constraints=len(self.safety_constraints))
            for _ in range(self.n_models)
        ]
        
        # Initialize ensemble parameters
        key = jax.random.PRNGKey(42)
        dummy_state = jnp.ones((1, self.state_dim))
        dummy_action = jnp.ones((1, self.action_dim))
        
        self.ensemble_states = []
        
        for i, member in enumerate(self.ensemble_members):
            member_key = jax.random.split(key, self.n_models)[i]
            params = member.init(member_key, dummy_state, dummy_action)
            
            state = train_state.TrainState.create(
                apply_fn=member.apply,
                params=params,
                tx=optax.adam(3e-4)
            )
            
            self.ensemble_states.append(state)
            
        # Temperature scaling for uncertainty calibration
        self.temperature = 1.0
        
    def compute_safety_violation_probability(
        self, 
        state: jnp.ndarray, 
        action: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute safety violation probability with uncertainty quantification."""
        
        # Get predictions from all ensemble members
        predictions = []
        
        for ensemble_state in self.ensemble_states:
            pred = ensemble_state.apply_fn(ensemble_state.params, state, action)
            predictions.append(pred)
            
        predictions = jnp.array(predictions)
        
        # Compute ensemble statistics
        mean_pred = jnp.mean(predictions, axis=0)
        std_pred = jnp.std(predictions, axis=0)
        
        # Apply temperature scaling for calibration
        calibrated_pred = mean_pred / self.temperature
        
        # Convert to probabilities with uncertainty penalty
        base_prob = nn.sigmoid(calibrated_pred)
        uncertainty_penalty = jnp.minimum(std_pred, 1.0)  # Cap uncertainty
        
        # Higher uncertainty increases violation probability (conservative)
        final_prob = base_prob + 0.5 * uncertainty_penalty
        
        return jnp.clip(final_prob, 0.0, 1.0)
        
    def get_safe_action(
        self, 
        state: jnp.ndarray,
        preferred_action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Get action with uncertainty-aware safety assessment."""
        
        # Evaluate preferred action
        violation_prob = self.compute_safety_violation_probability(state, preferred_action)
        
        # Compute prediction uncertainty
        predictions = []
        for ensemble_state in self.ensemble_states:
            pred = ensemble_state.apply_fn(ensemble_state.params, state, preferred_action)
            predictions.append(pred)
            
        predictions = jnp.array(predictions)
        uncertainty = jnp.std(predictions, axis=0)
        
        # Decision based on both violation probability and uncertainty
        max_acceptable_prob = self.uncertainty_threshold
        max_acceptable_uncertainty = 0.2
        
        is_safe = jnp.all(violation_prob < max_acceptable_prob)
        is_certain = jnp.all(uncertainty < max_acceptable_uncertainty)
        
        if is_safe and is_certain:
            return preferred_action, {
                "violation_prob": violation_prob,
                "uncertainty": uncertainty,
                "decision": "accept"
            }
        else:
            # Conservative fallback: return safe default action
            safe_action = jnp.zeros_like(preferred_action)  # Stop action
            
            return safe_action, {
                "violation_prob": violation_prob,
                "uncertainty": uncertainty,
                "decision": "reject_conservative"
            }
            
    def calibrate_uncertainty(
        self, 
        validation_data: Dict[str, jnp.ndarray],
        validation_labels: jnp.ndarray
    ) -> Dict[str, float]:
        """Calibrate uncertainty estimation using temperature scaling."""
        
        # Get ensemble predictions on validation data
        all_predictions = []
        
        for ensemble_state in self.ensemble_states:
            preds = ensemble_state.apply_fn(
                ensemble_state.params,
                validation_data["observations"],
                validation_data["actions"]
            )
            all_predictions.append(preds)
            
        mean_predictions = jnp.mean(jnp.array(all_predictions), axis=0)
        
        # Optimize temperature using maximum likelihood
        def temperature_loss(temperature):
            calibrated_logits = mean_predictions / temperature
            log_probs = jax.nn.log_softmax(calibrated_logits, axis=-1)
            return -jnp.mean(jnp.sum(validation_labels * log_probs, axis=-1))
            
        # Simple grid search for temperature
        temperatures = jnp.linspace(0.1, 5.0, 50)
        losses = jax.vmap(temperature_loss)(temperatures)
        
        best_temperature = temperatures[jnp.argmin(losses)]
        self.temperature = float(best_temperature)
        
        return {"optimal_temperature": self.temperature}


# Factory functions for creating safety-critical agents
def create_risk_aware_cql(
    state_dim: int,
    action_dim: int,
    safety_constraints: List[SafetyConstraint],
    **kwargs
) -> RiskAwareCQLAgent:
    """Create Risk-Aware CQL agent."""
    return RiskAwareCQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        safety_constraints=safety_constraints,
        **kwargs
    )


def create_constrained_iql(
    state_dim: int,
    action_dim: int,
    safety_constraints: List[SafetyConstraint],
    **kwargs
) -> ConstrainedIQLAgent:
    """Create Constrained IQL agent."""
    return ConstrainedIQLAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        safety_constraints=safety_constraints,
        **kwargs
    )


def create_safe_ensemble(
    state_dim: int,
    action_dim: int,
    safety_constraints: List[SafetyConstraint],
    **kwargs
) -> SafeEnsembleAgent:
    """Create Safe Ensemble agent."""
    return SafeEnsembleAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        safety_constraints=safety_constraints,
        **kwargs
    )