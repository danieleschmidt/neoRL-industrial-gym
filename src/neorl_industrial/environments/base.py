"""Base industrial environment with safety constraints."""

import abc
import gymnasium as gym
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

from ..core.types import (
    Array,
    IndustrialState, 
    SafetyConstraint,
    SafetyMetrics,
    StateArray,
    ActionArray,
)


class IndustrialEnv(gym.Env, abc.ABC):
    """Base class for industrial control environments with safety constraints."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        safety_constraints: Optional[List[SafetyConstraint]] = None,
        max_episode_steps: int = 1000,
        dt: float = 0.1,
    ):
        """Initialize industrial environment.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            safety_constraints: List of safety constraints
            max_episode_steps: Maximum steps per episode
            dt: Time step size
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_episode_steps = max_episode_steps
        self.dt = dt
        
        # Safety constraints
        self.safety_constraints = safety_constraints or []
        
        # Environment state
        self.current_step = 0
        self.state = None
        self.done = False
        self.info = {}
        
        # Safety tracking
        self.violation_count = 0
        self.total_violations = 0
        
        # Define spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(state_dim,),
            dtype=np.float32
        )
        
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )
    
    @abc.abstractmethod
    def _get_initial_state(self) -> StateArray:
        """Get initial state for environment."""
        pass
    
    @abc.abstractmethod 
    def _dynamics(self, state: StateArray, action: ActionArray) -> StateArray:
        """Compute next state given current state and action."""
        pass
    
    @abc.abstractmethod
    def _compute_reward(self, state: StateArray, action: ActionArray, next_state: StateArray) -> float:
        """Compute reward for transition."""
        pass
    
    @abc.abstractmethod
    def _is_done(self, state: StateArray) -> bool:
        """Check if episode should terminate."""
        pass
    
    def _check_safety_constraints(self, state: StateArray, action: ActionArray) -> SafetyMetrics:
        """Evaluate all safety constraints."""
        satisfied = 0
        violations = 0
        critical_violations = 0
        
        for constraint in self.safety_constraints:
            try:
                is_safe = constraint.check_fn(state, action)
                if is_safe:
                    satisfied += 1
                else:
                    violations += 1
                    if constraint.critical:
                        critical_violations += 1
            except Exception:
                # Conservative: assume constraint violated on error
                violations += 1
                if constraint.critical:
                    critical_violations += 1
        
        total_constraints = len(self.safety_constraints)
        safety_score = satisfied / total_constraints if total_constraints > 0 else 1.0
        
        return SafetyMetrics(
            constraints_satisfied=satisfied,
            total_constraints=total_constraints,
            violation_count=violations,
            critical_violations=critical_violations,
            safety_score=safety_score
        )
    
    def _get_safety_info(self, state: StateArray) -> Dict[str, Any]:
        """Extract safety-related information from state."""
        return {
            "safety_metrics": {},
            "constraint_values": {},
        }
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Array, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.done = False
        self.violation_count = 0
        
        # Get initial state
        self.state = self._get_initial_state()
        
        # Create observation
        obs = self.state.copy()
        
        # Get safety info
        info = self._get_safety_info(self.state)
        info.update({
            "step": self.current_step,
            "violations": self.violation_count,
            "total_violations": self.total_violations,
        })
        
        return obs, info
    
    def step(self, action: ActionArray) -> Tuple[Array, float, bool, bool, Dict]:
        """Execute one time step."""
        if self.done:
            raise RuntimeError("Environment is done. Call reset() first.")
        
        # Convert to numpy if needed
        if isinstance(action, jnp.ndarray):
            action = np.array(action)
        
        # Clip action to bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # Check safety constraints
        safety_metrics = self._check_safety_constraints(self.state, action)
        
        # Apply dynamics
        next_state = self._dynamics(self.state, action)
        
        # Compute reward (include safety penalty)
        reward = self._compute_reward(self.state, action, next_state)
        
        # Apply safety penalties
        for constraint in self.safety_constraints:
            if not constraint.check_fn(self.state, action):
                reward += constraint.penalty
                self.violation_count += 1
                self.total_violations += 1
        
        # Update state
        self.state = next_state
        self.current_step += 1
        
        # Check termination conditions
        terminated = self._is_done(self.state)
        truncated = self.current_step >= self.max_episode_steps
        self.done = terminated or truncated
        
        # Emergency shutdown for critical safety violations
        if safety_metrics.critical_violations > 0:
            terminated = True
            self.done = True
            reward -= 1000.0  # Large penalty for critical violations
        
        # Create observation
        obs = self.state.copy()
        
        # Collect info
        info = self._get_safety_info(self.state)
        info.update({
            "step": self.current_step,
            "violations": self.violation_count,
            "total_violations": self.total_violations,
            "safety_metrics": safety_metrics,
            "critical_shutdown": safety_metrics.critical_violations > 0,
        })
        
        return obs, reward, terminated, truncated, info
    
    @abc.abstractmethod
    def get_dataset(self, quality: str = "mixed") -> Dict[str, Array]:
        """Load offline dataset for this environment."""
        pass
    
    def add_safety_constraint(self, constraint: SafetyConstraint) -> None:
        """Add a safety constraint to the environment."""
        self.safety_constraints.append(constraint)
    
    def remove_safety_constraint(self, name: str) -> None:
        """Remove a safety constraint by name."""
        self.safety_constraints = [
            c for c in self.safety_constraints if c.name != name
        ]