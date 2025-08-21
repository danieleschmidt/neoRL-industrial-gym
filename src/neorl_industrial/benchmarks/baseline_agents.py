"""Baseline agents for comparison in research studies."""

import numpy as np
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaselineAgent(ABC):
    """Base class for baseline agents."""
    
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    @abstractmethod
    def act(self, state: np.ndarray) -> np.ndarray:
        """Select action given state."""
        pass
        
    def train(self, dataset: Dict[str, np.ndarray], **kwargs) -> Dict[str, Any]:
        """Train the agent (baseline agents may not require training)."""
        return {"training_complete": True}


class RandomAgent(BaselineAgent):
    """Random action baseline agent."""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 action_low: float = -1.0, action_high: float = 1.0):
        super().__init__(state_dim, action_dim)
        self.action_low = action_low
        self.action_high = action_high
        
    def act(self, state: np.ndarray) -> np.ndarray:
        """Return random action."""
        return np.random.uniform(
            self.action_low, 
            self.action_high, 
            size=(self.action_dim,)
        )


class PIDControllerAgent(BaselineAgent):
    """PID Controller baseline for industrial environments."""
    
    def __init__(self, state_dim: int, action_dim: int,
                 kp: float = 1.0, ki: float = 0.1, kd: float = 0.01,
                 setpoint: Optional[np.ndarray] = None):
        super().__init__(state_dim, action_dim)
        self.kp = kp
        self.ki = ki  
        self.kd = kd
        self.setpoint = setpoint if setpoint is not None else np.zeros(action_dim)
        
        # PID state
        self.previous_error = np.zeros(action_dim)
        self.integral = np.zeros(action_dim)
        
    def act(self, state: np.ndarray) -> np.ndarray:
        """PID control action."""
        # Use first action_dim states as process variables
        pv = state[:self.action_dim] if len(state) >= self.action_dim else state
        
        # Calculate error
        error = self.setpoint - pv
        
        # PID terms
        proportional = self.kp * error
        self.integral += error
        integral_term = self.ki * self.integral
        derivative = self.kd * (error - self.previous_error)
        
        # Control output
        action = proportional + integral_term + derivative
        action = np.clip(action, -1.0, 1.0)  # Clip to valid range
        
        self.previous_error = error
        return action


class MPC_Agent(BaselineAgent):
    """Model Predictive Control baseline."""
    
    def __init__(self, state_dim: int, action_dim: int,
                 horizon: int = 10, cost_weights: Optional[np.ndarray] = None):
        super().__init__(state_dim, action_dim)
        self.horizon = horizon
        self.cost_weights = cost_weights if cost_weights is not None else np.ones(action_dim)
        
    def act(self, state: np.ndarray) -> np.ndarray:
        """Simplified MPC action (using heuristics)."""
        # Simplified MPC: Move towards state center
        target_state = np.zeros_like(state)
        state_error = target_state - state
        
        # Simple proportional control with prediction
        action = 0.5 * state_error[:self.action_dim] if len(state) >= self.action_dim else np.zeros(self.action_dim)
        return np.clip(action, -1.0, 1.0)


class ConstantAgent(BaselineAgent):
    """Agent that always outputs constant action."""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 constant_action: Optional[np.ndarray] = None):
        super().__init__(state_dim, action_dim)
        self.constant_action = constant_action if constant_action is not None else np.zeros(action_dim)
        
    def act(self, state: np.ndarray) -> np.ndarray:
        """Return constant action."""
        return self.constant_action.copy()


class BaselineAgentFactory:
    """Factory for creating baseline agents."""
    
    AGENTS = {
        "random": RandomAgent,
        "pid": PIDControllerAgent,
        "mpc": MPC_Agent,
        "constant": ConstantAgent,
    }
    
    @classmethod
    def create(cls, agent_type: str, state_dim: int, action_dim: int, 
               **kwargs) -> BaselineAgent:
        """Create baseline agent of specified type."""
        if agent_type not in cls.AGENTS:
            raise ValueError(f"Unknown baseline agent type: {agent_type}. "
                           f"Available: {list(cls.AGENTS.keys())}")
        
        agent_class = cls.AGENTS[agent_type]
        return agent_class(state_dim, action_dim, **kwargs)
    
    @classmethod
    def create_all_baselines(cls, state_dim: int, action_dim: int) -> Dict[str, BaselineAgent]:
        """Create all baseline agents for comparison."""
        baselines = {}
        
        # Random baseline
        baselines["Random"] = cls.create("random", state_dim, action_dim)
        
        # PID baseline with different tunings
        baselines["PID_Conservative"] = cls.create("pid", state_dim, action_dim, 
                                                  kp=0.5, ki=0.05, kd=0.005)
        baselines["PID_Aggressive"] = cls.create("pid", state_dim, action_dim,
                                                kp=2.0, ki=0.2, kd=0.02)
        
        # MPC baseline
        baselines["MPC"] = cls.create("mpc", state_dim, action_dim, horizon=5)
        
        # Constant baselines
        baselines["Zero_Action"] = cls.create("constant", state_dim, action_dim)
        
        return baselines
    
    @classmethod
    def benchmark_baselines(cls, baselines: Dict[str, BaselineAgent],
                           environment, n_episodes: int = 10) -> Dict[str, Dict[str, float]]:
        """Benchmark all baseline agents."""
        results = {}
        
        for name, agent in baselines.items():
            logger.info(f"Benchmarking baseline: {name}")
            
            episode_returns = []
            episode_lengths = []
            safety_violations = []
            
            for episode in range(n_episodes):
                state = environment.reset()
                if isinstance(state, tuple):
                    state = state[0]
                    
                total_return = 0
                length = 0
                violations = 0
                done = False
                
                while not done and length < 1000:  # Max episode length
                    action = agent.act(state)
                    next_state, reward, done, info = environment.step(action)
                    
                    if isinstance(next_state, tuple):
                        next_state = next_state[0]
                    if isinstance(done, tuple):
                        done = done[0] if len(done) > 0 else True
                    
                    total_return += reward
                    length += 1
                    
                    # Check for safety violations
                    if hasattr(info, 'get') and info.get('safety_violation', False):
                        violations += 1
                    
                    state = next_state
                
                episode_returns.append(total_return)
                episode_lengths.append(length)
                safety_violations.append(violations)
            
            results[name] = {
                "mean_return": float(np.mean(episode_returns)),
                "std_return": float(np.std(episode_returns)),
                "mean_length": float(np.mean(episode_lengths)),
                "mean_violations": float(np.mean(safety_violations)),
                "success_rate": float(np.mean([r > 0 for r in episode_returns]))
            }
            
        return results