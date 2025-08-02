"""Industrial test data fixtures and generators."""

import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class IndustrialDataset:
    """Container for industrial dataset components."""
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminals: np.ndarray
    timeouts: np.ndarray
    safety_violations: np.ndarray
    metadata: Dict[str, Any]


class IndustrialDataGenerator:
    """Generate realistic industrial test data."""
    
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    
    def generate_chemical_reactor_data(self, n_samples: int = 1000) -> IndustrialDataset:
        """Generate chemical reactor process data."""
        state_dim = 12  # [temp, pressure, pH, flow_rate, concentration, ...]
        action_dim = 3  # [heating, valve_position, catalyst_feed]
        
        # Generate correlated states (realistic process dynamics)
        observations = self._generate_correlated_states(n_samples, state_dim, "chemical")
        actions = self._generate_control_actions(observations, action_dim, "chemical")
        next_observations = self._simulate_dynamics(observations, actions, "chemical")
        rewards = self._compute_rewards(observations, actions, "chemical")
        terminals = self._generate_episode_boundaries(n_samples, avg_length=200)
        timeouts = self.rng.choice([False, True], size=n_samples, p=[0.95, 0.05])
        safety_violations = self._check_safety_violations(observations, "chemical")
        
        metadata = {
            "environment": "ChemicalReactor-v0",
            "state_features": ["temperature", "pressure", "pH", "flow_rate", 
                             "concentration", "level", "agitator_speed", "cooling_rate",
                             "feed_rate", "product_quality", "viscosity", "density"],
            "action_features": ["heating_power", "valve_position", "catalyst_feed_rate"],
            "safety_constraints": ["temperature < 350K", "pressure < 5bar", "pH > 6.5"],
            "quality": "expert",
            "n_episodes": np.sum(terminals)
        }
        
        return IndustrialDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
            timeouts=timeouts,
            safety_violations=safety_violations,
            metadata=metadata
        )
    
    def generate_robot_assembly_data(self, n_samples: int = 1000) -> IndustrialDataset:
        """Generate robot assembly line data."""
        state_dim = 24  # Joint positions, velocities, forces, vision features
        action_dim = 7   # Joint torques
        
        observations = self._generate_correlated_states(n_samples, state_dim, "robotics")
        actions = self._generate_control_actions(observations, action_dim, "robotics")
        next_observations = self._simulate_dynamics(observations, actions, "robotics")
        rewards = self._compute_rewards(observations, actions, "robotics")
        terminals = self._generate_episode_boundaries(n_samples, avg_length=150)
        timeouts = self.rng.choice([False, True], size=n_samples, p=[0.98, 0.02])
        safety_violations = self._check_safety_violations(observations, "robotics")
        
        metadata = {
            "environment": "RobotAssembly-v0",
            "state_features": [f"joint_{i}_pos" for i in range(7)] + 
                            [f"joint_{i}_vel" for i in range(7)] +
                            ["gripper_force", "vision_x", "vision_y", "vision_z"] +
                            [f"force_sensor_{i}" for i in range(6)],
            "action_features": [f"joint_{i}_torque" for i in range(7)],
            "safety_constraints": ["force < 100N", "joint_limits", "collision_free"],
            "quality": "medium",
            "n_episodes": np.sum(terminals)
        }
        
        return IndustrialDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
            timeouts=timeouts,
            safety_violations=safety_violations,
            metadata=metadata
        )
    
    def _generate_correlated_states(self, n_samples: int, state_dim: int, 
                                  domain: str) -> np.ndarray:
        """Generate realistic correlated state sequences."""
        if domain == "chemical":
            # Chemical processes have smooth, correlated dynamics
            base_states = self.rng.randn(n_samples, state_dim) * 0.1
            # Add process correlations
            for i in range(1, n_samples):
                base_states[i] = 0.95 * base_states[i-1] + 0.05 * base_states[i]
            
            # Add realistic scales and offsets
            scales = np.array([100, 2, 7, 10, 0.5, 50, 1000, 5, 2, 0.8, 1.2, 1000])[:state_dim]
            offsets = np.array([300, 1, 7.5, 5, 0.3, 25, 500, 2, 1, 0.9, 1.0, 800])[:state_dim]
            
        elif domain == "robotics":
            # Robot states have position/velocity structure
            base_states = self.rng.randn(n_samples, state_dim) * 0.05
            # Add kinematic correlations
            for i in range(1, n_samples):
                base_states[i] = 0.98 * base_states[i-1] + 0.02 * base_states[i]
            
            # Realistic joint ranges and sensor scales
            scales = np.concatenate([
                np.ones(7) * np.pi,      # Joint positions (radians)
                np.ones(7) * 2.0,        # Joint velocities (rad/s)
                [50.0],                  # Gripper force (N)
                [0.5, 0.5, 0.3],        # Vision coordinates (m)
                np.ones(6) * 10.0        # Force sensors (N)
            ])[:state_dim]
            offsets = np.zeros(state_dim)
            
        else:  # Generic case
            base_states = self.rng.randn(n_samples, state_dim) * 0.1
            scales = np.ones(state_dim)
            offsets = np.zeros(state_dim)
        
        return base_states * scales + offsets
    
    def _generate_control_actions(self, states: np.ndarray, action_dim: int, 
                                domain: str) -> np.ndarray:
        """Generate realistic control actions based on states."""
        n_samples = states.shape[0]
        
        if domain == "chemical":
            # PID-like control responses
            actions = np.zeros((n_samples, action_dim))
            
            # Heating control based on temperature error
            temp_setpoint = 320.0  # Target temperature
            temp_error = temp_setpoint - states[:, 0]
            actions[:, 0] = np.clip(0.1 * temp_error + self.rng.randn(n_samples) * 0.02, 0, 1)
            
            # Valve control based on pressure
            pressure_setpoint = 2.0
            pressure_error = pressure_setpoint - states[:, 1]
            actions[:, 1] = np.clip(0.5 + 0.05 * pressure_error + self.rng.randn(n_samples) * 0.01, 0, 1)
            
            # Catalyst feed based on concentration
            conc_setpoint = 0.4
            conc_error = conc_setpoint - states[:, 4] if states.shape[1] > 4 else 0
            actions[:, 2] = np.clip(0.2 * conc_error + self.rng.randn(n_samples) * 0.005, 0, 0.1)
            
        elif domain == "robotics":
            # Joint torque control
            actions = np.zeros((n_samples, action_dim))
            
            # Simple position control for each joint
            for i in range(min(action_dim, 7)):
                if states.shape[1] > i:
                    position = states[:, i]
                    velocity = states[:, i + 7] if states.shape[1] > i + 7 else np.zeros(n_samples)
                    
                    # PD control
                    target_pos = 0.0  # Target position
                    kp, kd = 10.0, 1.0
                    actions[:, i] = -kp * (position - target_pos) - kd * velocity
                    actions[:, i] += self.rng.randn(n_samples) * 0.1  # Control noise
                    actions[:, i] = np.clip(actions[:, i], -5.0, 5.0)  # Torque limits
            
        else:  # Generic case
            actions = self.rng.randn(n_samples, action_dim) * 0.1
        
        return actions.astype(np.float32)
    
    def _simulate_dynamics(self, states: np.ndarray, actions: np.ndarray, 
                         domain: str) -> np.ndarray:
        """Simulate basic system dynamics."""
        next_states = states.copy()
        
        # Add action effects and noise
        if domain == "chemical":
            # Temperature dynamics
            if states.shape[1] > 0:
                next_states[:, 0] += 0.5 * actions[:, 0] - 0.02 * (states[:, 0] - 300)
                next_states[:, 0] += self.rng.randn(states.shape[0]) * 0.1
        
        elif domain == "robotics":
            # Basic kinematic simulation
            dt = 0.02  # 50Hz control
            for i in range(min(7, states.shape[1]//2)):
                if states.shape[1] > i + 7:
                    # Velocity integration
                    next_states[:, i] += dt * states[:, i + 7]
                    # Acceleration from torque
                    next_states[:, i + 7] += dt * actions[:, i] * 0.1
        
        # Add process noise
        next_states += self.rng.randn(*next_states.shape) * 0.01
        
        return next_states.astype(np.float32)
    
    def _compute_rewards(self, states: np.ndarray, actions: np.ndarray, 
                        domain: str) -> np.ndarray:
        """Compute reward signals."""
        n_samples = states.shape[0]
        
        if domain == "chemical":
            # Reward based on temperature stability and efficiency
            temp_target = 320.0
            temp_error = np.abs(states[:, 0] - temp_target) if states.shape[1] > 0 else 0
            temp_reward = -temp_error / 10.0
            
            # Efficiency penalty for high energy use
            energy_penalty = -0.1 * np.abs(actions[:, 0])
            
            rewards = temp_reward + energy_penalty
            
        elif domain == "robotics":
            # Task completion reward with smoothness penalty
            task_reward = np.ones(n_samples) * 0.1  # Base task progress
            
            # Smoothness penalty
            action_penalty = -0.01 * np.sum(np.abs(actions), axis=1)
            
            rewards = task_reward + action_penalty
            
        else:  # Generic case
            rewards = self.rng.randn(n_samples) * 0.1
        
        # Add safety penalties
        safety_violations = self._check_safety_violations(states, domain)
        rewards[safety_violations] -= 10.0  # Large penalty for violations
        
        return rewards.astype(np.float32)
    
    def _generate_episode_boundaries(self, n_samples: int, avg_length: int = 200) -> np.ndarray:
        """Generate realistic episode termination points."""
        terminals = np.zeros(n_samples, dtype=bool)
        
        # Exponential distribution for episode lengths
        current_pos = 0
        while current_pos < n_samples:
            episode_length = int(self.rng.exponential(avg_length))
            episode_end = min(current_pos + episode_length, n_samples - 1)
            terminals[episode_end] = True
            current_pos = episode_end + 1
        
        return terminals
    
    def _check_safety_violations(self, states: np.ndarray, domain: str) -> np.ndarray:
        """Check for safety constraint violations."""
        n_samples = states.shape[0]
        violations = np.zeros(n_samples, dtype=bool)
        
        if domain == "chemical" and states.shape[1] >= 2:
            # Temperature and pressure limits
            temp_violations = (states[:, 0] > 350) | (states[:, 0] < 280)
            pressure_violations = states[:, 1] > 5.0
            violations = temp_violations | pressure_violations
            
        elif domain == "robotics" and states.shape[1] >= 7:
            # Joint limit violations
            joint_limits = np.pi
            joint_violations = np.any(np.abs(states[:, :7]) > joint_limits, axis=1)
            violations = joint_violations
        
        # Add some random safety events (sensor failures, etc.)
        random_violations = self.rng.choice([False, True], size=n_samples, p=[0.99, 0.01])
        violations = violations | random_violations
        
        return violations


def load_test_datasets() -> Dict[str, IndustrialDataset]:
    """Load all available test datasets."""
    generator = IndustrialDataGenerator()
    
    datasets = {
        "chemical_reactor": generator.generate_chemical_reactor_data(1000),
        "robot_assembly": generator.generate_robot_assembly_data(800),
    }
    
    return datasets


def save_test_dataset(dataset: IndustrialDataset, filepath: Path) -> None:
    """Save a test dataset to disk."""
    data_dict = {
        "observations": dataset.observations.tolist(),
        "actions": dataset.actions.tolist(),
        "rewards": dataset.rewards.tolist(),
        "next_observations": dataset.next_observations.tolist(),
        "terminals": dataset.terminals.tolist(),
        "timeouts": dataset.timeouts.tolist(),
        "safety_violations": dataset.safety_violations.tolist(),
        "metadata": dataset.metadata
    }
    
    with open(filepath, 'w') as f:
        json.dump(data_dict, f, indent=2)


def load_test_dataset(filepath: Path) -> IndustrialDataset:
    """Load a test dataset from disk."""
    with open(filepath, 'r') as f:
        data_dict = json.load(f)
    
    return IndustrialDataset(
        observations=np.array(data_dict["observations"], dtype=np.float32),
        actions=np.array(data_dict["actions"], dtype=np.float32),
        rewards=np.array(data_dict["rewards"], dtype=np.float32),
        next_observations=np.array(data_dict["next_observations"], dtype=np.float32),
        terminals=np.array(data_dict["terminals"], dtype=bool),
        timeouts=np.array(data_dict["timeouts"], dtype=bool),
        safety_violations=np.array(data_dict["safety_violations"], dtype=bool),
        metadata=data_dict["metadata"]
    )