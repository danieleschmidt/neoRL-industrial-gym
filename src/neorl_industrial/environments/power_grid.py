"""Power grid control environment with safety constraints."""

import numpy as np
from typing import Dict, List, Optional

from .base import IndustrialEnv
from ..core.types import Array, StateArray, ActionArray, SafetyConstraint


def frequency_constraint(state: StateArray, action: ActionArray) -> bool:
    """Ensure frequency stays within acceptable bounds."""
    # Assume state[0] is frequency deviation from 50Hz
    freq_deviation = state[0]
    return abs(freq_deviation) < 0.5  # ±0.5 Hz tolerance


def voltage_constraint(state: StateArray, action: ActionArray) -> bool:
    """Ensure voltage levels remain stable."""
    # Assume state[1:9] are voltage levels at different buses
    voltage_levels = state[1:9]
    return np.all((voltage_levels >= 0.95) & (voltage_levels <= 1.05))  # ±5% tolerance


def generation_constraint(state: StateArray, action: ActionArray) -> bool:
    """Ensure generation doesn't exceed capacity."""
    # Action represents generation adjustments
    current_gen = state[9:17]  # Current generation levels
    max_capacity = np.ones(8) * 100  # MW
    new_gen = current_gen + action
    return np.all((new_gen >= 0) & (new_gen <= max_capacity))


class PowerGridEnv(IndustrialEnv):
    """Power grid control environment.
    
    State space (32D):
    - [0]: Frequency deviation (Hz)
    - [1:9]: Bus voltage levels (p.u.)
    - [9:17]: Generator power outputs (MW)
    - [17:25]: Load demands (MW) 
    - [25:32]: Line flows (MW)
    
    Action space (8D): Generator power adjustments (MW)
    
    Rewards based on:
    - Frequency regulation
    - Voltage stability
    - Economic dispatch
    - Safety constraint satisfaction
    """
    
    def __init__(self, **kwargs):
        safety_constraints = [
            SafetyConstraint(
                name="frequency_stability",
                check_fn=frequency_constraint,
                penalty=-50.0,
                critical=True
            ),
            SafetyConstraint(
                name="voltage_limits",
                check_fn=voltage_constraint,
                penalty=-30.0,
                critical=True
            ),
            SafetyConstraint(
                name="generation_limits",
                check_fn=generation_constraint,
                penalty=-20.0,
                critical=False
            ),
        ]
        
        super().__init__(
            state_dim=32,
            action_dim=8,
            safety_constraints=safety_constraints,
            **kwargs
        )
        
        # System parameters
        self.base_load = np.array([50, 60, 45, 55, 40, 65, 35, 50])  # MW
        self.load_variation = 0.2  # ±20% load variation
        self.inertia_constant = 5.0  # Grid inertia
        self.damping_factor = 1.0
        
        # Economic parameters
        self.generation_cost = np.array([25, 30, 28, 35, 32, 27, 40, 33])  # $/MWh
    
    def _get_initial_state(self) -> StateArray:
        """Initialize power grid state."""
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Frequency deviation starts at zero
        state[0] = 0.0
        
        # Bus voltages at nominal (1.0 p.u.)
        state[1:9] = 1.0 + np.random.normal(0, 0.01, 8)  # Small variations
        
        # Initial generation levels (balanced)
        state[9:17] = self.base_load + np.random.normal(0, 2, 8)
        
        # Load demands with random variation
        load_multiplier = 1.0 + np.random.uniform(-self.load_variation, self.load_variation, 8)
        state[17:25] = self.base_load * load_multiplier
        
        # Line flows (simplified)
        state[25:32] = np.random.normal(0, 10, 7)
        
        return state
    
    def _dynamics(self, state: StateArray, action: ActionArray) -> StateArray:
        """Power grid dynamics simulation."""
        next_state = state.copy()
        
        # Current values
        freq_dev = state[0]
        voltages = state[1:9]
        generation = state[9:17]
        loads = state[17:25]
        line_flows = state[25:32]
        
        # Apply generation adjustments
        new_generation = np.clip(generation + action, 0, 100)
        
        # Power balance
        total_gen = np.sum(new_generation)
        total_load = np.sum(loads)
        power_imbalance = total_gen - total_load
        
        # Frequency dynamics (simplified swing equation)
        freq_derivative = (-self.damping_factor * freq_dev + power_imbalance) / self.inertia_constant
        new_freq_dev = freq_dev + freq_derivative * self.dt
        
        # Voltage dynamics (simplified)
        voltage_variations = np.random.normal(0, 0.005, 8)  # Small random variations
        new_voltages = voltages + voltage_variations
        
        # Load variations (realistic fluctuations)
        load_changes = np.random.normal(0, 1, 8)  # MW changes
        new_loads = np.maximum(loads + load_changes, 0)
        
        # Line flow updates (simplified)
        new_line_flows = line_flows + np.random.normal(0, 2, 7)
        
        # Update state
        next_state[0] = new_freq_dev
        next_state[1:9] = new_voltages
        next_state[9:17] = new_generation
        next_state[17:25] = new_loads
        next_state[25:32] = new_line_flows
        
        return next_state
    
    def _compute_reward(self, state: StateArray, action: ActionArray, next_state: StateArray) -> float:
        """Compute reward based on grid performance."""
        freq_dev = next_state[0]
        voltages = next_state[1:9]
        generation = next_state[9:17]
        
        # Frequency regulation reward
        freq_reward = -100 * freq_dev**2
        
        # Voltage stability reward
        voltage_deviations = np.abs(voltages - 1.0)
        voltage_reward = -50 * np.sum(voltage_deviations**2)
        
        # Economic dispatch reward (minimize generation cost)
        generation_cost = np.sum(self.generation_cost * generation)
        economic_reward = -generation_cost / 1000  # Scale down
        
        # Action smoothness (avoid large changes)
        action_penalty = -5 * np.sum(action**2)
        
        total_reward = freq_reward + voltage_reward + economic_reward + action_penalty
        
        return float(total_reward)
    
    def _is_done(self, state: StateArray) -> bool:
        """Check termination conditions."""
        freq_dev = state[0]
        voltages = state[1:9]
        
        # Critical frequency deviation
        if abs(freq_dev) > 1.0:  # ±1 Hz is critical
            return True
        
        # Critical voltage violations
        if np.any((voltages < 0.9) | (voltages > 1.1)):
            return True
        
        return False
    
    def get_dataset(self, quality: str = "mixed") -> Dict[str, Array]:
        """Generate or load offline dataset."""
        # For now, generate synthetic data
        n_samples = {
            "expert": 100000,
            "medium": 150000,
            "mixed": 200000,
            "random": 80000,
        }[quality]
        
        # Generate dataset by running random policy
        observations = []
        actions = []
        rewards = []
        terminals = []
        
        for _ in range(n_samples // 1000):  # Generate in episodes
            obs, _ = self.reset()
            done = False
            episode_length = 0
            
            while not done and episode_length < 1000:
                if quality == "expert":
                    # Simple optimal control heuristic
                    freq_error = obs[0]
                    load_gen_imbalance = np.sum(obs[17:25]) - np.sum(obs[9:17])
                    action = -0.5 * freq_error * np.ones(self.action_dim)
                    action += 0.1 * load_gen_imbalance / self.action_dim
                elif quality == "random":
                    action = np.random.uniform(-5, 5, self.action_dim)
                else:
                    # Mixed quality
                    if np.random.rand() < 0.6:
                        # Good policy
                        freq_error = obs[0]
                        action = -0.3 * freq_error * np.ones(self.action_dim)
                    else:
                        # Random policy
                        action = np.random.uniform(-3, 3, self.action_dim)
                
                observations.append(obs.copy())
                actions.append(action)
                
                obs, reward, terminated, truncated, _ = self.step(action)
                rewards.append(reward)
                terminals.append(terminated)
                
                done = terminated or truncated
                episode_length += 1
        
        return {
            "observations": np.array(observations, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "terminals": np.array(terminals, dtype=bool),
        }
