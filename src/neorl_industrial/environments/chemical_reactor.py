"""ChemicalReactor-v0: Industrial chemical reactor control environment."""

import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional

from ..core.types import Array, SafetyConstraint, StateArray, ActionArray
from .base import IndustrialEnv


class ChemicalReactorEnv(IndustrialEnv):
    """Chemical reactor control environment with temperature and pressure safety constraints.
    
    State space (12D):
        [0] Reactor temperature (K)
        [1] Reactor pressure (Pa) 
        [2] Cooling water flow rate (L/min)
        [3] Feed flow rate (L/min)
        [4] Product concentration (mol/L)
        [5] Catalyst activity (%)
        [6] Heat exchanger temperature (K)
        [7] Pressure relief valve position (%)
        [8] Emergency shutdown status (0/1)
        [9] Process alarm status (0/1)
        [10] Reactor level (%)
        [11] Time in current batch (min)
    
    Action space (3D):
        [0] Heating power adjustment (-1 to 1)
        [1] Cooling water valve adjustment (-1 to 1) 
        [2] Feed rate adjustment (-1 to 1)
    """
    
    def __init__(self, **kwargs):
        """Initialize chemical reactor environment."""
        
        # Default safety constraints for chemical reactor
        safety_constraints = [
            SafetyConstraint(
                name="temperature_limit",
                check_fn=self._temperature_constraint,
                penalty=-100.0,
                critical=True,
                description="Reactor temperature must stay below 350K"
            ),
            SafetyConstraint(
                name="pressure_limit", 
                check_fn=self._pressure_constraint,
                penalty=-50.0,
                critical=True,
                description="Reactor pressure must stay below 5 atm"
            ),
            SafetyConstraint(
                name="level_safety",
                check_fn=self._level_constraint,
                penalty=-25.0,
                critical=False,
                description="Reactor level must stay between 20-90%"
            )
        ]
        
        super().__init__(
            state_dim=12,
            action_dim=3,
            safety_constraints=safety_constraints,
            max_episode_steps=500,
            dt=0.1,
            **kwargs
        )
        
        # Physical parameters
        self.temp_min = 280.0  # K
        self.temp_max = 350.0  # K (safety limit)
        self.temp_target = 320.0  # K (optimal operating temperature)
        
        self.pressure_min = 101325.0  # Pa (1 atm)
        self.pressure_max = 506625.0  # Pa (5 atm, safety limit)
        self.pressure_target = 253312.5  # Pa (2.5 atm)
        
        # Process parameters
        self.reaction_rate_constant = 0.1
        self.heat_capacity = 4.18e3  # J/(kg*K)
        self.reactor_volume = 1000.0  # L
        
        # Disturbance parameters for realism
        self.temp_noise_std = 1.0
        self.pressure_noise_std = 5000.0
    
    def _get_initial_state(self) -> StateArray:
        """Get initial reactor state."""
        # Start near optimal operating conditions with small random variation
        initial_state = np.array([
            self.temp_target + np.random.normal(0, 2),  # temperature
            self.pressure_target + np.random.normal(0, 10000),  # pressure  
            50.0 + np.random.normal(0, 5),  # cooling water flow
            30.0 + np.random.normal(0, 3),  # feed flow
            0.5 + np.random.normal(0, 0.1),  # product concentration
            95.0 + np.random.normal(0, 2),  # catalyst activity
            295.0 + np.random.normal(0, 1),  # heat exchanger temp
            0.0,  # pressure relief valve (closed initially)
            0.0,  # emergency shutdown (off initially)
            0.0,  # process alarm (off initially) 
            60.0 + np.random.normal(0, 5),  # reactor level
            0.0   # batch time
        ], dtype=np.float32)
        
        return initial_state
    
    def _dynamics(self, state: StateArray, action: ActionArray) -> StateArray:
        """Compute reactor dynamics."""
        # Current state variables
        temp = state[0]
        pressure = state[1] 
        cooling_flow = state[2]
        feed_flow = state[3]
        concentration = state[4]
        catalyst_activity = state[5]
        hx_temp = state[6]
        relief_valve = state[7]
        emergency_stop = state[8]
        alarm = state[9]
        level = state[10]
        batch_time = state[11]
        
        # Action effects (if not in emergency shutdown)
        if emergency_stop < 0.5:
            heating_power = action[0] * 50000  # W
            cooling_valve_adj = action[1] * 0.1
            feed_adj = action[2] * 0.1
        else:
            # Emergency shutdown: no manual control
            heating_power = -10000  # Emergency cooling
            cooling_valve_adj = 0.1  # Max cooling
            feed_adj = -0.1  # Reduce feed
        
        # Temperature dynamics
        reaction_heat = (
            self.reaction_rate_constant * concentration * 
            (catalyst_activity / 100.0) * 10000
        )
        cooling_heat = cooling_flow * 100 * (temp - hx_temp) * 0.1
        
        dTemp_dt = (
            (heating_power + reaction_heat - cooling_heat) / 
            (self.heat_capacity * 1000 * self.dt)
        )
        
        # Add process noise
        dTemp_dt += np.random.normal(0, self.temp_noise_std / 10)
        
        new_temp = temp + dTemp_dt * self.dt
        
        # Pressure dynamics (simplified ideal gas approximation)
        # Pressure increases with temperature and reaction rate
        pressure_from_temp = pressure * (new_temp / temp)
        pressure_from_reaction = concentration * self.reaction_rate_constant * 1000
        
        new_pressure = pressure_from_temp + pressure_from_reaction * self.dt
        new_pressure += np.random.normal(0, self.pressure_noise_std / 10)
        
        # Pressure relief valve activation
        new_relief_valve = max(0, min(100, relief_valve + 
                                    (new_pressure - self.pressure_max) * 0.001))
        
        # Pressure relief effect
        if new_relief_valve > 0:
            pressure_relief = new_relief_valve * 0.01 * 10000
            new_pressure = max(self.pressure_min, new_pressure - pressure_relief)
        
        # Flow dynamics
        new_cooling_flow = max(10, min(100, cooling_flow + cooling_valve_adj))
        new_feed_flow = max(5, min(50, feed_flow + feed_adj))
        
        # Concentration dynamics
        reaction_rate = (
            self.reaction_rate_constant * concentration * 
            (catalyst_activity / 100.0) * np.exp(-(new_temp - 320) / 20)
        )
        
        feed_dilution = new_feed_flow * 0.001
        new_concentration = max(0, concentration + 
                               (reaction_rate - feed_dilution) * self.dt)
        
        # Catalyst deactivation over time
        deactivation_rate = 0.001 if new_temp > 340 else 0.0001
        new_catalyst = max(50, catalyst_activity - deactivation_rate)
        
        # Heat exchanger temperature (follows cooling water with lag)
        hx_lag = 0.1
        new_hx_temp = hx_temp + hx_lag * (290 + cooling_flow * 0.1 - hx_temp) * self.dt
        
        # Emergency shutdown logic
        new_emergency_stop = emergency_stop
        new_alarm = alarm
        
        if new_temp > 345 or new_pressure > 480000:  # Approaching limits
            new_alarm = 1.0
            
        if new_temp > 350 or new_pressure > 506625:  # At safety limits
            new_emergency_stop = 1.0
            new_alarm = 1.0
            
        # Level dynamics (simplified)
        level_change = (new_feed_flow - 20) * 0.1  # Baseline outflow of 20 L/min
        new_level = max(0, min(100, level + level_change * self.dt))
        
        # Batch time increment
        new_batch_time = batch_time + self.dt
        
        # Create new state vector
        next_state = np.array([
            new_temp,
            new_pressure, 
            new_cooling_flow,
            new_feed_flow,
            new_concentration,
            new_catalyst,
            new_hx_temp,
            new_relief_valve,
            new_emergency_stop,
            new_alarm,
            new_level,
            new_batch_time
        ], dtype=np.float32)
        
        return next_state
    
    def _compute_reward(self, state: StateArray, action: ActionArray, next_state: StateArray) -> float:
        """Compute reward focusing on optimal operation and safety."""
        temp = next_state[0]
        pressure = next_state[1]
        concentration = next_state[4]
        catalyst = next_state[5]
        emergency_stop = next_state[8]
        alarm = next_state[9]
        level = next_state[10]
        
        reward = 0.0
        
        # Production reward (higher concentration is better)
        reward += concentration * 100
        
        # Temperature optimization (prefer target temperature)
        temp_error = abs(temp - self.temp_target)
        reward -= temp_error * 0.5
        
        # Pressure optimization (prefer target pressure)
        pressure_error = abs(pressure - self.pressure_target) / 1000
        reward -= pressure_error * 0.1
        
        # Catalyst preservation (higher activity is better)
        reward += (catalyst / 100.0) * 10
        
        # Level maintenance
        if 30 <= level <= 80:
            reward += 5
        else:
            reward -= abs(level - 55) * 0.2
            
        # Penalty for alarms and emergency stops
        if alarm > 0.5:
            reward -= 50
        if emergency_stop > 0.5:
            reward -= 200
            
        # Action smoothness (penalize large actions)
        action_penalty = np.sum(np.abs(action)) * 0.1
        reward -= action_penalty
        
        return reward
    
    def _is_done(self, state: StateArray) -> bool:
        """Check termination conditions."""
        emergency_stop = state[8]
        batch_time = state[11]
        level = state[10]
        
        # Terminate on emergency shutdown
        if emergency_stop > 0.5:
            return True
            
        # Terminate if reactor level too low/high
        if level < 5 or level > 95:
            return True
            
        # Terminate after maximum batch time
        if batch_time > 50:  # 5 minutes in simulation time
            return True
            
        return False
    
    def _temperature_constraint(self, state: StateArray, action: ActionArray) -> bool:
        """Check temperature safety constraint."""
        temp = state[0] if len(state.shape) == 1 else state[0, 0]
        return temp <= self.temp_max
    
    def _pressure_constraint(self, state: StateArray, action: ActionArray) -> bool:
        """Check pressure safety constraint.""" 
        pressure = state[1] if len(state.shape) == 1 else state[1, 0]
        return pressure <= self.pressure_max
    
    def _level_constraint(self, state: StateArray, action: ActionArray) -> bool:
        """Check reactor level constraint."""
        level = state[10] if len(state.shape) == 1 else state[10, 0]
        return 20 <= level <= 90
    
    def _get_safety_info(self, state: StateArray) -> Dict:
        """Get safety-related information."""
        return {
            "safety_metrics": {
                "temperature": state[0],
                "pressure": state[1],
                "level": state[10], 
                "emergency_stop": state[8],
                "alarm_status": state[9],
            },
            "constraint_values": {
                "temp_margin": self.temp_max - state[0],
                "pressure_margin": self.pressure_max - state[1],
                "level_in_bounds": 20 <= state[10] <= 90,
            }
        }
    
    def get_dataset(self, quality: str = "mixed") -> Dict[str, Array]:
        """Generate synthetic dataset for chemical reactor.
        
        Args:
            quality: Dataset quality level ('expert', 'medium', 'mixed', 'random')
            
        Returns:
            Dictionary with offline dataset
        """
        if quality == "expert":
            n_episodes, n_steps = 100, 400
            noise_level = 0.1
            
        elif quality == "medium": 
            n_episodes, n_steps = 200, 350
            noise_level = 0.3
            
        elif quality == "mixed":
            n_episodes, n_steps = 300, 300
            noise_level = 0.5
            
        else:  # random
            n_episodes, n_steps = 500, 200
            noise_level = 1.0
        
        # Generate synthetic trajectory data
        observations = []
        actions = []
        rewards = []
        terminals = []
        
        for episode in range(n_episodes):
            obs, _ = self.reset()
            ep_obs = [obs]
            ep_actions = []
            ep_rewards = []
            ep_terminals = []
            
            for step in range(n_steps):
                # Generate action based on quality level
                if quality == "expert":
                    # PID-like controller for expert behavior
                    temp_error = (obs[0] - self.temp_target) / 50
                    pressure_error = (obs[1] - self.pressure_target) / 100000
                    level_error = (obs[10] - 55) / 50
                    
                    action = np.array([
                        -temp_error * 0.5 + np.random.normal(0, noise_level * 0.1),
                        temp_error * 0.3 + np.random.normal(0, noise_level * 0.1), 
                        -level_error * 0.2 + np.random.normal(0, noise_level * 0.1)
                    ])
                    
                else:
                    # Suboptimal or random actions
                    if np.random.random() < (1 - noise_level):
                        # Some intelligent action
                        temp_error = (obs[0] - self.temp_target) / 50
                        action = np.array([
                            -temp_error * 0.2 + np.random.normal(0, noise_level * 0.3),
                            np.random.normal(0, noise_level * 0.5),
                            np.random.normal(0, noise_level * 0.3)
                        ])
                    else:
                        # Random action
                        action = np.random.uniform(-1, 1, 3)
                
                action = np.clip(action, -1, 1)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = self.step(action)
                done = terminated or truncated
                
                # Store transition
                ep_actions.append(action)
                ep_rewards.append(reward)
                ep_terminals.append(done)
                
                if not done:
                    ep_obs.append(next_obs)
                    obs = next_obs
                else:
                    break
            
            # Add episode data
            observations.extend(ep_obs[:-1])  # Remove last obs
            actions.extend(ep_actions)
            rewards.extend(ep_rewards)
            terminals.extend(ep_terminals)
        
        return {
            "observations": np.array(observations, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "terminals": np.array(terminals, dtype=bool),
            "timeouts": np.zeros_like(terminals, dtype=bool),
        }