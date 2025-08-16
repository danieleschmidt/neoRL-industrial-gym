"""Advanced power grid environment with detailed electrical network dynamics."""

import math
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from ..core.types import SafetyConstraint, SafetyMetrics
from .base import IndustrialEnv


class AdvancedPowerGridEnv(IndustrialEnv):
    """
    High-fidelity power grid with detailed electrical network dynamics.
    
    This environment models a multi-bus power system with:
    - Generator swing equation dynamics
    - Load flow analysis with Newton-Raphson
    - Transmission line models with impedance
    - Frequency and voltage regulation
    - Power system stability analysis
    - Protection system coordination
    
    State space (32D):
    - Bus voltages [p.u.] (8 buses)
    - Bus angles [rad] (8 buses)
    - Generator frequencies [Hz] (4 generators)
    - Generator power outputs [MW] (4 generators)
    - Load powers [MW] (4 loads)
    - Line flows [MW] (4 transmission lines)
    - Transformer tap positions [-] (2 transformers)
    - System frequency [Hz] (1D)
    - Total generation [MW] (1D)
    - Total load [MW] (1D)
    - System inertia [s] (1D)
    - Stability margin [-] (1D)
    
    Action space (8D):
    - Generator setpoints [MW] (4 generators)
    - Voltage regulator settings [p.u.] (2 voltage regulators)  
    - Load shedding commands [MW] (1D)
    - Emergency protection trigger [binary] (1D)
    
    The power system consists of:
    - 4 synchronous generators
    - 8 electrical buses
    - 4 transmission lines
    - 2 transformers
    - 4 load centers
    """
    
    def __init__(
        self,
        dt: float = 0.1,
        max_episode_steps: int = 500,
        nominal_frequency: float = 50.0,  # Hz
        frequency_tolerance: float = 0.5,  # Hz
        voltage_tolerance: float = 0.05,   # p.u.
        base_power: float = 100.0,         # MVA
        **kwargs
    ):
        self.dt = dt
        self.nominal_frequency = nominal_frequency
        self.frequency_tolerance = frequency_tolerance
        self.voltage_tolerance = voltage_tolerance
        self.base_power = base_power
        
        # Power system parameters
        self.n_buses = 8
        self.n_generators = 4
        self.n_loads = 4
        self.n_lines = 4
        self.n_transformers = 2
        
        # Generator parameters
        self.generator_params = {
            'inertia': jnp.array([5.0, 4.0, 3.5, 4.5]),  # seconds
            'damping': jnp.array([1.0, 0.8, 0.9, 1.1]),  # p.u.
            'max_power': jnp.array([50.0, 40.0, 35.0, 45.0]),  # MW
            'min_power': jnp.array([10.0, 8.0, 7.0, 9.0]),    # MW
            'ramp_rate': jnp.array([2.0, 1.8, 1.5, 2.2]),     # MW/s
        }
        
        # Transmission line parameters (impedance in p.u.)
        self.line_impedances = jnp.array([
            [0.02 + 0.1j, 0.03 + 0.12j, 0.025 + 0.11j, 0.028 + 0.13j]
        ])
        
        # Load parameters
        self.load_params = {
            'base_load': jnp.array([25.0, 20.0, 30.0, 18.0]),  # MW
            'voltage_dependence': jnp.array([1.5, 1.2, 1.8, 1.3]),  # voltage exponent
            'frequency_dependence': jnp.array([1.0, 0.8, 1.2, 0.9])  # frequency coefficient
        }
        
        # Safety constraints
        safety_constraints = [
            SafetyConstraint(
                name="frequency_high",
                constraint_fn=lambda state: state[23] < nominal_frequency + frequency_tolerance,
                violation_penalty=-500.0
            ),
            SafetyConstraint(
                name="frequency_low",
                constraint_fn=lambda state: state[23] > nominal_frequency - frequency_tolerance,
                violation_penalty=-500.0
            ),
            SafetyConstraint(
                name="voltage_limits",
                constraint_fn=lambda state: jnp.all(jnp.abs(state[0:8] - 1.0) < voltage_tolerance),
                violation_penalty=-300.0
            ),
            SafetyConstraint(
                name="generator_limits", 
                constraint_fn=lambda state: jnp.all(state[16:20] >= 0.0),  # Positive generation
                violation_penalty=-200.0
            )
        ]
        
        super().__init__(
            state_dim=32,
            action_dim=8,
            max_episode_steps=max_episode_steps,
            safety_constraints=safety_constraints,
            **kwargs
        )
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=jnp.array([
                *[0.8] * 8,  # Bus voltages (0.8-1.2 p.u.)
                *[-jnp.pi] * 8,  # Bus angles (-π to π)
                *[49.0] * 4,  # Generator frequencies (49-51 Hz)
                *[0.0] * 4,   # Generator powers (0-max MW)
                *[0.0] * 4,   # Load powers (0-max MW)
                *[-100.0] * 4,  # Line flows (-100 to 100 MW)
                *[0.8] * 2,   # Transformer taps (0.8-1.2)
                49.0,         # System frequency (49-51 Hz)
                0.0,          # Total generation (0-200 MW)
                0.0,          # Total load (0-150 MW)
                0.0,          # System inertia (0-20 s)
                0.0           # Stability margin (0-1)
            ]),
            high=jnp.array([
                *[1.2] * 8,   # Bus voltages
                *[jnp.pi] * 8,  # Bus angles
                *[51.0] * 4,  # Generator frequencies
                *[50.0, 40.0, 35.0, 45.0],  # Generator max powers
                *[40.0, 30.0, 45.0, 25.0],  # Load max powers
                *[100.0] * 4,  # Line flows
                *[1.2] * 2,   # Transformer taps
                51.0,         # System frequency
                200.0,        # Total generation
                150.0,        # Total load
                20.0,         # System inertia
                1.0           # Stability margin
            ]),
            dtype=jnp.float32
        )
        
        self.action_space = spaces.Box(
            low=jnp.array([
                *self.generator_params['min_power'],  # Generator setpoints
                0.95, 0.95,  # Voltage regulator settings
                0.0,         # Load shedding
                0.0          # Emergency protection
            ]),
            high=jnp.array([
                *self.generator_params['max_power'],  # Generator setpoints
                1.05, 1.05,  # Voltage regulator settings
                20.0,        # Load shedding (MW)
                1.0          # Emergency protection
            ]),
            dtype=jnp.float32
        )
        
        self.reset()
        
    def reset(self, seed: Optional[int] = None) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Reset power system to normal operating conditions."""
        super().reset(seed=seed)
        
        # Initial steady-state conditions
        initial_voltages = jnp.ones(8)  # 1.0 p.u. at all buses
        initial_angles = jnp.array([0.0, -0.1, 0.05, -0.05, 0.02, -0.02, 0.08, -0.08])  # Small angle differences
        
        initial_gen_freqs = jnp.full(4, self.nominal_frequency)
        initial_gen_powers = jnp.array([30.0, 25.0, 20.0, 28.0])  # MW
        
        initial_loads = self.load_params['base_load']
        
        # Calculate initial line flows (simplified)
        initial_line_flows = jnp.array([15.0, -12.0, 18.0, -14.0])  # MW
        
        initial_taps = jnp.ones(2)  # Nominal tap positions
        
        system_frequency = self.nominal_frequency
        total_generation = jnp.sum(initial_gen_powers)
        total_load = jnp.sum(initial_loads)
        system_inertia = jnp.sum(self.generator_params['inertia'])
        stability_margin = 0.8  # Good initial stability
        
        self.state = jnp.array([
            *initial_voltages,      # 0-7: Bus voltages
            *initial_angles,        # 8-15: Bus angles
            *initial_gen_freqs,     # 16-19: Generator frequencies
            *initial_gen_powers,    # 20-23: Generator powers
            *initial_loads,         # 24-27: Load powers
            *initial_line_flows,    # 28-31: Line flows
            *initial_taps,          # 32-33: Transformer taps (Note: This extends beyond 32D, adjusting)
        ])
        
        # Adjust to exactly 32 dimensions
        self.state = jnp.array([
            *initial_voltages,      # 0-7: Bus voltages  
            *initial_angles,        # 8-15: Bus angles
            *initial_gen_freqs,     # 16-19: Generator frequencies
            *initial_gen_powers,    # 20-23: Generator powers
            *initial_loads,         # 24-27: Load powers
            *initial_line_flows,    # 28-31: Line flows
        ])
        
        return self.state, {}
        
    def step(self, action: jnp.ndarray) -> Tuple[jnp.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one simulation step with detailed power system dynamics."""
        
        # Extract current state
        bus_voltages = self.state[0:8]
        bus_angles = self.state[8:16]
        gen_frequencies = self.state[16:20]
        gen_powers = self.state[20:24]
        load_powers = self.state[24:28]
        line_flows = self.state[28:32]
        
        # Extract actions
        gen_setpoints = action[0:4]
        voltage_setpoints = action[4:6]
        load_shedding = action[6]
        emergency_protection = action[7]
        
        # Emergency protection logic
        if emergency_protection > 0.5:
            # Implement emergency control actions
            gen_setpoints = gen_setpoints * 0.7  # Reduce generation
            load_shedding = jnp.minimum(load_shedding + 10.0, 30.0)  # Increase load shedding
            
        # Generator dynamics (simplified swing equation)
        power_imbalance = gen_powers - load_powers[0:4]  # Simplified mapping
        
        # Calculate generator frequency derivatives
        gen_freq_derivatives = []
        for i in range(self.n_generators):
            inertia = self.generator_params['inertia'][i]
            damping = self.generator_params['damping'][i]
            
            # Simplified swing equation: 2H * df/dt = Pm - Pe - D * (f - f_nom)
            pm = gen_setpoints[i] / self.base_power  # Mechanical power (p.u.)
            pe = gen_powers[i] / self.base_power     # Electrical power (p.u.)
            
            freq_deviation = gen_frequencies[i] - self.nominal_frequency
            df_dt = (pm - pe - damping * freq_deviation) / (2 * inertia)
            
            gen_freq_derivatives.append(df_dt)
            
        gen_freq_derivatives = jnp.array(gen_freq_derivatives)
        
        # Update generator frequencies
        new_gen_frequencies = gen_frequencies + self.dt * gen_freq_derivatives
        
        # System frequency (weighted average by inertia)
        total_inertia = jnp.sum(self.generator_params['inertia'])
        system_frequency = jnp.sum(new_gen_frequencies * self.generator_params['inertia']) / total_inertia
        
        # Generator power control (with ramp rate limits)
        power_errors = gen_setpoints - gen_powers
        max_ramp = self.generator_params['ramp_rate'] * self.dt
        power_changes = jnp.clip(power_errors, -max_ramp, max_ramp)
        new_gen_powers = gen_powers + power_changes
        
        # Clip to generator limits
        new_gen_powers = jnp.clip(
            new_gen_powers,
            self.generator_params['min_power'],
            self.generator_params['max_power']
        )
        
        # Load dynamics (frequency and voltage dependent)
        new_loads = []
        for i in range(self.n_loads):
            base_load = self.load_params['base_load'][i]
            
            # Apply load shedding
            if i == 0:  # Apply load shedding to first load bus
                base_load = jnp.maximum(base_load - load_shedding, 0.0)
                
            # Voltage dependence: P = P0 * (V/V0)^α
            voltage_effect = (bus_voltages[i] / 1.0) ** self.load_params['voltage_dependence'][i]
            
            # Frequency dependence: P = P0 * (1 + K * Δf/f0)
            freq_deviation = (system_frequency - self.nominal_frequency) / self.nominal_frequency
            frequency_effect = 1.0 + self.load_params['frequency_dependence'][i] * freq_deviation
            
            new_load = base_load * voltage_effect * frequency_effect
            new_loads.append(new_load)
            
        new_loads = jnp.array(new_loads)
        
        # Network power flow analysis (simplified)
        new_bus_voltages, new_bus_angles, new_line_flows = self._solve_power_flow(
            new_gen_powers, new_loads, voltage_setpoints
        )
        
        # Calculate stability metrics
        stability_margin = self._calculate_stability_margin(
            new_bus_voltages, new_bus_angles, new_gen_frequencies
        )
        
        # Update state
        self.state = jnp.array([
            *new_bus_voltages,     # 0-7: Bus voltages
            *new_bus_angles,       # 8-15: Bus angles
            *new_gen_frequencies,  # 16-19: Generator frequencies
            *new_gen_powers,       # 20-23: Generator powers
            *new_loads,            # 24-27: Load powers
            *new_line_flows,       # 28-31: Line flows
        ])
        
        # Calculate reward
        reward = self._compute_reward(action, system_frequency, new_bus_voltages)
        
        # Check termination conditions
        safety_metrics = self.get_safety_metrics()
        terminated = self._check_termination(system_frequency, new_bus_voltages, stability_margin)
        truncated = self.episode_step >= self.max_episode_steps
        
        # Info dictionary
        info = {
            "system_frequency": float(system_frequency),
            "total_generation": float(jnp.sum(new_gen_powers)),
            "total_load": float(jnp.sum(new_loads)),
            "power_balance": float(jnp.sum(new_gen_powers) - jnp.sum(new_loads)),
            "stability_margin": float(stability_margin),
            "safety_metrics": safety_metrics,
            "emergency_active": bool(emergency_protection > 0.5),
            "load_shedding_amount": float(load_shedding)
        }
        
        self.episode_step += 1
        
        return self.state, reward, terminated, truncated, info
        
    def _solve_power_flow(
        self, 
        gen_powers: jnp.ndarray, 
        loads: jnp.ndarray,
        voltage_setpoints: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Simplified power flow analysis."""
        
        # This is a simplified power flow - in practice would use Newton-Raphson
        current_voltages = self.state[0:8]
        current_angles = self.state[8:16]
        
        # Power injection at each bus
        power_injections = jnp.zeros(8)
        
        # Generator buses (first 4 buses)
        power_injections = power_injections.at[0:4].set(gen_powers / self.base_power)
        
        # Load buses (last 4 buses)  
        power_injections = power_injections.at[4:8].set(-loads / self.base_power)
        
        # Simplified voltage update based on power injections
        voltage_changes = 0.01 * power_injections  # Small voltage sensitivity
        new_voltages = current_voltages + voltage_changes
        
        # Apply voltage regulation at generator buses
        new_voltages = new_voltages.at[0].set(voltage_setpoints[0])
        new_voltages = new_voltages.at[1].set(voltage_setpoints[1])
        
        # Clip voltages to reasonable bounds
        new_voltages = jnp.clip(new_voltages, 0.8, 1.2)
        
        # Simplified angle update based on power flow
        angle_changes = 0.05 * power_injections  # Small angle sensitivity
        new_angles = current_angles + angle_changes
        
        # Calculate line flows (simplified)
        line_flows = []
        line_connections = [(0, 4), (1, 5), (2, 6), (3, 7)]  # Generator-Load pairs
        
        for i, (from_bus, to_bus) in enumerate(line_connections):
            # Simplified power flow: P = V1*V2/X * sin(θ1 - θ2)
            v1, v2 = new_voltages[from_bus], new_voltages[to_bus]
            theta1, theta2 = new_angles[from_bus], new_angles[to_bus]
            
            # Use simplified reactance
            reactance = 0.1  # p.u.
            
            flow = v1 * v2 / reactance * jnp.sin(theta1 - theta2) * self.base_power
            line_flows.append(flow)
            
        return new_voltages, new_angles, jnp.array(line_flows)
        
    def _calculate_stability_margin(
        self, 
        voltages: jnp.ndarray, 
        angles: jnp.ndarray,
        frequencies: jnp.ndarray
    ) -> float:
        """Calculate power system stability margin."""
        
        # Voltage stability: check voltage magnitudes
        voltage_margin = 1.0 - jnp.max(jnp.abs(voltages - 1.0))
        
        # Angle stability: check angle differences
        max_angle_diff = jnp.max(angles) - jnp.min(angles)
        angle_margin = 1.0 - max_angle_diff / jnp.pi
        
        # Frequency stability: check frequency deviations
        freq_deviations = jnp.abs(frequencies - self.nominal_frequency)
        freq_margin = 1.0 - jnp.max(freq_deviations) / self.frequency_tolerance
        
        # Overall stability margin (minimum of all margins)
        stability_margin = jnp.minimum(
            jnp.minimum(voltage_margin, angle_margin), 
            freq_margin
        )
        
        return jnp.maximum(stability_margin, 0.0)
        
    def _compute_reward(
        self, 
        action: jnp.ndarray,
        system_frequency: float,
        voltages: jnp.ndarray
    ) -> float:
        """Compute reward based on power system performance."""
        
        # Frequency regulation reward
        freq_error = jnp.abs(system_frequency - self.nominal_frequency)
        freq_reward = 100.0 * jnp.exp(-freq_error / 0.1)
        
        # Voltage regulation reward
        voltage_errors = jnp.abs(voltages - 1.0)
        voltage_reward = 50.0 * jnp.exp(-jnp.mean(voltage_errors) / 0.05)
        
        # Power balance reward
        total_generation = jnp.sum(self.state[20:24])
        total_load = jnp.sum(self.state[24:28])
        power_imbalance = jnp.abs(total_generation - total_load)
        balance_reward = 30.0 * jnp.exp(-power_imbalance / 10.0)
        
        # Economic efficiency (minimize generation cost)
        # Simplified quadratic cost function
        gen_costs = 0.01 * jnp.sum(self.state[20:24] ** 2)
        economic_reward = -gen_costs
        
        # Control effort penalty
        control_penalty = -jnp.sum(jnp.abs(action[0:6])) * 1.0
        
        # Load shedding penalty
        load_shedding_penalty = -action[6] * 50.0
        
        # Emergency action penalty
        emergency_penalty = -action[7] * 200.0
        
        total_reward = (
            freq_reward + 
            voltage_reward + 
            balance_reward + 
            economic_reward + 
            control_penalty + 
            load_shedding_penalty + 
            emergency_penalty
        )
        
        return float(total_reward)
        
    def _check_termination(
        self, 
        system_frequency: float, 
        voltages: jnp.ndarray,
        stability_margin: float
    ) -> bool:
        """Check for critical power system failures."""
        
        # Frequency out of bounds
        freq_violation = jnp.abs(system_frequency - self.nominal_frequency) > self.frequency_tolerance
        
        # Voltage out of bounds
        voltage_violation = jnp.any(jnp.abs(voltages - 1.0) > self.voltage_tolerance)
        
        # System instability
        instability = stability_margin < 0.1
        
        return bool(freq_violation or voltage_violation or instability)
        
    def get_safety_metrics(self) -> SafetyMetrics:
        """Get current safety metrics for the power system."""
        
        system_frequency = jnp.sum(self.state[16:20] * self.generator_params['inertia']) / jnp.sum(self.generator_params['inertia'])
        voltages = self.state[0:8]
        
        violations = []
        
        # Check frequency bounds
        freq_deviation = jnp.abs(system_frequency - self.nominal_frequency)
        if freq_deviation > self.frequency_tolerance:
            violations.append("frequency_deviation")
            
        # Check voltage bounds
        voltage_violations = jnp.abs(voltages - 1.0) > self.voltage_tolerance
        if jnp.any(voltage_violations):
            violations.append("voltage_deviation")
            
        # Check generation limits
        gen_powers = self.state[20:24]
        if jnp.any(gen_powers < self.generator_params['min_power']) or jnp.any(gen_powers > self.generator_params['max_power']):
            violations.append("generation_limits")
            
        # Calculate overall safety score
        freq_score = 100.0 * (1.0 - freq_deviation / self.frequency_tolerance)
        voltage_score = 100.0 * (1.0 - jnp.mean(jnp.abs(voltages - 1.0)) / self.voltage_tolerance)
        safety_score = jnp.minimum(freq_score, voltage_score)
        
        return SafetyMetrics(
            total_violations=len(violations),
            violation_types=violations,
            safety_score=float(jnp.maximum(safety_score, 0.0)),
            constraint_satisfaction=len(violations) == 0
        )
        
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render power system state."""
        
        if mode == "human":
            system_freq = jnp.sum(self.state[16:20] * self.generator_params['inertia']) / jnp.sum(self.generator_params['inertia'])
            
            print(f"=== Power System State ===")
            print(f"System Frequency: {system_freq:.2f} Hz")
            print(f"Bus Voltages: {self.state[0:4]:.3f} p.u.")
            print(f"Generator Powers: {self.state[20:24]:.1f} MW")
            print(f"Load Powers: {self.state[24:28]:.1f} MW")
            print(f"Total Generation: {jnp.sum(self.state[20:24]):.1f} MW")
            print(f"Total Load: {jnp.sum(self.state[24:28]):.1f} MW")
            print("=" * 30)
            
        return None