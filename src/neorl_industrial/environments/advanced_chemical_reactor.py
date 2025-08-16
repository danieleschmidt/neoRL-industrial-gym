"""Advanced chemical reactor environment with high-fidelity CFD-informed dynamics."""

import math
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from ..core.types import SafetyConstraint, SafetyMetrics
from .base import IndustrialEnv


class AdvancedChemicalReactorEnv(IndustrialEnv):
    """
    High-fidelity chemical reactor with CFD-informed dynamics.
    
    This environment models a Continuous Stirred Tank Reactor (CSTR) with:
    - Multi-component reaction kinetics
    - Heat transfer with jacket cooling
    - Mass transfer effects
    - Non-ideal mixing patterns
    - Temperature and pressure safety constraints
    
    State space (20D):
    - Reactor temperature [K] (1D)
    - Jacket temperature [K] (1D) 
    - Pressure [Pa] (1D)
    - Concentrations [mol/L] (4 components: A, B, C, D)
    - Flow rates [m³/s] (3 streams: feed, product, coolant)
    - Heat transfer coefficient [W/m²K] (1D)
    - Mixing efficiency [-] (1D)
    - Wall temperatures [K] (4 sections)
    - Residence time [s] (1D)
    - Conversion rate [-] (1D)
    - Safety margins [%] (2D: temperature, pressure)
    
    Action space (6D):
    - Feed flow rate control [m³/s]
    - Coolant flow rate control [m³/s]
    - Reactor agitation speed [rpm]
    - Feed temperature setpoint [K]
    - Pressure relief valve opening [%]
    - Emergency shutdown trigger [binary]
    
    Reaction: A + B → C + D (exothermic, irreversible)
    """
    
    def __init__(
        self,
        dt: float = 1.0,
        max_episode_steps: int = 1000,
        safety_temperature_limit: float = 673.15,  # 400°C
        safety_pressure_limit: float = 5e6,  # 5 MPa
        reactor_volume: float = 1.0,  # m³
        heat_capacity: float = 4180.0,  # J/kg·K (water-like)
        density: float = 1000.0,  # kg/m³
        activation_energy: float = 8.314e4,  # J/mol
        pre_exponential: float = 1e8,  # 1/s
        heat_of_reaction: float = -5e4,  # J/mol (exothermic)
        **kwargs
    ):
        # Physical parameters
        self.dt = dt
        self.reactor_volume = reactor_volume
        self.heat_capacity = heat_capacity
        self.density = density
        self.activation_energy = activation_energy
        self.pre_exponential = pre_exponential
        self.heat_of_reaction = heat_of_reaction
        
        # Safety limits
        self.safety_temperature_limit = safety_temperature_limit
        self.safety_pressure_limit = safety_pressure_limit
        
        # Heat transfer parameters
        self.jacket_area = 4.0 * math.pi * (reactor_volume / (4/3 * math.pi))**(2/3)  # Sphere surface
        self.wall_thickness = 0.01  # m
        self.wall_conductivity = 50.0  # W/m·K (steel)
        
        # Operating ranges
        self.temp_range = (273.15, 473.15)  # 0-200°C
        self.pressure_range = (1e5, 3e6)    # 1-30 bar
        self.concentration_range = (0.0, 10.0)  # mol/L
        self.flow_range = (0.0, 0.01)       # m³/s
        
        # Safety constraints
        safety_constraints = [
            SafetyConstraint(
                name="temperature_limit",
                constraint_fn=lambda state: state[0] < safety_temperature_limit,
                violation_penalty=-1000.0
            ),
            SafetyConstraint(
                name="pressure_limit", 
                constraint_fn=lambda state: state[2] < safety_pressure_limit,
                violation_penalty=-1000.0
            ),
            SafetyConstraint(
                name="conversion_efficiency",
                constraint_fn=lambda state: state[19] > 0.1,  # Minimum conversion
                violation_penalty=-100.0
            )
        ]
        
        super().__init__(
            state_dim=20,
            action_dim=6,
            max_episode_steps=max_episode_steps,
            safety_constraints=safety_constraints,
            **kwargs
        )
        
        # Define observation and action spaces
        self.observation_space = spaces.Box(
            low=jnp.array([
                self.temp_range[0],      # Reactor temperature
                self.temp_range[0],      # Jacket temperature  
                self.pressure_range[0],  # Pressure
                *[self.concentration_range[0]] * 4,  # Concentrations A,B,C,D
                *[self.flow_range[0]] * 3,           # Flow rates
                0.0,                     # Heat transfer coefficient
                0.0,                     # Mixing efficiency
                *[self.temp_range[0]] * 4,           # Wall temperatures
                0.0,                     # Residence time
                0.0,                     # Conversion rate
                0.0, 0.0                 # Safety margins
            ]),
            high=jnp.array([
                self.temp_range[1],      # Reactor temperature
                self.temp_range[1],      # Jacket temperature
                self.pressure_range[1],  # Pressure  
                *[self.concentration_range[1]] * 4,  # Concentrations A,B,C,D
                *[self.flow_range[1]] * 3,           # Flow rates
                1000.0,                  # Heat transfer coefficient
                1.0,                     # Mixing efficiency  
                *[self.temp_range[1]] * 4,           # Wall temperatures
                1000.0,                  # Residence time
                1.0,                     # Conversion rate
                100.0, 100.0             # Safety margins
            ]),
            dtype=jnp.float32
        )
        
        self.action_space = spaces.Box(
            low=jnp.array([0.0, 0.0, 0.0, self.temp_range[0], 0.0, 0.0]),
            high=jnp.array([self.flow_range[1], self.flow_range[1], 3000.0, self.temp_range[1], 100.0, 1.0]),
            dtype=jnp.float32
        )
        
        # Initialize state
        self.reset()
        
    def reset(self, seed: Optional[int] = None) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Reset reactor to initial operating conditions."""
        super().reset(seed=seed)
        
        # Initial state: steady-state conditions
        initial_temp = 323.15  # 50°C
        initial_pressure = 2e5  # 2 bar
        initial_ca = 2.0  # mol/L component A
        initial_cb = 1.5  # mol/L component B
        initial_cc = 0.1  # mol/L component C (product)
        initial_cd = 0.1  # mol/L component D (product)
        
        self.state = jnp.array([
            initial_temp,              # Reactor temperature
            initial_temp - 10.0,       # Jacket temperature (cooler)
            initial_pressure,          # Pressure
            initial_ca,                # Concentration A
            initial_cb,                # Concentration B  
            initial_cc,                # Concentration C
            initial_cd,                # Concentration D
            0.001,                     # Feed flow rate
            0.001,                     # Product flow rate
            0.005,                     # Coolant flow rate
            300.0,                     # Heat transfer coefficient
            0.8,                       # Mixing efficiency
            initial_temp,              # Wall temperature 1
            initial_temp,              # Wall temperature 2
            initial_temp,              # Wall temperature 3
            initial_temp,              # Wall temperature 4
            self.reactor_volume / 0.001,  # Residence time
            0.05,                      # Conversion rate
            50.0,                      # Temperature safety margin %
            60.0                       # Pressure safety margin %
        ])
        
        return self.state, {}
        
    def step(self, action: jnp.ndarray) -> Tuple[jnp.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one simulation step with high-fidelity reactor dynamics."""
        
        # Extract current state variables
        temp_reactor = self.state[0]
        temp_jacket = self.state[1] 
        pressure = self.state[2]
        ca, cb, cc, cd = self.state[3:7]
        flow_feed, flow_product, flow_coolant = self.state[7:10]
        heat_coeff = self.state[10]
        mixing_eff = self.state[11]
        wall_temps = self.state[12:16]
        residence_time = self.state[16]
        conversion = self.state[17]
        
        # Extract action variables
        feed_flow_action = action[0]
        coolant_flow_action = action[1]
        agitation_speed = action[2]
        feed_temp_setpoint = action[3]
        pressure_valve_opening = action[4]
        emergency_shutdown = action[5]
        
        # Emergency shutdown logic
        if emergency_shutdown > 0.5:
            # Immediate safety shutdown
            feed_flow_action = 0.0
            coolant_flow_action = self.flow_range[1]  # Maximum cooling
            agitation_speed = 0.0
            
        # Update flow rates with control dynamics
        flow_feed_new = flow_feed + 0.1 * (feed_flow_action - flow_feed)
        flow_coolant_new = flow_coolant + 0.2 * (coolant_flow_action - flow_coolant)
        
        # Reaction kinetics (Arrhenius equation)
        rate_constant = self.pre_exponential * jnp.exp(-self.activation_energy / (8.314 * temp_reactor))
        reaction_rate = rate_constant * ca * cb * mixing_eff
        
        # Mass balance for components
        ca_dot = (flow_feed_new * 5.0 - flow_product * ca) / self.reactor_volume - reaction_rate
        cb_dot = (flow_feed_new * 3.0 - flow_product * cb) / self.reactor_volume - reaction_rate  
        cc_dot = -flow_product * cc / self.reactor_volume + reaction_rate
        cd_dot = -flow_product * cd / self.reactor_volume + reaction_rate
        
        # Heat generation from reaction
        heat_generation = -self.heat_of_reaction * reaction_rate * self.reactor_volume
        
        # Heat transfer with jacket cooling
        heat_transfer_jacket = heat_coeff * self.jacket_area * (temp_reactor - temp_jacket)
        
        # Heat transfer through reactor walls (multi-section model)
        wall_heat_transfer = 0.0
        for i, wall_temp in enumerate(wall_temps):
            area_section = self.jacket_area / 4
            heat_flux = (self.wall_conductivity * area_section / self.wall_thickness) * (temp_reactor - wall_temp)
            wall_heat_transfer += heat_flux
            
        # Feed heating/cooling
        feed_heat = flow_feed_new * self.density * self.heat_capacity * (feed_temp_setpoint - temp_reactor)
        
        # Energy balance for reactor temperature
        total_mass = self.density * self.reactor_volume
        temp_reactor_dot = (
            heat_generation - heat_transfer_jacket - wall_heat_transfer + feed_heat
        ) / (total_mass * self.heat_capacity)
        
        # Jacket temperature dynamics
        coolant_heat_capacity = 4180.0  # J/kg·K
        jacket_mass = 100.0  # kg coolant in jacket
        temp_jacket_dot = (
            heat_transfer_jacket - flow_coolant_new * self.density * coolant_heat_capacity * (temp_jacket - 293.15)
        ) / (jacket_mass * coolant_heat_capacity)
        
        # Wall temperature dynamics (thermal diffusion)
        wall_temps_new = []
        for i, wall_temp in enumerate(wall_temps):
            # Heat conduction from reactor and to environment
            heat_from_reactor = (self.wall_conductivity / self.wall_thickness) * (temp_reactor - wall_temp)
            heat_to_environment = 10.0 * (wall_temp - 293.15)  # Natural convection
            
            wall_mass = 50.0  # kg per wall section
            wall_cp = 500.0   # J/kg·K (steel)
            
            wall_temp_dot = (heat_from_reactor - heat_to_environment) / (wall_mass * wall_cp)
            wall_temp_new = wall_temp + self.dt * wall_temp_dot
            wall_temps_new.append(wall_temp_new)
            
        # Pressure dynamics (ideal gas law + vapor pressure)
        gas_constant = 8.314  # J/mol·K
        total_moles = (ca + cb + cc + cd) * self.reactor_volume
        
        # Vapor pressure contribution (simplified)
        vapor_pressure = 1000.0 * jnp.exp(20.0 - 5000.0 / temp_reactor)
        
        # Pressure from gas phase and vapor
        pressure_new = (gas_constant * temp_reactor * total_moles / self.reactor_volume + 
                       vapor_pressure + 
                       self.pressure_range[0])  # Base pressure
        
        # Pressure relief valve effect
        if pressure_new > self.pressure_range[1] * 0.8:  # 80% of max pressure
            pressure_relief = pressure_valve_opening / 100.0 * (pressure_new - self.pressure_range[1] * 0.8)
            pressure_new = pressure_new - pressure_relief
            
        # Agitation effects on mixing and heat transfer
        mixing_eff_new = jnp.tanh(agitation_speed / 1000.0) * 0.9 + 0.1  # 0.1 to 1.0
        reynolds_number = agitation_speed * 0.1 * self.density / 0.001  # Simplified
        nusselt_number = 0.023 * (reynolds_number ** 0.8)
        heat_coeff_new = nusselt_number * 0.6 / 0.1  # Simplified correlation
        
        # Product flow rate (pressure driven)
        pressure_gradient = (pressure_new - 1e5) / 1e5  # Normalized pressure difference
        flow_product_new = 0.001 * (1.0 + 0.5 * pressure_gradient)
        
        # Update state variables
        new_ca = jnp.maximum(0.0, ca + self.dt * ca_dot)
        new_cb = jnp.maximum(0.0, cb + self.dt * cb_dot)
        new_cc = jnp.maximum(0.0, cc + self.dt * cc_dot)
        new_cd = jnp.maximum(0.0, cd + self.dt * cd_dot)
        
        new_temp_reactor = temp_reactor + self.dt * temp_reactor_dot
        new_temp_jacket = temp_jacket + self.dt * temp_jacket_dot
        
        # Calculate derived quantities
        residence_time_new = self.reactor_volume / jnp.maximum(flow_product_new, 1e-6)
        
        # Conversion calculation
        initial_ca = 2.0  # Reference initial concentration
        conversion_new = (initial_ca - new_ca) / initial_ca
        
        # Safety margins
        temp_safety_margin = (self.safety_temperature_limit - new_temp_reactor) / self.safety_temperature_limit * 100.0
        pressure_safety_margin = (self.safety_pressure_limit - pressure_new) / self.safety_pressure_limit * 100.0
        
        # Update complete state
        self.state = jnp.array([
            new_temp_reactor,
            new_temp_jacket,
            pressure_new,
            new_ca, new_cb, new_cc, new_cd,
            flow_feed_new, flow_product_new, flow_coolant_new,
            heat_coeff_new,
            mixing_eff_new,
            *wall_temps_new,
            residence_time_new,
            conversion_new,
            temp_safety_margin,
            pressure_safety_margin
        ])
        
        # Calculate reward
        reward = self._compute_reward(action)
        
        # Check safety violations and termination
        safety_metrics = self.get_safety_metrics()
        terminated = self._check_termination()
        truncated = self.episode_step >= self.max_episode_steps
        
        # Info dictionary with detailed metrics
        info = {
            "reaction_rate": float(reaction_rate),
            "heat_generation": float(heat_generation),
            "conversion": float(conversion_new),
            "residence_time": float(residence_time_new),
            "safety_metrics": safety_metrics,
            "emergency_shutdown": bool(emergency_shutdown > 0.5),
            "pressure_relief_active": bool(pressure_valve_opening > 0),
        }
        
        self.episode_step += 1
        
        return self.state, reward, terminated, truncated, info
        
    def _compute_reward(self, action: jnp.ndarray) -> float:
        """Compute reward based on reactor performance and safety."""
        
        temp_reactor = self.state[0]
        pressure = self.state[2]
        cc = self.state[5]  # Product concentration
        conversion = self.state[17]
        temp_safety_margin = self.state[18]
        pressure_safety_margin = self.state[19]
        
        # Production reward (higher product concentration and conversion)
        production_reward = 100.0 * (cc / 5.0 + conversion)
        
        # Safety reward (positive margins)
        safety_reward = (temp_safety_margin + pressure_safety_margin) / 2.0
        
        # Operating efficiency (avoid extreme conditions)
        temp_efficiency = 1.0 - jnp.abs(temp_reactor - 373.15) / 100.0  # Prefer ~100°C
        pressure_efficiency = 1.0 - jnp.abs(pressure - 3e5) / 1e5      # Prefer ~3 bar
        
        efficiency_reward = 50.0 * (temp_efficiency + pressure_efficiency)
        
        # Control penalty (smooth operation)
        control_penalty = -jnp.sum(jnp.abs(action[:-1])) * 10.0  # Exclude emergency shutdown
        
        # Emergency shutdown penalty
        emergency_penalty = -1000.0 if action[5] > 0.5 else 0.0
        
        total_reward = (
            production_reward + 
            safety_reward + 
            efficiency_reward + 
            control_penalty + 
            emergency_penalty
        )
        
        return float(total_reward)
        
    def _check_termination(self) -> bool:
        """Check if episode should terminate due to safety violations."""
        
        temp_reactor = self.state[0]
        pressure = self.state[2]
        
        # Critical safety violations
        temp_violation = temp_reactor > self.safety_temperature_limit
        pressure_violation = pressure > self.safety_pressure_limit
        
        # Concentration violations (reactor runaway)
        cc = self.state[5]
        runaway_condition = cc > 8.0  # Excessive product formation
        
        return bool(temp_violation or pressure_violation or runaway_condition)
        
    def get_safety_metrics(self) -> SafetyMetrics:
        """Get current safety metrics for the reactor."""
        
        temp_reactor = self.state[0]
        pressure = self.state[2]
        temp_safety_margin = self.state[18]
        pressure_safety_margin = self.state[19]
        
        # Safety constraint violations
        violations = []
        
        if temp_reactor > self.safety_temperature_limit:
            violations.append("temperature_limit")
            
        if pressure > self.safety_pressure_limit:
            violations.append("pressure_limit")
            
        if temp_safety_margin < 10.0:  # Less than 10% margin
            violations.append("temperature_margin")
            
        if pressure_safety_margin < 10.0:  # Less than 10% margin
            violations.append("pressure_margin")
            
        return SafetyMetrics(
            total_violations=len(violations),
            violation_types=violations,
            safety_score=float(jnp.minimum(temp_safety_margin, pressure_safety_margin)),
            constraint_satisfaction=len(violations) == 0
        )
        
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render reactor state visualization."""
        
        if mode == "human":
            print(f"=== Chemical Reactor State ===")
            print(f"Temperature: {self.state[0]:.1f} K ({self.state[0]-273.15:.1f}°C)")
            print(f"Pressure: {self.state[2]:.0f} Pa ({self.state[2]/1e5:.1f} bar)")
            print(f"Concentrations: A={self.state[3]:.2f}, B={self.state[4]:.2f}, C={self.state[5]:.2f}, D={self.state[6]:.2f} mol/L")
            print(f"Conversion: {self.state[17]:.1%}")
            print(f"Safety margins: T={self.state[18]:.1f}%, P={self.state[19]:.1f}%")
            print("=" * 30)
            
        return None