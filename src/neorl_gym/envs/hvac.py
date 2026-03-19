"""HVACControlEnv — temperature regulation via heater power control.

State  : (temp, setpoint, outside_temp)   shape (3,)
Action : (heater_power,)                  in [-1, 1]  (fraction of max power)
Reward : -(temp_error)²  — penalises deviation from setpoint
"""

import numpy as np
from neorl_gym.envs.base import IndustrialEnv

# Physical constants
MAX_HEATER_WATTS = 5000.0   # W
THERMAL_MASS = 1e5          # J/K  (room thermal capacity)
HEAT_LOSS_COEFF = 50.0      # W/K  (UA value, loss to outside)

OBS_LOW  = np.array([-20.0,  15.0, -30.0], dtype=np.float32)
OBS_HIGH = np.array([ 45.0,  30.0,  45.0], dtype=np.float32)
ACT_LOW  = np.array([-1.0],               dtype=np.float32)
ACT_HIGH = np.array([ 1.0],               dtype=np.float32)


class HVACControlEnv(IndustrialEnv):
    """HVAC temperature regulation environment.

    The agent controls a heater (positive = heat, negative = cool)
    to drive indoor temperature toward a time-varying setpoint,
    against an outdoor temperature disturbance.

    Episodes are 200 steps (~3 h 20 min at dt=60 s).
    Terminal: indoor temp exceeds hard bounds [10 °C, 40 °C].
    """

    def __init__(self, dt: float = 60.0, **kwargs):
        super().__init__(
            obs_low=OBS_LOW,
            obs_high=OBS_HIGH,
            action_low=ACT_LOW,
            action_high=ACT_HIGH,
            max_episode_steps=200,
            dt=dt,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # IndustrialEnv abstract methods
    # ------------------------------------------------------------------

    def _get_initial_state(self) -> np.ndarray:
        temp = self.rng.uniform(16.0, 28.0)
        setpoint = self.rng.uniform(19.0, 23.0)
        outside = self.rng.uniform(-5.0, 35.0)
        return np.array([temp, setpoint, outside], dtype=np.float32)

    def _dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        temp, setpoint, outside = state
        heater_fraction = float(action[0])

        # Net heat flow
        heater_power = heater_fraction * MAX_HEATER_WATTS      # W (can be negative = cooling)
        heat_loss = HEAT_LOSS_COEFF * (temp - outside)          # W
        net_heat = heater_power - heat_loss                     # W

        # Euler integration
        dtemp = net_heat / THERMAL_MASS * self.dt
        new_temp = temp + dtemp + self.rng.normal(0.0, 0.05)   # small sensor noise

        # Setpoint drifts slowly (±0.01 °C/step)
        new_setpoint = setpoint + self.rng.normal(0.0, 0.01)
        new_setpoint = float(np.clip(new_setpoint, 19.0, 23.0))

        # Outside temperature follows a slow random walk
        new_outside = outside + self.rng.normal(0.0, 0.2)
        new_outside = float(np.clip(new_outside, -30.0, 45.0))

        return np.array([new_temp, new_setpoint, new_outside], dtype=np.float32)

    def _compute_reward(self, state, action, next_state) -> float:
        temp, setpoint, _ = next_state
        error = temp - setpoint
        return -float(error ** 2)

    def _is_done(self, state: np.ndarray) -> bool:
        temp = state[0]
        return bool(temp < 10.0 or temp > 40.0)

    # ------------------------------------------------------------------
    # Expert policy: bang-bang proportional controller
    # ------------------------------------------------------------------

    def _expert_action(self, obs: np.ndarray) -> np.ndarray:
        temp, setpoint, _ = obs
        error = setpoint - temp
        # Proportional control, clipped to [-1, 1]
        action = np.clip([error * 0.5], -1.0, 1.0)
        action += self.rng.normal(0.0, 0.05, size=1)
        return np.clip(action, -1.0, 1.0).astype(np.float32)
