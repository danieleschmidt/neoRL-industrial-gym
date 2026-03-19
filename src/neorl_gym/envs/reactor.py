"""ChemicalReactorEnv — yield optimisation with safety constraints.

State  : (concentration mol/L, temperature K, pressure kPa)  shape (3,)
Action : (feed_rate,)  in [-1, 1]  (fractional adjustment to feed)
Reward : product concentration gain, minus safety violation penalties

Safety constraints:
  - Temperature must stay below 380 K  (exothermic runaway risk)
  - Pressure must stay below 300 kPa   (vessel integrity)
"""

import numpy as np
from neorl_gym.envs.base import IndustrialEnv

# Operating bounds
TEMP_TARGET = 330.0   # K
TEMP_MAX    = 380.0   # K  — hard safety limit
PRESS_MAX   = 300.0   # kPa — hard safety limit
PRESS_NOM   = 150.0   # kPa — nominal operating pressure
CONC_MAX    = 2.0     # mol/L

OBS_LOW  = np.array([0.0,   250.0,  50.0], dtype=np.float32)
OBS_HIGH = np.array([2.0,   400.0, 350.0], dtype=np.float32)
ACT_LOW  = np.array([-1.0],               dtype=np.float32)
ACT_HIGH = np.array([ 1.0],               dtype=np.float32)


class ChemicalReactorEnv(IndustrialEnv):
    """CSTR-inspired reactor for yield optimisation.

    The agent adjusts feed rate to maximise product concentration
    while keeping temperature and pressure within safe bounds.

    Episodes are 200 steps.
    Terminal: safety constraint violated.
    """

    def __init__(self, dt: float = 1.0, **kwargs):
        super().__init__(
            obs_low=OBS_LOW,
            obs_high=OBS_HIGH,
            action_low=ACT_LOW,
            action_high=ACT_HIGH,
            max_episode_steps=200,
            dt=dt,
            **kwargs,
        )
        self._base_feed = 1.0   # mol/L/s nominal feed rate
        self._k0 = 0.5          # pre-exponential factor
        self._Ea_R = 5000.0     # activation energy / R (K)
        self._dH = -2e4         # heat of reaction (J/mol)
        self._Cp_rho = 4.0e3    # density × heat capacity (J/L/K)
        self._V = 10.0          # reactor volume (L)
        self._UA = 200.0        # heat transfer coefficient (J/s/K)
        self._T_cool = 300.0    # coolant temperature (K)

    # ------------------------------------------------------------------

    def _get_initial_state(self) -> np.ndarray:
        conc = self.rng.uniform(0.3, 0.8)
        temp = self.rng.uniform(310.0, 350.0)
        press = self.rng.uniform(120.0, 180.0)
        return np.array([conc, temp, press], dtype=np.float32)

    def _dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        conc, temp, press = state
        feed_adj = float(action[0])

        # Feed rate: base + fractional adjustment
        feed_rate = self._base_feed * (1.0 + 0.5 * feed_adj)
        feed_rate = max(0.1, feed_rate)

        # Reaction rate (Arrhenius)
        k = self._k0 * np.exp(-self._Ea_R / max(temp, 1.0))
        r = k * conc                       # mol/L/s

        # Concentration ODE
        d_conc = (feed_rate / self._V) * (1.0 - conc) - r
        new_conc = conc + d_conc * self.dt
        new_conc = max(0.0, new_conc)

        # Temperature ODE (CSTR energy balance)
        heat_gen = -self._dH * r * self._V          # J/s
        heat_remove = self._UA * (temp - self._T_cool)  # J/s
        d_temp = (heat_gen - heat_remove) / (self._Cp_rho * self._V)
        new_temp = temp + d_temp * self.dt
        new_temp += self.rng.normal(0.0, 0.2)

        # Pressure scales with temperature (ideal gas)
        new_press = press * (new_temp / max(temp, 1.0))
        new_press += self.rng.normal(0.0, 0.5)
        new_press = max(50.0, new_press)

        return np.array([new_conc, new_temp, new_press], dtype=np.float32)

    def _compute_reward(self, state, action, next_state) -> float:
        conc, temp, press = next_state
        # Main objective: maximise concentration (yield proxy)
        reward = float(conc) * 10.0

        # Safety penalties
        if temp > TEMP_MAX:
            reward -= 50.0 + (temp - TEMP_MAX) * 2.0
        if press > PRESS_MAX:
            reward -= 50.0 + (press - PRESS_MAX) * 2.0

        # Small penalty for large feed rate changes (smooth operation)
        reward -= 0.1 * float(action[0]) ** 2

        return reward

    def _is_done(self, state: np.ndarray) -> bool:
        _, temp, press = state
        return bool(temp > TEMP_MAX + 20 or press > PRESS_MAX + 50)

    def _expert_action(self, obs: np.ndarray) -> np.ndarray:
        conc, temp, press = obs
        # Heuristic: cool down by reducing feed if overheating,
        # ramp up feed if temperature is low and there's room for yield
        temp_margin = (TEMP_MAX - temp) / TEMP_MAX
        if temp > TEMP_TARGET:
            action = np.array([-temp_margin * 0.8], dtype=np.float32)
        else:
            action = np.array([temp_margin * 0.4], dtype=np.float32)
        action += self.rng.normal(0.0, 0.05, size=1).astype(np.float32)
        return np.clip(action, -1.0, 1.0)
