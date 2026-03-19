"""PIDTuningEnv — learn PID gains from data, reward = settling_time⁻¹.

State  : (error, integral, derivative)   shape (3,)
Action : (Kp, Ki, Kd)                    each in [0, 10]
Reward : settling_time⁻¹  (fast settling is better)

The environment simulates a 2nd-order plant (mass-spring-damper).
Each step, the agent proposes PID gains; the env simulates the closed-loop
response for `sim_steps` timesteps and returns the settling quality.
"""

import numpy as np
from neorl_gym.envs.base import IndustrialEnv

# Plant: ẍ + 2ζωₙẋ + ωₙ²x = ωₙ²·u
OMEGA_N = 2.0     # natural frequency rad/s
ZETA    = 0.3     # damping ratio (underdamped — hard to tune)

SIM_DT     = 0.01   # inner simulation step (s)
SIM_STEPS  = 300    # inner steps per outer step (~3 s of plant time)
SETPOINT   = 1.0    # step setpoint

OBS_LOW  = np.array([-5.0, -20.0, -20.0], dtype=np.float32)
OBS_HIGH = np.array([ 5.0,  20.0,  20.0], dtype=np.float32)
ACT_LOW  = np.array([0.0, 0.0, 0.0], dtype=np.float32)
ACT_HIGH = np.array([10.0, 10.0, 10.0], dtype=np.float32)


class PIDTuningEnv(IndustrialEnv):
    """PID auto-tuning environment.

    Each episode the agent iteratively refines PID gains (Kp, Ki, Kd).
    The state reflects the current error signature; the reward is the
    reciprocal of settling time (or negative ISE if not settled).
    """

    def __init__(self, dt: float = 1.0, **kwargs):
        super().__init__(
            obs_low=OBS_LOW,
            obs_high=OBS_HIGH,
            action_low=ACT_LOW,
            action_high=ACT_HIGH,
            max_episode_steps=100,
            dt=dt,
            **kwargs,
        )
        self._plant_x  = 0.0   # plant position
        self._plant_xd = 0.0   # plant velocity

    # ------------------------------------------------------------------

    def _get_initial_state(self) -> np.ndarray:
        self._plant_x  = 0.0
        self._plant_xd = 0.0
        error = SETPOINT - self._plant_x
        return np.array([error, 0.0, 0.0], dtype=np.float32)

    def _simulate_closed_loop(self, Kp: float, Ki: float, Kd: float):
        """Run inner closed-loop simulation. Returns (settling_time, ise, trajectory)."""
        x  = self._plant_x
        xd = self._plant_xd
        integral = 0.0
        prev_err = SETPOINT - x
        settling_time = None
        settle_thresh = 0.05  # 5% of setpoint
        ise = 0.0
        traj = []

        for i in range(SIM_STEPS):
            err = SETPOINT - x
            integral  += err * SIM_DT
            derivative = (err - prev_err) / SIM_DT
            prev_err   = err

            # PID output = control effort
            u = Kp * err + Ki * integral + Kd * derivative
            u = np.clip(u, -10.0, 10.0)

            # Plant: mass-spring-damper
            xdd = -2 * ZETA * OMEGA_N * xd - OMEGA_N**2 * x + OMEGA_N**2 * u
            xd  = xd + xdd * SIM_DT
            x   = x  + xd  * SIM_DT

            ise += err**2 * SIM_DT
            traj.append(x)

            if settling_time is None and abs(err) < settle_thresh * SETPOINT:
                settling_time = i * SIM_DT

        # Update plant state for next outer step
        self._plant_x  = x
        self._plant_xd = xd
        # Final error signature for next obs
        final_err = SETPOINT - x
        return settling_time, ise, final_err, derivative, integral

    def _dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        Kp, Ki, Kd = float(action[0]), float(action[1]), float(action[2])
        settling_time, ise, final_err, deriv, integral = self._simulate_closed_loop(Kp, Ki, Kd)
        # Clamp integral to obs space
        integral = np.clip(integral, -20.0, 20.0)
        deriv    = np.clip(deriv,    -20.0, 20.0)
        return np.array([final_err, integral, deriv], dtype=np.float32)

    def _compute_reward(self, state, action, next_state) -> float:
        Kp, Ki, Kd = float(action[0]), float(action[1]), float(action[2])
        settling_time, ise, _, _, _ = self._simulate_closed_loop(Kp, Ki, Kd)
        if settling_time is not None:
            return float(1.0 / (settling_time + 0.1))
        else:
            # Didn't settle — penalise with negative ISE proxy
            return float(-ise * 0.01)

    def _is_done(self, state: np.ndarray) -> bool:
        error = abs(state[0])
        return bool(error < 0.02)  # very tight convergence → episode done

    def _expert_action(self, obs: np.ndarray) -> np.ndarray:
        # Ziegler-Nichols-inspired nominal gains
        Kp = 4.0 + self.rng.normal(0.0, 0.3)
        Ki = 2.0 + self.rng.normal(0.0, 0.2)
        Kd = 1.0 + self.rng.normal(0.0, 0.1)
        return np.clip([Kp, Ki, Kd], 0.0, 10.0).astype(np.float32)
