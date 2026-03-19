"""Base class for industrial offline RL environments."""
import abc
from typing import Dict, Optional, Tuple

import numpy as np

from neorl_gym.spaces import Box


class IndustrialEnv(abc.ABC):
    """Abstract base for industrial control environments.

    Follows a gym-like interface (step / reset) but has NO gymnasium dependency.
    All environments support generating offline datasets for batch RL.

    Subclasses must implement:
        _get_initial_state() -> np.ndarray
        _dynamics(state, action) -> np.ndarray
        _compute_reward(state, action, next_state) -> float
        _is_done(state) -> bool
        generate_dataset(n_transitions, policy) -> dict
    """

    def __init__(
        self,
        obs_low: np.ndarray,
        obs_high: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
        max_episode_steps: int = 200,
        dt: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        obs_shape = obs_low.shape
        act_shape = action_low.shape

        self.observation_space = Box(obs_low, obs_high, obs_shape)
        self.action_space = Box(action_low, action_high, act_shape)

        self.max_episode_steps = max_episode_steps
        self.dt = dt
        self.rng = rng or np.random.default_rng()

        self._state: Optional[np.ndarray] = None
        self._step_count = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _get_initial_state(self) -> np.ndarray:
        """Sample a valid initial state."""

    @abc.abstractmethod
    def _dynamics(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Compute next state from (state, action). Pure function."""

    @abc.abstractmethod
    def _compute_reward(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray
    ) -> float:
        """Compute scalar reward for a transition."""

    @abc.abstractmethod
    def _is_done(self, state: np.ndarray) -> bool:
        """Return True if state is terminal (unsafe / goal reached)."""

    # ------------------------------------------------------------------
    # Gym-like interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._state = self._get_initial_state().astype(np.float32)
        self._step_count = 0
        return self._state.copy(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the environment.

        Returns:
            obs, reward, terminated, truncated, info
        """
        assert self._state is not None, "Call reset() before step()."
        action = np.asarray(action, dtype=np.float32)
        action = self.action_space.clip(action)

        next_state = self._dynamics(self._state, action).astype(np.float32)
        reward = float(self._compute_reward(self._state, action, next_state))
        terminated = self._is_done(next_state)

        self._state = next_state
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps

        return next_state.copy(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Offline data generation
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        n_transitions: int = 50_000,
        policy: str = "mixed",
    ) -> Dict[str, np.ndarray]:
        """Generate an offline dataset by rolling out a policy.

        Args:
            n_transitions: Total (s, a, r, s') transitions to collect.
            policy: One of 'expert', 'medium', 'random', 'mixed'.

        Returns:
            Dictionary with keys: observations, actions, rewards,
            next_observations, terminals.
        """
        observations, actions, rewards, next_observations, terminals = [], [], [], [], []

        obs, _ = self.reset()
        collected = 0

        while collected < n_transitions:
            action = self._policy_action(obs, policy)
            next_obs, reward, terminated, truncated, _ = self.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            terminals.append(terminated)

            collected += 1
            obs = next_obs

            if terminated or truncated:
                obs, _ = self.reset()

        return {
            "observations": np.array(observations, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "next_observations": np.array(next_observations, dtype=np.float32),
            "terminals": np.array(terminals, dtype=np.float32),
        }

    def _policy_action(self, obs: np.ndarray, policy: str) -> np.ndarray:
        """Default policy dispatch — subclasses override for domain-specific expert."""
        if policy == "random":
            return self.action_space.sample()
        if policy == "expert":
            return self._expert_action(obs)
        if policy == "medium":
            if self.rng.random() < 0.6:
                return self._expert_action(obs)
            return self.action_space.sample()
        # mixed
        if self.rng.random() < 0.4:
            return self._expert_action(obs)
        return self.action_space.sample()

    def _expert_action(self, obs: np.ndarray) -> np.ndarray:
        """Override in subclass for domain-specific expert policy."""
        return self.action_space.sample()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def state(self) -> Optional[np.ndarray]:
        return self._state

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"obs_shape={self.observation_space.shape}, "
            f"act_shape={self.action_space.shape})"
        )
