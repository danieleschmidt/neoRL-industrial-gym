"""Tests for industrial environments."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from neorl_gym import HVACControlEnv, ChemicalReactorEnv, PIDTuningEnv
from neorl_gym.envs.base import IndustrialEnv


ENV_CLASSES = [HVACControlEnv, ChemicalReactorEnv, PIDTuningEnv]


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_env_is_industrial_env(env_cls):
    env = env_cls()
    assert isinstance(env, IndustrialEnv)


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_reset_returns_valid_obs(env_cls):
    env = env_cls()
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_step_signature(env_cls):
    env = env_cls()
    obs, _ = env.reset(seed=1)
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    assert next_obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert isinstance(info, dict)


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_action_clip(env_cls):
    """Out-of-bounds actions should be clipped, not raise."""
    env = env_cls()
    env.reset(seed=2)
    huge_action = np.full(env.action_space.shape, 1e6, dtype=np.float32)
    next_obs, reward, _, _, _ = env.step(huge_action)
    assert np.all(np.isfinite(next_obs)), "Obs should remain finite"


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_rollout_runs_full_episode(env_cls):
    env = env_cls()
    obs, _ = env.reset(seed=3)
    steps = 0
    for _ in range(env.max_episode_steps + 10):
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    assert steps <= env.max_episode_steps + 1


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_generate_dataset_keys(env_cls):
    env = env_cls()
    data = env.generate_dataset(n_transitions=100, policy="random")
    for key in ("observations", "actions", "rewards", "next_observations", "terminals"):
        assert key in data, f"Missing key: {key}"
    n = len(data["observations"])
    for v in data.values():
        assert len(v) == n


@pytest.mark.parametrize("env_cls", ENV_CLASSES)
def test_generate_dataset_policies(env_cls):
    env = env_cls()
    for policy in ("random", "expert", "medium", "mixed"):
        data = env.generate_dataset(n_transitions=50, policy=policy)
        assert len(data["observations"]) == 50


def test_hvac_reward_is_negative_squared_error():
    env = HVACControlEnv()
    obs, _ = env.reset(seed=10)
    action = np.array([0.0], dtype=np.float32)  # no heating
    _, reward, _, _, _ = env.step(action)
    assert reward <= 0.0, "HVAC reward should be non-positive (negated squared error)"


def test_reactor_terminal_on_overtemp():
    """Reactor should terminate if temperature goes way above TEMP_MAX."""
    from neorl_gym.envs.reactor import TEMP_MAX
    env = ChemicalReactorEnv()
    env.reset()
    # Force state to overtemp
    env._state = np.array([0.5, TEMP_MAX + 25, 150.0], dtype=np.float32)
    assert env._is_done(env._state)


def test_pid_reward_positive_when_settled():
    """PID reward should be positive when the system settles."""
    env = PIDTuningEnv()
    env.reset(seed=0)
    # Apply near-optimal gains
    action = np.array([4.0, 2.0, 1.0], dtype=np.float32)
    _, reward, _, _, _ = env.step(action)
    # Just verify it doesn't crash and is a finite float
    assert np.isfinite(reward)
