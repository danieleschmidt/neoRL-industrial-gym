"""Tests for BCAgent."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
import tempfile

from neorl_gym import BCAgent, OfflineDataset, HVACControlEnv


def _make_bc(obs_dim=3, act_dim=1):
    low  = np.full(act_dim, -1.0, dtype=np.float32)
    high = np.full(act_dim,  1.0, dtype=np.float32)
    return BCAgent(obs_dim, act_dim, low, high)


def _make_dataset(n=500, obs_dim=3, act_dim=1):
    rng = np.random.default_rng(42)
    data = {
        "observations":      rng.standard_normal((n, obs_dim)).astype(np.float32),
        "actions":           rng.uniform(-1, 1, (n, act_dim)).astype(np.float32),
        "rewards":           rng.standard_normal(n).astype(np.float32),
        "next_observations": rng.standard_normal((n, obs_dim)).astype(np.float32),
        "terminals":         np.zeros(n, dtype=np.float32),
    }
    return OfflineDataset.from_dict(data)


def test_bc_predict_shape():
    agent = _make_bc(obs_dim=3, act_dim=1)
    obs = np.random.randn(3).astype(np.float32)
    act = agent.predict(obs)
    assert act.shape == (1,)


def test_bc_predict_in_action_bounds():
    low  = np.array([-1.0], dtype=np.float32)
    high = np.array([ 1.0], dtype=np.float32)
    agent = BCAgent(3, 1, low, high)
    for _ in range(20):
        obs = np.random.randn(3).astype(np.float32)
        act = agent.predict(obs)
        assert np.all(act >= low - 1e-5) and np.all(act <= high + 1e-5)


def test_bc_train_reduces_loss():
    agent   = _make_bc()
    dataset = _make_dataset()
    losses  = agent.train(dataset, epochs=10, batch_size=64, verbose=False)
    assert losses[-1] < losses[0], "Loss should decrease during training"


def test_bc_evaluate_returns_dict():
    env   = HVACControlEnv()
    agent = BCAgent(
        obs_dim=3, act_dim=1,
        act_low=env.action_space.low,
        act_high=env.action_space.high,
    )
    # Untrained, but should not crash
    metrics = agent.evaluate(env, n_episodes=3, max_steps=50)
    for key in ("mean_return", "std_return", "mean_length"):
        assert key in metrics


def test_bc_save_load():
    agent = _make_bc(obs_dim=3, act_dim=1)
    obs   = np.random.randn(3).astype(np.float32)
    act_before = agent.predict(obs).copy()

    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        agent.save(f.name)
        agent2 = _make_bc(obs_dim=3, act_dim=1)
        agent2.load(f.name)

    act_after = agent2.predict(obs)
    np.testing.assert_allclose(act_before, act_after, atol=1e-5)
