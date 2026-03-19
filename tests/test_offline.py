"""Tests for OfflineDataset."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
import tempfile

from neorl_gym import OfflineDataset


def _make_raw(n=1000, obs_dim=3, act_dim=1):
    rng = np.random.default_rng(0)
    return {
        "observations":      rng.standard_normal((n, obs_dim)).astype(np.float32),
        "actions":           rng.uniform(-1, 1, (n, act_dim)).astype(np.float32),
        "rewards":           rng.standard_normal(n).astype(np.float32) * 10,
        "next_observations": rng.standard_normal((n, obs_dim)).astype(np.float32),
        "terminals":         (rng.random(n) < 0.05).astype(np.float32),
    }


def test_from_dict_basic():
    data = _make_raw(500)
    ds = OfflineDataset.from_dict(data)
    assert len(ds) == 500


def test_from_dict_min_reward_filter():
    data = _make_raw(1000)
    ds = OfflineDataset.from_dict(data, min_reward=0.0)
    rewards = ds.rewards.numpy()
    assert float(rewards.min()) >= 0.0


def test_from_dict_safe_only():
    data = _make_raw(1000)
    ds = OfflineDataset.from_dict(data, safe_only=True)
    assert float(ds.terminals.sum()) == 0.0


def test_from_dict_combined_filters():
    data = _make_raw(2000)
    ds = OfflineDataset.from_dict(data, min_reward=0.0, safe_only=True)
    assert float(ds.rewards.min()) >= 0.0
    assert float(ds.terminals.sum()) == 0.0


def test_getitem_shapes():
    data = _make_raw(100, obs_dim=4, act_dim=2)
    ds = OfflineDataset.from_dict(data)
    obs, act, rew, next_obs, done = ds[0]
    assert obs.shape  == (4,)
    assert act.shape  == (2,)
    assert rew.shape  == ()
    assert next_obs.shape == (4,)
    assert done.shape == ()


def test_stats():
    data = _make_raw(200)
    ds = OfflineDataset.from_dict(data)
    stats = ds.stats()
    assert "reward_mean" in stats
    assert "n_transitions" in stats
    assert stats["n_transitions"] == len(ds)


def test_save_load_roundtrip():
    data = _make_raw(100)
    ds = OfflineDataset.from_dict(data)
    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        ds.save(f.name)
        ds2 = OfflineDataset.load(f.name)
    assert len(ds2) == len(ds)
    np.testing.assert_allclose(ds.rewards.numpy(), ds2.rewards.numpy(), atol=1e-6)


def test_torch_dataloader_compatible():
    import torch
    from torch.utils.data import DataLoader
    data = _make_raw(200, obs_dim=3, act_dim=1)
    ds = OfflineDataset.from_dict(data)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    obs, act, rew, next_obs, done = batch
    assert obs.shape == (32, 3)
    assert act.shape == (32, 1)
