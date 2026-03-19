"""OfflineDataset — stores (s, a, r, s') transitions with quality filtering."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class OfflineDataset(Dataset):
    """Offline RL dataset wrapping pre-collected transitions.

    Supports quality filtering to remove unsafe or poor trajectories:
      - ``min_reward``: drop transitions with reward below threshold
      - ``max_reward``: drop transitions with reward above threshold (outlier clamp)
      - ``safe_only``: drop transitions that are terminal (assumed unsafe)

    Usage::

        dataset = OfflineDataset.from_dict(raw_data, min_reward=-50.0, safe_only=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
        for obs, act, rew, next_obs, done in loader:
            ...

    ``raw_data`` must be a dict with keys:
        observations, actions, rewards, next_observations, terminals
    each a numpy array with the same leading dimension (n_transitions).
    """

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
    ):
        n = len(observations)
        assert len(actions)          == n
        assert len(rewards)          == n
        assert len(next_observations) == n
        assert len(terminals)        == n

        self.observations      = torch.tensor(observations,       dtype=torch.float32)
        self.actions           = torch.tensor(actions,            dtype=torch.float32)
        self.rewards           = torch.tensor(rewards,            dtype=torch.float32)
        self.next_observations = torch.tensor(next_observations,  dtype=torch.float32)
        self.terminals         = torch.tensor(terminals,          dtype=torch.float32)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx) -> Tuple:
        return (
            self.observations[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_observations[idx],
            self.terminals[idx],
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, np.ndarray],
        *,
        min_reward: Optional[float] = None,
        max_reward: Optional[float] = None,
        safe_only: bool = False,
    ) -> "OfflineDataset":
        """Create a filtered dataset from a raw transition dictionary.

        Args:
            data: Dict with keys observations, actions, rewards,
                  next_observations, terminals.
            min_reward: Drop transitions with reward < min_reward.
            max_reward: Drop transitions with reward > max_reward.
            safe_only: If True, drop terminal transitions (assumed unsafe).

        Returns:
            OfflineDataset with filtered transitions.
        """
        rewards   = np.asarray(data["rewards"],   dtype=np.float32)
        terminals = np.asarray(data["terminals"], dtype=np.float32)

        mask = np.ones(len(rewards), dtype=bool)

        if min_reward is not None:
            mask &= rewards >= min_reward
        if max_reward is not None:
            mask &= rewards <= max_reward
        if safe_only:
            mask &= terminals < 0.5   # keep non-terminal transitions

        n_before = len(rewards)
        n_after  = int(mask.sum())
        pct_kept = 100.0 * n_after / max(n_before, 1)
        print(
            f"[OfflineDataset] {n_after}/{n_before} transitions kept "
            f"({pct_kept:.1f}%) after quality filtering."
        )

        return cls(
            observations      = np.asarray(data["observations"],      dtype=np.float32)[mask],
            actions           = np.asarray(data["actions"],           dtype=np.float32)[mask],
            rewards           = rewards[mask],
            next_observations = np.asarray(data["next_observations"], dtype=np.float32)[mask],
            terminals         = terminals[mask],
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, float]:
        """Return basic dataset statistics."""
        r = self.rewards.numpy()
        return {
            "n_transitions": len(self),
            "reward_mean":   float(r.mean()),
            "reward_std":    float(r.std()),
            "reward_min":    float(r.min()),
            "reward_max":    float(r.max()),
            "terminal_rate": float(self.terminals.mean()),
        }

    def save(self, path: str) -> None:
        """Save dataset to a .npz file."""
        np.savez_compressed(
            path,
            observations      = self.observations.numpy(),
            actions           = self.actions.numpy(),
            rewards           = self.rewards.numpy(),
            next_observations = self.next_observations.numpy(),
            terminals         = self.terminals.numpy(),
        )
        print(f"[OfflineDataset] Saved {len(self)} transitions to {path}")

    @classmethod
    def load(cls, path: str) -> "OfflineDataset":
        """Load dataset from a .npz file."""
        d = np.load(path)
        return cls(
            observations      = d["observations"],
            actions           = d["actions"],
            rewards           = d["rewards"],
            next_observations = d["next_observations"],
            terminals         = d["terminals"],
        )
