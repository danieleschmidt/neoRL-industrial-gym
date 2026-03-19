"""BCAgent — Behaviour Cloning baseline for offline RL.

Trains a neural network to directly imitate actions from an offline dataset.
This is the simplest offline RL baseline: no value estimation, no importance
weighting — just supervised regression from observations to actions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from neorl_gym.offline.dataset import OfflineDataset


class _MLP(nn.Module):
    """Simple MLP for action prediction."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),   # outputs in (-1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BCAgent:
    """Behaviour Cloning agent.

    Trains a policy π(a|s) by minimising MSE loss against dataset actions.

    Args:
        obs_dim:    Observation dimension.
        act_dim:    Action dimension.
        act_low:    Lower bound of action space (numpy array, shape (act_dim,)).
        act_high:   Upper bound of action space (numpy array, shape (act_dim,)).
        hidden_dim: Width of hidden layers.
        lr:         Learning rate.
        device:     'cpu' or 'cuda'.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low: np.ndarray,
        act_high: np.ndarray,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        device: str = "cpu",
    ):
        self.obs_dim  = obs_dim
        self.act_dim  = act_dim
        self.device   = torch.device(device)

        # Action scaling: network outputs in (-1, 1), we scale to [low, high]
        self.act_low   = torch.tensor(act_low,  dtype=torch.float32, device=self.device)
        self.act_high  = torch.tensor(act_high, dtype=torch.float32, device=self.device)
        self.act_scale = (self.act_high - self.act_low) / 2.0
        self.act_shift = (self.act_high + self.act_low) / 2.0

        self.policy  = _MLP(obs_dim, act_dim, hidden_dim).to(self.device)
        self.opt     = optim.Adam(self.policy.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.train_losses: list[float] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        dataset: OfflineDataset,
        epochs: int = 50,
        batch_size: int = 256,
        verbose: bool = True,
    ) -> list[float]:
        """Train the BC policy on the offline dataset.

        Args:
            dataset:    OfflineDataset instance.
            epochs:     Training epochs.
            batch_size: Mini-batch size.
            verbose:    Print progress.

        Returns:
            List of per-epoch mean losses.
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        self.policy.train()
        epoch_losses = []

        # Normalise actions to (-1, 1) for training
        for ep in range(1, epochs + 1):
            batch_losses = []
            for obs, act, *_ in loader:
                obs = obs.to(self.device)
                act = act.to(self.device)

                # Normalise actions to tanh output range (-1, 1)
                act_norm = (act - self.act_shift) / (self.act_scale + 1e-8)
                act_norm = torch.clamp(act_norm, -1.0, 1.0)

                pred = self.policy(obs)
                loss = self.loss_fn(pred, act_norm)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                batch_losses.append(loss.item())

            mean_loss = float(np.mean(batch_losses))
            epoch_losses.append(mean_loss)
            self.train_losses.append(mean_loss)

            if verbose and (ep % max(1, epochs // 10) == 0 or ep == 1):
                print(f"  Epoch {ep:3d}/{epochs} — loss: {mean_loss:.4f}")

        return epoch_losses

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Predict action for a single observation.

        Args:
            obs: numpy array of shape (obs_dim,).

        Returns:
            action numpy array of shape (act_dim,) in [act_low, act_high].
        """
        self.policy.eval()
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_norm = self.policy(obs_t).squeeze(0)
        action = action_norm * self.act_scale + self.act_shift
        return action.cpu().numpy()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, env, n_episodes: int = 10, max_steps: int = 200) -> dict:
        """Evaluate the BC policy in an environment.

        Args:
            env:        An IndustrialEnv instance.
            n_episodes: Number of evaluation episodes.
            max_steps:  Maximum steps per episode.

        Returns:
            Dict with mean_return, std_return, mean_length.
        """
        returns = []
        lengths = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_return = 0.0
            step = 0
            done = False

            while not done and step < max_steps:
                action = self.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += reward
                done = terminated or truncated
                step += 1

            returns.append(ep_return)
            lengths.append(step)

        return {
            "mean_return": float(np.mean(returns)),
            "std_return":  float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "episodes":    n_episodes,
        }

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
