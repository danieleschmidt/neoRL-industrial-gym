#!/usr/bin/env python3
"""Demo: generate offline data, train BCAgent, compare vs random policy.

Run with:
    python examples/demo.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from neorl_gym import HVACControlEnv, ChemicalReactorEnv, PIDTuningEnv
from neorl_gym import OfflineDataset, BCAgent


N_TRANSITIONS = 10_000
BC_EPOCHS     = 30
EVAL_EPISODES = 10


def evaluate_random(env, n_episodes=10, max_steps=200):
    returns = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ret = 0.0
        for _ in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            ret += reward
            if terminated or truncated:
                break
        returns.append(ret)
    return {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns))}


def run_env(name: str, env_cls, policy: str = "expert"):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    env = env_cls()
    print(f"  obs_shape = {env.observation_space.shape}")
    print(f"  act_shape = {env.action_space.shape}")

    # 1. Generate offline dataset
    print(f"\n[1] Generating {N_TRANSITIONS} transitions with '{policy}' policy...")
    raw_data = env.generate_dataset(n_transitions=N_TRANSITIONS, policy=policy)

    # 2. Build OfflineDataset with quality filtering
    dataset = OfflineDataset.from_dict(
        raw_data,
        min_reward=np.percentile(raw_data["rewards"], 10),  # drop bottom 10%
        safe_only=True,
    )
    stats = dataset.stats()
    print(f"  reward mean={stats['reward_mean']:.2f}  std={stats['reward_std']:.2f}")

    # 3. Train BC agent
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = BCAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_low=env.action_space.low,
        act_high=env.action_space.high,
    )
    print(f"\n[2] Training BCAgent for {BC_EPOCHS} epochs...")
    agent.train(dataset, epochs=BC_EPOCHS, batch_size=256, verbose=True)

    # 4. Evaluate
    print(f"\n[3] Evaluation ({EVAL_EPISODES} episodes)...")
    bc_metrics  = agent.evaluate(env, n_episodes=EVAL_EPISODES)
    rnd_metrics = evaluate_random(env, n_episodes=EVAL_EPISODES)

    print(f"\n  BCAgent    : return = {bc_metrics['mean_return']:+.1f} ± {bc_metrics['std_return']:.1f}")
    print(f"  Random     : return = {rnd_metrics['mean_return']:+.1f} ± {rnd_metrics['std_return']:.1f}")

    improvement = bc_metrics["mean_return"] - rnd_metrics["mean_return"]
    print(f"  Improvement: {improvement:+.1f}")


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    run_env("HVAC Control (temperature regulation)",   HVACControlEnv,     policy="expert")
    run_env("Chemical Reactor (yield optimisation)",   ChemicalReactorEnv, policy="expert")
    run_env("PID Tuning (settling time optimisation)", PIDTuningEnv,       policy="expert")

    print(f"\n{'='*60}")
    print("  Demo complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
