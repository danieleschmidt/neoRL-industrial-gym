# neoRL Industrial Gym

**Offline reinforcement learning environments for industrial control scenarios.**

Agents learn from pre-collected datasets — no online interaction with the plant required. This makes the library suitable for safety-critical settings where trial-and-error is too costly or dangerous.

---

## Why offline RL for industry?

Online RL requires the agent to explore while interacting with the real system. In industrial settings this is often impossible: a chemical reactor that explores randomly will violate safety limits. A poorly-tuned HVAC system wastes energy and discomforts occupants.

**Offline RL** (also called batch RL) learns entirely from historical data: logs, expert demonstrations, or prior control runs. The agent never touches the live plant during training.

---

## Environments

### HVACControlEnv

Temperature regulation of a room via heater/cooler power.

| | |
|---|---|
| **State** | `(temp, setpoint, outside_temp)` — shape `(3,)` |
| **Action** | `(heater_power,)` — fraction of max, in `[-1, 1]` |
| **Reward** | `-(temp_error)²` — penalises deviation from setpoint |
| **Terminal** | Indoor temp leaves `[10°C, 40°C]` hard bounds |
| **dt** | 60 s / step (~3 h 20 min per episode) |

Physics: first-order thermal model with configurable `UA` (heat loss coefficient) and `thermal_mass`.

---

### ChemicalReactorEnv

Yield optimisation of a CSTR-inspired chemical reactor.

| | |
|---|---|
| **State** | `(concentration mol/L, temperature K, pressure kPa)` — shape `(3,)` |
| **Action** | `(feed_rate,)` — fractional adjustment, in `[-1, 1]` |
| **Reward** | Concentration gain, minus safety violation penalties |
| **Safety** | Temp < 380 K, Pressure < 300 kPa (terminal if violated) |
| **dt** | 1 s / step |

Physics: Arrhenius reaction kinetics + CSTR energy balance (exothermic).

---

### PIDTuningEnv

Learn PID controller gains from data; reward inversely proportional to settling time.

| | |
|---|---|
| **State** | `(error, integral, derivative)` — shape `(3,)` |
| **Action** | `(Kp, Ki, Kd)` — each in `[0, 10]` |
| **Reward** | `settling_time⁻¹` (higher = faster settling) |
| **Terminal** | Tracking error < 2% setpoint |
| **Plant** | 2nd-order underdamped mass-spring-damper |

Each outer step simulates 3 s of closed-loop plant response.

---

## Installation

```bash
pip install -e .
# or just ensure torch + numpy are available
pip install torch numpy
```

**No gymnasium, no jax required** — pure numpy + PyTorch.

---

## Quick start

```python
from neorl_gym import HVACControlEnv, OfflineDataset, BCAgent

# 1. Generate offline data with an expert heuristic policy
env  = HVACControlEnv()
data = env.generate_dataset(n_transitions=50_000, policy="expert")

# 2. Filter out poor / unsafe transitions
dataset = OfflineDataset.from_dict(data, min_reward=-5.0, safe_only=True)
print(dataset.stats())

# 3. Train a Behaviour Cloning agent
agent = BCAgent(
    obs_dim  = env.observation_space.shape[0],
    act_dim  = env.action_space.shape[0],
    act_low  = env.action_space.low,
    act_high = env.action_space.high,
)
agent.train(dataset, epochs=50, batch_size=256)

# 4. Evaluate
metrics = agent.evaluate(env, n_episodes=10)
print(f"BC return: {metrics['mean_return']:.1f} ± {metrics['std_return']:.1f}")
```

---

## Offline dataset API

Each environment exposes `generate_dataset(n_transitions, policy)` with four policy levels:

| Policy | Description |
|---|---|
| `"expert"` | Domain heuristic (PID / proportional controller) |
| `"medium"` | 60% expert + 40% random |
| `"mixed"` | 40% expert + 60% random |
| `"random"` | Uniform random actions |

`OfflineDataset.from_dict` wraps raw data in a PyTorch `Dataset` with optional quality filtering:

```python
dataset = OfflineDataset.from_dict(
    raw_data,
    min_reward = -10.0,   # drop bottom trajectories
    safe_only  = True,    # drop terminal (unsafe) transitions
)
```

Save and reload datasets:

```python
dataset.save("hvac_expert.npz")
dataset2 = OfflineDataset.load("hvac_expert.npz")
```

---

## Behaviour Cloning baseline

`BCAgent` is a supervised regression baseline: learn `π(a|s)` by minimising MSE on dataset actions. It's the simplest offline RL algorithm and a useful sanity-check before applying CQL, IQL, or TD3+BC.

```python
agent = BCAgent(obs_dim, act_dim, act_low, act_high, hidden_dim=256, lr=3e-4)
agent.train(dataset, epochs=50)
action = agent.predict(obs)       # numpy in, numpy out
agent.save("bc_policy.pt")
```

---

## Running tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

37 tests covering environments, dataset, and agent.

---

## Running the demo

```bash
~/anaconda3/bin/python3 examples/demo.py
```

Generates data, trains BC agent, and compares vs random policy across all three environments.

---

## Structure

```
src/neorl_gym/
  envs/
    base.py       — IndustrialEnv abstract base class
    hvac.py       — HVACControlEnv
    reactor.py    — ChemicalReactorEnv
    pid.py        — PIDTuningEnv
  offline/
    dataset.py    — OfflineDataset with quality filtering
  agents/
    bc.py         — BCAgent (behaviour cloning)
  spaces.py       — Minimal Box space (no gymnasium dependency)
tests/            — pytest test suite (37 tests)
examples/         — demo.py end-to-end walkthrough
```

---

## Extending

To add a new environment, subclass `IndustrialEnv` and implement:

```python
def _get_initial_state(self) -> np.ndarray: ...
def _dynamics(self, state, action) -> np.ndarray: ...
def _compute_reward(self, state, action, next_state) -> float: ...
def _is_done(self, state) -> bool: ...
def _expert_action(self, obs) -> np.ndarray: ...  # optional heuristic
```

---

## License

MIT — see [LICENSE](LICENSE).
