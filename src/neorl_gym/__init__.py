"""neoRL Industrial Gym — offline RL environments for industrial control.

Agents learn from pre-collected datasets without online interaction,
suitable for safety-critical industrial settings where trial-and-error
is not feasible.
"""

from neorl_gym.envs.base import IndustrialEnv
from neorl_gym.envs.hvac import HVACControlEnv
from neorl_gym.envs.reactor import ChemicalReactorEnv
from neorl_gym.envs.pid import PIDTuningEnv
from neorl_gym.offline.dataset import OfflineDataset
from neorl_gym.agents.bc import BCAgent

__version__ = "0.2.0"

__all__ = [
    "IndustrialEnv",
    "HVACControlEnv",
    "ChemicalReactorEnv",
    "PIDTuningEnv",
    "OfflineDataset",
    "BCAgent",
]
