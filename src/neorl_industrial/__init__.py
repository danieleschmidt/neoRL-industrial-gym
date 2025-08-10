"""neoRL-industrial-gym: Industrial-grade Offline RL benchmark & library."""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .core.types import DatasetQuality, SafetyConstraint, SafetyMetrics
from .environments import IndustrialEnv, ChemicalReactorEnv, PowerGridEnv, RobotAssemblyEnv
from .agents import OfflineAgent, CQLAgent, IQLAgent, TD3BCAgent
from .utils import make, evaluate_with_safety

__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.ai"

__all__ = [
    "__version__",
    # Core types
    "DatasetQuality",
    "SafetyConstraint", 
    "SafetyMetrics",
    # Environments
    "IndustrialEnv",
    "ChemicalReactorEnv",
    "PowerGridEnv",
    "RobotAssemblyEnv",
    # Agents
    "OfflineAgent",
    "CQLAgent",
    "IQLAgent",
    "TD3BCAgent",
    # Utils
    "make",
    "evaluate_with_safety",
]