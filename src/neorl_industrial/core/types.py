"""Core type definitions for industrial RL environments."""

import jax.numpy as jnp
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Type aliases
Array = Union[np.ndarray, jnp.ndarray]
StateArray = Array
ActionArray = Array
ObservationArray = Array


@dataclass
class IndustrialState:
    """Industrial system state with safety tracking."""
    
    observation: StateArray
    safety_metrics: Dict[str, float]
    timestamp: float
    system_status: str = "normal"
    

class DatasetQuality(Enum):
    """Quality levels for offline datasets."""
    
    EXPERT = "expert"
    MEDIUM = "medium"  
    MIXED = "mixed"
    RANDOM = "random"


@dataclass
class SafetyConstraint:
    """Safety constraint definition."""
    
    name: str
    check_fn: Callable[[StateArray, ActionArray], bool]
    penalty: float
    critical: bool = False
    description: str = ""


@dataclass
class SafetyMetrics:
    """Safety monitoring metrics."""
    
    constraints_satisfied: int
    total_constraints: int
    violation_count: int
    critical_violations: int
    safety_score: float
    
    @property
    def satisfaction_rate(self) -> float:
        """Calculate constraint satisfaction rate."""
        if self.total_constraints == 0:
            return 1.0
        return self.constraints_satisfied / self.total_constraints