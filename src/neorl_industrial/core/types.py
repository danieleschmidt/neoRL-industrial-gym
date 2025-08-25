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
MetricsDict = Dict[str, Union[float, int, str]]
HyperparametersDict = Dict[str, Union[float, int, str, bool]]


@dataclass
class IndustrialState:
    """Industrial system state with safety tracking."""
    
    observation: StateArray
    safety_metrics: Dict[str, float]
    timestamp: float
    system_status: str = "normal"
    confidence_score: float = 1.0
    uncertainty_bounds: Tuple[float, float] = (0.0, 0.0)
    anomaly_score: float = 0.0
    
    @property
    def is_safe(self) -> bool:
        """Check if current state is considered safe."""
        return (
            self.system_status in ["normal", "warning"] and
            self.anomaly_score < 0.5 and
            self.confidence_score > 0.7
        )
    
    def update_confidence(self, prediction_variance: float) -> None:
        """Update confidence score based on model uncertainty."""
        self.confidence_score = max(0.0, min(1.0, 1.0 - prediction_variance))
        half_range = prediction_variance * 0.5
        self.uncertainty_bounds = (-half_range, half_range)
    

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
    adaptive_threshold: float = 0.95
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    violation_severity: Dict[str, float] = None
    
    def __post_init__(self):
        if self.violation_severity is None:
            self.violation_severity = {}
    
    @property
    def satisfaction_rate(self) -> float:
        """Calculate constraint satisfaction rate."""
        if self.total_constraints == 0:
            return 1.0
        return self.constraints_satisfied / self.total_constraints
    
    @property
    def adaptive_safety_score(self) -> float:
        """Calculate adaptive safety score with confidence."""
        base_score = self.safety_score
        confidence_penalty = abs(self.confidence_interval[1] - self.confidence_interval[0]) * 0.1
        return max(0.0, base_score - confidence_penalty)
    
    def update_adaptive_threshold(self, performance_history: List[float]) -> None:
        """Update adaptive threshold based on performance history."""
        if len(performance_history) >= 10:
            mean_perf = np.mean(performance_history[-10:])
            std_perf = np.std(performance_history[-10:])
            self.adaptive_threshold = max(0.8, min(0.99, mean_perf - 2 * std_perf))