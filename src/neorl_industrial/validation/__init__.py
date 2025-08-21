"""Validation module for neoRL-industrial-gym."""

import numpy as np
import jax.numpy as jnp
from typing import Union
from ..core.types import Array

from .input_validator import InputValidator, ValidationResult
from .comprehensive_validator import ComprehensiveValidator


def validate_array_input(array: Array, name: str = "array") -> Array:
    """Validate array input with basic checks."""
    if array is None:
        raise ValueError(f"{name} cannot be None")
        
    if not isinstance(array, (np.ndarray, jnp.ndarray)):
        try:
            array = np.array(array)
        except Exception as e:
            raise ValueError(f"Cannot convert {name} to array: {e}")
            
    if array.size == 0:
        raise ValueError(f"{name} cannot be empty")
        
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
        
    return array


__all__ = [
    "InputValidator",
    "ValidationResult", 
    "ComprehensiveValidator",
    "validate_array_input",
]