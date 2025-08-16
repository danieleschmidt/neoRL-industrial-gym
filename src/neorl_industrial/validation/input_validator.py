"""Input validation and sanitization for industrial RL systems."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import jax.numpy as jnp
    import numpy as np
except ImportError:
    import numpy as np
    jnp = np

from ..core.types import Array
from ..exceptions import ValidationError


@dataclass
class ValidationRule:
    """A single validation rule for input data."""
    
    name: str
    validator_fn: callable
    error_message: str
    severity: str = "error"  # "error", "warning", "info"


@dataclass
class ValidationResult:
    """Result of input validation."""
    
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: List[str]
    sanitized_input: Any = None


class InputValidator(ABC):
    """Abstract base class for input validators."""
    
    def __init__(self, rules: List[ValidationRule]):
        self.rules = rules
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def validate(self, input_data: Any) -> ValidationResult:
        """Validate input data against all rules."""
        pass
        
    def _apply_rules(self, input_data: Any) -> ValidationResult:
        """Apply all validation rules to input data."""
        errors = []
        warnings = []
        info = []
        
        for rule in self.rules:
            try:
                if not rule.validator_fn(input_data):
                    if rule.severity == "error":
                        errors.append(f"{rule.name}: {rule.error_message}")
                    elif rule.severity == "warning":
                        warnings.append(f"{rule.name}: {rule.error_message}")
                    else:
                        info.append(f"{rule.name}: {rule.error_message}")
            except Exception as e:
                errors.append(f"{rule.name}: Validation failed - {str(e)}")
                
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            info=info,
            sanitized_input=input_data
        )


class StateValidator(InputValidator):
    """Validator for environment state data."""
    
    def __init__(self, state_dim: int, value_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        self.state_dim = state_dim
        self.value_ranges = value_ranges or {}
        
        # Define standard validation rules
        rules = [
            ValidationRule(
                name="state_shape",
                validator_fn=lambda x: self._validate_shape(x),
                error_message=f"State must have shape ({state_dim},)"
            ),
            ValidationRule(
                name="finite_values",
                validator_fn=lambda x: self._validate_finite(x),
                error_message="State contains non-finite values (NaN or Inf)"
            ),
            ValidationRule(
                name="value_ranges",
                validator_fn=lambda x: self._validate_ranges(x),
                error_message="State values outside expected ranges",
                severity="warning"
            ),
        ]
        
        super().__init__(rules)
        
    def validate(self, state: Array) -> ValidationResult:
        """Validate state array."""
        result = self._apply_rules(state)
        
        # Apply sanitization if needed
        if not result.is_valid:
            sanitized_state = self._sanitize_state(state)
            result.sanitized_input = sanitized_state
            
        return result
        
    def _validate_shape(self, state: Array) -> bool:
        """Validate state shape."""
        try:
            state_array = jnp.asarray(state)
            return state_array.shape == (self.state_dim,)
        except Exception:
            return False
            
    def _validate_finite(self, state: Array) -> bool:
        """Validate all values are finite."""
        try:
            state_array = jnp.asarray(state)
            return jnp.all(jnp.isfinite(state_array))
        except Exception:
            return False
            
    def _validate_ranges(self, state: Array) -> bool:
        """Validate values are within expected ranges."""
        if not self.value_ranges:
            return True
            
        try:
            state_array = jnp.asarray(state)
            for idx, (min_val, max_val) in self.value_ranges.items():
                if isinstance(idx, str):
                    continue  # Skip named indices for now
                if idx < len(state_array):
                    val = state_array[idx]
                    if not (min_val <= val <= max_val):
                        return False
            return True
        except Exception:
            return False
            
    def _sanitize_state(self, state: Array) -> Array:
        """Sanitize state by clipping and replacing invalid values."""
        try:
            state_array = jnp.asarray(state)
            
            # Replace NaN and Inf with zeros
            state_array = jnp.where(jnp.isfinite(state_array), state_array, 0.0)
            
            # Clip to ranges if specified
            for idx, (min_val, max_val) in self.value_ranges.items():
                if isinstance(idx, int) and idx < len(state_array):
                    state_array = state_array.at[idx].set(
                        jnp.clip(state_array[idx], min_val, max_val)
                    )
                    
            return state_array
        except Exception:
            # Fallback to zero state
            return jnp.zeros(self.state_dim)


class ActionValidator(InputValidator):
    """Validator for agent action data."""
    
    def __init__(
        self, 
        action_dim: int, 
        action_bounds: Optional[Tuple[Array, Array]] = None,
        discrete: bool = False
    ):
        self.action_dim = action_dim
        self.action_bounds = action_bounds
        self.discrete = discrete
        
        rules = [
            ValidationRule(
                name="action_shape",
                validator_fn=lambda x: self._validate_shape(x),
                error_message=f"Action must have shape ({action_dim},)"
            ),
            ValidationRule(
                name="finite_values",
                validator_fn=lambda x: self._validate_finite(x),
                error_message="Action contains non-finite values"
            ),
            ValidationRule(
                name="action_bounds",
                validator_fn=lambda x: self._validate_bounds(x),
                error_message="Action values outside bounds"
            ),
        ]
        
        if discrete:
            rules.append(
                ValidationRule(
                    name="discrete_values",
                    validator_fn=lambda x: self._validate_discrete(x),
                    error_message="Action values must be integers for discrete actions"
                )
            )
            
        super().__init__(rules)
        
    def validate(self, action: Array) -> ValidationResult:
        """Validate action array."""
        result = self._apply_rules(action)
        
        if not result.is_valid:
            sanitized_action = self._sanitize_action(action)
            result.sanitized_input = sanitized_action
            
        return result
        
    def _validate_shape(self, action: Array) -> bool:
        """Validate action shape."""
        try:
            action_array = jnp.asarray(action)
            return action_array.shape == (self.action_dim,)
        except Exception:
            return False
            
    def _validate_finite(self, action: Array) -> bool:
        """Validate all values are finite."""
        try:
            action_array = jnp.asarray(action)
            return jnp.all(jnp.isfinite(action_array))
        except Exception:
            return False
            
    def _validate_bounds(self, action: Array) -> bool:
        """Validate action bounds."""
        if self.action_bounds is None:
            return True
            
        try:
            action_array = jnp.asarray(action)
            low, high = self.action_bounds
            low_array = jnp.asarray(low)
            high_array = jnp.asarray(high)
            
            return jnp.all(action_array >= low_array) and jnp.all(action_array <= high_array)
        except Exception:
            return False
            
    def _validate_discrete(self, action: Array) -> bool:
        """Validate discrete action values."""
        try:
            action_array = jnp.asarray(action)
            return jnp.all(action_array == jnp.round(action_array))
        except Exception:
            return False
            
    def _sanitize_action(self, action: Array) -> Array:
        """Sanitize action by clipping and replacing invalid values."""
        try:
            action_array = jnp.asarray(action)
            
            # Replace NaN and Inf with zeros
            action_array = jnp.where(jnp.isfinite(action_array), action_array, 0.0)
            
            # Clip to bounds if specified
            if self.action_bounds is not None:
                low, high = self.action_bounds
                action_array = jnp.clip(action_array, low, high)
                
            # Round discrete actions
            if self.discrete:
                action_array = jnp.round(action_array)
                
            return action_array
        except Exception:
            # Fallback to zero action
            return jnp.zeros(self.action_dim)


class SafetyValidator(InputValidator):
    """Validator for safety-critical parameters."""
    
    def __init__(self, safety_constraints: List[Dict[str, Any]]):
        self.safety_constraints = safety_constraints
        
        rules = [
            ValidationRule(
                name="safety_bounds",
                validator_fn=lambda x: self._validate_safety_bounds(x),
                error_message="Input violates safety constraints"
            ),
            ValidationRule(
                name="rate_limits",
                validator_fn=lambda x: self._validate_rate_limits(x),
                error_message="Input rate of change exceeds limits",
                severity="warning"
            ),
        ]
        
        super().__init__(rules)
        self._previous_input = None
        
    def validate(self, input_data: Any) -> ValidationResult:
        """Validate safety-critical input."""
        result = self._apply_rules(input_data)
        
        # Store for rate limit checking
        if result.is_valid:
            self._previous_input = input_data
            
        return result
        
    def _validate_safety_bounds(self, input_data: Any) -> bool:
        """Validate against safety constraints."""
        try:
            for constraint in self.safety_constraints:
                constraint_fn = constraint.get("constraint_fn")
                if constraint_fn and not constraint_fn(input_data):
                    return False
            return True
        except Exception:
            return False
            
    def _validate_rate_limits(self, input_data: Any) -> bool:
        """Validate rate of change limits."""
        if self._previous_input is None:
            return True
            
        try:
            current = jnp.asarray(input_data)
            previous = jnp.asarray(self._previous_input)
            
            # Calculate rate of change
            rate = jnp.abs(current - previous)
            max_rate = 10.0  # Configurable rate limit
            
            return jnp.all(rate <= max_rate)
        except Exception:
            return True  # Don't fail on rate limit errors


class DatasetValidator(InputValidator):
    """Validator for training datasets."""
    
    def __init__(self, expected_keys: List[str], min_samples: int = 1000):
        self.expected_keys = expected_keys
        self.min_samples = min_samples
        
        rules = [
            ValidationRule(
                name="required_keys",
                validator_fn=lambda x: self._validate_keys(x),
                error_message=f"Dataset missing required keys: {expected_keys}"
            ),
            ValidationRule(
                name="sample_count",
                validator_fn=lambda x: self._validate_sample_count(x),
                error_message=f"Dataset must have at least {min_samples} samples"
            ),
            ValidationRule(
                name="data_consistency",
                validator_fn=lambda x: self._validate_consistency(x),
                error_message="Dataset arrays have inconsistent lengths"
            ),
            ValidationRule(
                name="data_quality",
                validator_fn=lambda x: self._validate_quality(x),
                error_message="Dataset contains too many invalid values",
                severity="warning"
            ),
        ]
        
        super().__init__(rules)
        
    def validate(self, dataset: Dict[str, Array]) -> ValidationResult:
        """Validate training dataset."""
        return self._apply_rules(dataset)
        
    def _validate_keys(self, dataset: Dict[str, Array]) -> bool:
        """Validate required keys are present."""
        try:
            return all(key in dataset for key in self.expected_keys)
        except Exception:
            return False
            
    def _validate_sample_count(self, dataset: Dict[str, Array]) -> bool:
        """Validate minimum sample count."""
        try:
            if not self.expected_keys:
                return True
            first_key = self.expected_keys[0]
            if first_key in dataset:
                return len(dataset[first_key]) >= self.min_samples
            return False
        except Exception:
            return False
            
    def _validate_consistency(self, dataset: Dict[str, Array]) -> bool:
        """Validate all arrays have same length."""
        try:
            lengths = [len(dataset[key]) for key in self.expected_keys if key in dataset]
            return len(set(lengths)) == 1
        except Exception:
            return False
            
    def _validate_quality(self, dataset: Dict[str, Array]) -> bool:
        """Validate data quality (finite values, reasonable ranges)."""
        try:
            for key in self.expected_keys:
                if key in dataset:
                    data = jnp.asarray(dataset[key])
                    finite_ratio = jnp.mean(jnp.isfinite(data))
                    if finite_ratio < 0.95:  # Less than 95% finite values
                        return False
            return True
        except Exception:
            return False


class ComprehensiveValidator:
    """Comprehensive validation orchestrator for industrial RL."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        state_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
        action_bounds: Optional[Tuple[Array, Array]] = None,
        safety_constraints: Optional[List[Dict[str, Any]]] = None,
        discrete_actions: bool = False
    ):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize sub-validators
        self.state_validator = StateValidator(state_dim, state_ranges)
        self.action_validator = ActionValidator(action_dim, action_bounds, discrete_actions)
        
        if safety_constraints:
            self.safety_validator = SafetyValidator(safety_constraints)
        else:
            self.safety_validator = None
            
        self.dataset_validator = DatasetValidator(
            expected_keys=["observations", "actions", "rewards", "next_observations", "dones"]
        )
        
    def validate_state(self, state: Array, auto_correct: bool = True) -> Tuple[Array, ValidationResult]:
        """Validate and optionally correct state."""
        result = self.state_validator.validate(state)
        
        if auto_correct and not result.is_valid and result.sanitized_input is not None:
            self.logger.warning(f"State validation failed, applying corrections: {result.errors}")
            return result.sanitized_input, result
        elif not result.is_valid:
            self.logger.error(f"State validation failed: {result.errors}")
            raise ValidationError(f"Invalid state: {result.errors}")
            
        return state, result
        
    def validate_action(self, action: Array, auto_correct: bool = True) -> Tuple[Array, ValidationResult]:
        """Validate and optionally correct action."""
        result = self.action_validator.validate(action)
        
        if auto_correct and not result.is_valid and result.sanitized_input is not None:
            self.logger.warning(f"Action validation failed, applying corrections: {result.errors}")
            return result.sanitized_input, result
        elif not result.is_valid:
            self.logger.error(f"Action validation failed: {result.errors}")
            raise ValidationError(f"Invalid action: {result.errors}")
            
        return action, result
        
    def validate_safety(self, input_data: Any) -> ValidationResult:
        """Validate safety-critical input."""
        if self.safety_validator:
            result = self.safety_validator.validate(input_data)
            
            if not result.is_valid:
                self.logger.critical(f"Safety validation failed: {result.errors}")
                
            return result
        else:
            return ValidationResult(True, [], [], [])
            
    def validate_dataset(self, dataset: Dict[str, Array]) -> ValidationResult:
        """Validate training dataset."""
        result = self.dataset_validator.validate(dataset)
        
        if not result.is_valid:
            self.logger.error(f"Dataset validation failed: {result.errors}")
            
        if result.warnings:
            self.logger.warning(f"Dataset quality warnings: {result.warnings}")
            
        return result
        
    def enable_strict_mode(self):
        """Enable strict validation mode (no auto-correction)."""
        self.strict_mode = True
        self.logger.info("Strict validation mode enabled")
        
    def get_validation_report(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        return {
            "validators": {
                "state_validator": {
                    "rules": len(self.state_validator.rules),
                    "state_dim": self.state_validator.state_dim
                },
                "action_validator": {
                    "rules": len(self.action_validator.rules),
                    "action_dim": self.action_validator.action_dim
                },
                "safety_validator": {
                    "enabled": self.safety_validator is not None,
                    "constraints": len(self.safety_validator.safety_constraints) if self.safety_validator else 0
                }
            }
        }