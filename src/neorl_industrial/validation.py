"""Robust validation utilities for industrial RL."""

import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import logging

from .core.types import Array, StateArray, ActionArray
from .exceptions import ValidationError


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    error_message: str = ""
    warnings: List[str] = None
    corrected_value: Any = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class DataValidator:
    """Comprehensive data validation for industrial RL systems."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_array(
        self,
        arr: Array,
        name: str,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dtype: Optional[type] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_nan: bool = False,
        allow_inf: bool = False,
        non_empty: bool = True,
    ) -> ValidationResult:
        """Validate array properties.
        
        Args:
            arr: Array to validate
            name: Name for error messages
            expected_shape: Expected array shape (None to skip)
            expected_dtype: Expected data type
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_nan: Whether NaN values are allowed
            allow_inf: Whether infinite values are allowed
            non_empty: Whether array must be non-empty
            
        Returns:
            ValidationResult with validation status
        """
        warnings = []
        
        try:
            # Convert to numpy array if needed
            if not isinstance(arr, np.ndarray):
                try:
                    arr = np.array(arr)
                    warnings.append(f"{name}: Converted input to numpy array")
                except Exception as e:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"{name}: Cannot convert to array - {e}"
                    )
            
            # Check if empty
            if non_empty and arr.size == 0:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"{name}: Array is empty"
                )
            
            # Check shape
            if expected_shape is not None:
                if isinstance(expected_shape, int):
                    expected_shape = (expected_shape,)
                
                # Allow flexible batch dimension for multi-dimensional arrays
                if len(expected_shape) > 1 and len(arr.shape) == len(expected_shape):
                    if arr.shape[1:] != expected_shape[1:]:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"{name}: Shape mismatch - got {arr.shape}, expected (..., {expected_shape[1:]})"
                        )
                elif len(expected_shape) == 1 and len(arr.shape) <= 2:
                    # Allow both (N,) and (batch, N) shapes
                    if len(arr.shape) == 1 and arr.shape[0] != expected_shape[0]:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"{name}: Shape mismatch - got {arr.shape}, expected {expected_shape}"
                        )
                    elif len(arr.shape) == 2 and arr.shape[1] != expected_shape[0]:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"{name}: Shape mismatch - got {arr.shape}, expected (batch, {expected_shape[0]})"
                        )
                else:
                    if arr.shape != expected_shape:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"{name}: Shape mismatch - got {arr.shape}, expected {expected_shape}"
                        )
            
            # Check dtype
            if expected_dtype is not None:
                if arr.dtype != expected_dtype:
                    # Try to cast
                    try:
                        arr = arr.astype(expected_dtype)
                        warnings.append(f"{name}: Cast from {arr.dtype} to {expected_dtype}")
                    except Exception:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"{name}: Cannot cast from {arr.dtype} to {expected_dtype}"
                        )
            
            # Check for NaN values
            if not allow_nan and np.isnan(arr).any():
                nan_count = np.isnan(arr).sum()
                return ValidationResult(
                    is_valid=False,
                    error_message=f"{name}: Contains {nan_count} NaN values"
                )
            
            # Check for infinite values
            if not allow_inf and np.isinf(arr).any():
                inf_count = np.isinf(arr).sum()
                return ValidationResult(
                    is_valid=False,
                    error_message=f"{name}: Contains {inf_count} infinite values"
                )
            
            # Check value ranges
            if min_value is not None:
                if arr.min() < min_value:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"{name}: Minimum value {arr.min()} below threshold {min_value}"
                    )
            
            if max_value is not None:
                if arr.max() > max_value:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"{name}: Maximum value {arr.max()} above threshold {max_value}"
                    )
            
            return ValidationResult(
                is_valid=True,
                warnings=warnings,
                corrected_value=arr
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"{name}: Validation failed - {e}"
            )
    
    def validate_dataset(
        self,
        dataset: Dict[str, Array],
        state_dim: int,
        action_dim: int,
        min_samples: int = 1000,
    ) -> ValidationResult:
        """Validate offline RL dataset.
        
        Args:
            dataset: Dataset dictionary
            state_dim: Expected state dimension
            action_dim: Expected action dimension
            min_samples: Minimum number of samples
            
        Returns:
            ValidationResult with validation status
        """
        warnings = []
        
        try:
            # Check required keys
            required_keys = ["observations", "actions", "rewards"]
            missing_keys = [key for key in required_keys if key not in dataset]
            if missing_keys:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Dataset missing required keys: {missing_keys}"
                )
            
            # Check optional keys
            optional_keys = ["next_observations", "terminals", "timeouts"]
            for key in optional_keys:
                if key not in dataset:
                    warnings.append(f"Dataset missing optional key: {key}")
            
            # Validate each component
            n_samples = len(dataset["observations"])
            
            # Check minimum samples
            if n_samples < min_samples:
                warnings.append(f"Dataset has only {n_samples} samples (recommended: {min_samples}+)")
            
            # Validate observations
            obs_result = self.validate_array(
                dataset["observations"],
                "observations",
                expected_shape=(n_samples, state_dim),
                expected_dtype=np.float32,
                allow_nan=False,
                allow_inf=False
            )
            if not obs_result.is_valid:
                return obs_result
            warnings.extend(obs_result.warnings)
            
            # Validate actions
            act_result = self.validate_array(
                dataset["actions"],
                "actions",
                expected_shape=(n_samples, action_dim),
                expected_dtype=np.float32,
                allow_nan=False,
                allow_inf=False
            )
            if not act_result.is_valid:
                return act_result
            warnings.extend(act_result.warnings)
            
            # Validate rewards
            rew_result = self.validate_array(
                dataset["rewards"],
                "rewards",
                expected_shape=(n_samples,),
                expected_dtype=np.float32,
                allow_nan=False,
                allow_inf=False
            )
            if not rew_result.is_valid:
                return rew_result
            warnings.extend(rew_result.warnings)
            
            # Check reward distribution
            rewards = dataset["rewards"]
            if np.std(rewards) < 1e-6:
                warnings.append("Rewards have very low variance - check reward engineering")
            
            reward_range = rewards.max() - rewards.min()
            if reward_range > 10000:
                warnings.append(f"Large reward range ({reward_range:.2f}) - consider reward scaling")
            
            # Validate terminals if present
            if "terminals" in dataset:
                term_result = self.validate_array(
                    dataset["terminals"],
                    "terminals",
                    expected_shape=(n_samples,),
                    expected_dtype=bool,
                    allow_nan=False,
                    allow_inf=False
                )
                if not term_result.is_valid:
                    return term_result
                warnings.extend(term_result.warnings)
            
            # Check data quality indicators
            action_range = dataset["actions"].max() - dataset["actions"].min()
            if action_range < 0.01:
                warnings.append("Actions have very low variance - check policy diversity")
            
            # Check for potential data leaks
            if "next_observations" in dataset:
                next_obs = dataset["next_observations"]
                if np.allclose(dataset["observations"][:-1], next_obs[:-1], atol=1e-6):
                    warnings.append("Next observations suspiciously similar to current - check data collection")
            
            return ValidationResult(
                is_valid=True,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Dataset validation failed: {e}"
            )
    
    def validate_hyperparameters(
        self,
        hyperparams: Dict[str, Any],
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> ValidationResult:
        """Validate hyperparameters for safety and reasonableness.
        
        Args:
            hyperparams: Dictionary of hyperparameters
            param_ranges: Optional ranges for each parameter
            
        Returns:
            ValidationResult with validation status
        """
        warnings = []
        
        # Default safe ranges
        default_ranges = {
            "learning_rate": (1e-6, 1e-1),
            "batch_size": (8, 8192),
            "gamma": (0.8, 0.9999),
            "tau": (1e-4, 1e-1),
            "n_epochs": (1, 10000),
            "hidden_dims": (8, 2048),  # For individual layer sizes
            "dropout_rate": (0.0, 0.8),
            "weight_decay": (0.0, 1e-1),
        }
        
        param_ranges = param_ranges or {}
        
        try:
            for param, value in hyperparams.items():
                # Skip non-numeric parameters
                if not isinstance(value, (int, float)):
                    if param in ["activation", "optimizer", "device"]:
                        continue
                    elif isinstance(value, (list, tuple)):
                        # Handle sequences like hidden_dims
                        if param == "hidden_dims":
                            for dim in value:
                                if not isinstance(dim, int) or dim < 1:
                                    return ValidationResult(
                                        is_valid=False,
                                        error_message=f"{param}: All dimensions must be positive integers"
                                    )
                                if param in default_ranges:
                                    min_val, max_val = default_ranges[param]
                                    if dim < min_val or dim > max_val:
                                        warnings.append(
                                            f"{param}: Dimension {dim} outside recommended range [{min_val}, {max_val}]"
                                        )
                        continue
                    else:
                        continue
                
                # Check parameter ranges
                ranges = param_ranges.get(param, default_ranges.get(param))
                if ranges is not None:
                    min_val, max_val = ranges
                    if value < min_val:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"{param}: Value {value} below minimum {min_val}"
                        )
                    elif value > max_val:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"{param}: Value {value} above maximum {max_val}"
                        )
                    
                    # Warn about suspicious values
                    if param == "learning_rate" and value > 1e-2:
                        warnings.append(f"{param}: High learning rate {value} may cause instability")
                    elif param == "batch_size" and value < 32:
                        warnings.append(f"{param}: Small batch size {value} may increase variance")
                    elif param == "gamma" and value > 0.995:
                        warnings.append(f"{param}: Very high discount factor {value} may cause slow learning")
            
            # Check parameter combinations
            if "learning_rate" in hyperparams and "batch_size" in hyperparams:
                lr = hyperparams["learning_rate"]
                bs = hyperparams["batch_size"]
                if lr > 1e-3 and bs < 64:
                    warnings.append("High learning rate with small batch size may cause instability")
            
            return ValidationResult(
                is_valid=True,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Hyperparameter validation failed: {e}"
            )
    
    def validate_environment(
        self,
        env: Any,
        expected_state_dim: int,
        expected_action_dim: int,
        n_test_steps: int = 100,
    ) -> ValidationResult:
        """Validate environment compatibility and safety.
        
        Args:
            env: Environment to validate
            expected_state_dim: Expected observation dimension
            expected_action_dim: Expected action dimension
            n_test_steps: Number of test steps to run
            
        Returns:
            ValidationResult with validation status
        """
        warnings = []
        
        try:
            # Check basic interface
            required_methods = ["reset", "step"]
            for method in required_methods:
                if not hasattr(env, method) or not callable(getattr(env, method)):
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Environment missing required method: {method}"
                    )
            
            # Test reset
            try:
                obs, info = env.reset()
                if obs is None:
                    return ValidationResult(
                        is_valid=False,
                        error_message="Environment reset() returned None observation"
                    )
                
                # Validate observation shape
                obs_result = self.validate_array(
                    obs,
                    "reset_observation",
                    expected_shape=(expected_state_dim,),
                    expected_dtype=np.float32,
                    allow_nan=False,
                    allow_inf=False
                )
                if not obs_result.is_valid:
                    return obs_result
                warnings.extend(obs_result.warnings)
                
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Environment reset() failed: {e}"
                )
            
            # Test step function with random actions
            violation_count = 0
            reward_sum = 0.0
            
            for i in range(min(n_test_steps, 100)):  # Limit test steps
                try:
                    # Generate random action
                    action = np.random.uniform(-1, 1, expected_action_dim).astype(np.float32)
                    
                    # Take step
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    
                    # Validate outputs
                    if next_obs is None:
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"Environment step() returned None observation at step {i}"
                        )
                    
                    # Check observation
                    obs_result = self.validate_array(
                        next_obs,
                        f"step_{i}_observation",
                        expected_shape=(expected_state_dim,),
                        expected_dtype=np.float32,
                        allow_nan=False,
                        allow_inf=False
                    )
                    if not obs_result.is_valid:
                        return obs_result
                    
                    # Check reward
                    if not isinstance(reward, (int, float)) or np.isnan(reward) or np.isinf(reward):
                        return ValidationResult(
                            is_valid=False,
                            error_message=f"Invalid reward at step {i}: {reward}"
                        )
                    
                    reward_sum += reward
                    
                    # Check safety violations
                    if info and "safety_metrics" in info:
                        violation_count += info["safety_metrics"].violation_count
                    
                    # Update observation for next step
                    obs = next_obs
                    
                    if terminated or truncated:
                        break
                        
                except Exception as e:
                    return ValidationResult(
                        is_valid=False,
                        error_message=f"Environment step() failed at step {i}: {e}"
                    )
            
            # Check test results
            if violation_count > n_test_steps * 0.5:  # More than 50% violations
                warnings.append(f"High safety violation rate: {violation_count}/{n_test_steps} steps")
            
            avg_reward = reward_sum / min(i + 1, n_test_steps)
            if abs(avg_reward) > 1000:
                warnings.append(f"Large average reward magnitude: {avg_reward:.2f}")
            
            return ValidationResult(
                is_valid=True,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                error_message=f"Environment validation failed: {e}"
            )


# Global validator instance
_validator = DataValidator()


def validate_array(*args, **kwargs) -> ValidationResult:
    """Global array validation function."""
    return _validator.validate_array(*args, **kwargs)


def validate_dataset(*args, **kwargs) -> ValidationResult:
    """Global dataset validation function."""
    return _validator.validate_dataset(*args, **kwargs)


def validate_hyperparameters(*args, **kwargs) -> ValidationResult:
    """Global hyperparameter validation function."""
    return _validator.validate_hyperparameters(*args, **kwargs)


def validate_environment(*args, **kwargs) -> ValidationResult:
    """Global environment validation function."""
    return _validator.validate_environment(*args, **kwargs)
