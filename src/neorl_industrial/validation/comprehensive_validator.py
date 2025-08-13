"""Comprehensive validation framework for industrial RL systems."""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
import threading
from collections import defaultdict
import logging

from ..core.types import Array, StateArray, ActionArray, SafetyConstraint

@dataclass 
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    confidence: float  # 0.0 to 1.0
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    validation_time: Optional[float] = None
    recommendations: Optional[List[str]] = None

class BaseValidator(ABC):
    """Base class for validation components."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.validation_count = 0
        self.success_count = 0
        self.last_validation_time = None
        
    @abstractmethod
    def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Perform validation."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        success_rate = self.success_count / self.validation_count if self.validation_count > 0 else 0.0
        return {
            "name": self.name,
            "enabled": self.enabled,
            "validation_count": self.validation_count,
            "success_count": self.success_count,
            "success_rate": success_rate,
            "last_validation_time": self.last_validation_time,
        }

class DataShapeValidator(BaseValidator):
    """Validates data shapes and dimensions."""
    
    def __init__(self, expected_shapes: Dict[str, Tuple[int, ...]], allow_batch_dim: bool = True):
        super().__init__("data_shape_validator")
        self.expected_shapes = expected_shapes
        self.allow_batch_dim = allow_batch_dim
    
    def validate(self, data: Dict[str, Array], context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data shapes."""
        start_time = time.time()
        self.validation_count += 1
        
        try:
            for key, expected_shape in self.expected_shapes.items():
                if key not in data:
                    return ValidationResult(
                        is_valid=False,
                        confidence=1.0,
                        error_message=f"Missing required key: {key}",
                        validation_time=time.time() - start_time
                    )
                
                actual_shape = data[key].shape
                
                # Handle batch dimension
                if self.allow_batch_dim and len(actual_shape) == len(expected_shape) + 1:
                    actual_shape = actual_shape[1:]  # Remove batch dimension
                
                if actual_shape != expected_shape:
                    return ValidationResult(
                        is_valid=False,
                        confidence=1.0,
                        error_message=f"Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}",
                        details={"expected": expected_shape, "actual": data[key].shape},
                        validation_time=time.time() - start_time
                    )
            
            self.success_count += 1
            self.last_validation_time = time.time()
            
            return ValidationResult(
                is_valid=True,
                confidence=1.0,
                validation_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                error_message=f"Validation error: {str(e)}",
                validation_time=time.time() - start_time
            )

class DataRangeValidator(BaseValidator):
    """Validates data ranges and bounds."""
    
    def __init__(self, value_ranges: Dict[str, Tuple[float, float]], strict: bool = True):
        super().__init__("data_range_validator")
        self.value_ranges = value_ranges
        self.strict = strict  # If False, allows some values outside range
    
    def validate(self, data: Dict[str, Array], context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate data ranges."""
        start_time = time.time()
        self.validation_count += 1
        
        violations = []
        
        try:
            for key, (min_val, max_val) in self.value_ranges.items():
                if key not in data:
                    continue
                
                values = data[key]
                
                # Check for NaN or infinite values
                if np.any(np.isnan(values)):
                    violations.append(f"{key} contains NaN values")
                
                if np.any(np.isinf(values)):
                    violations.append(f"{key} contains infinite values")
                
                # Check range violations
                below_min = np.sum(values < min_val)
                above_max = np.sum(values > max_val)
                
                if below_min > 0:
                    pct = (below_min / values.size) * 100
                    violations.append(f"{key}: {below_min} values ({pct:.1f}%) below minimum {min_val}")
                
                if above_max > 0:
                    pct = (above_max / values.size) * 100
                    violations.append(f"{key}: {above_max} values ({pct:.1f}%) above maximum {max_val}")
            
            # Determine validity
            if not violations:
                is_valid = True
                confidence = 1.0
                error_message = None
            elif self.strict:
                is_valid = False
                confidence = 0.0
                error_message = "; ".join(violations)
            else:
                # Allow some violations in non-strict mode
                total_violations = len(violations)
                is_valid = total_violations <= 2  # Allow up to 2 violations
                confidence = max(0.1, 1.0 - (total_violations * 0.3))
                error_message = "; ".join(violations) if not is_valid else None
            
            if is_valid:
                self.success_count += 1
            
            self.last_validation_time = time.time()
            
            return ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                error_message=error_message,
                details={"violations": violations} if violations else None,
                validation_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                error_message=f"Range validation error: {str(e)}",
                validation_time=time.time() - start_time
            )

class SafetyConstraintValidator(BaseValidator):
    """Validates safety constraints for industrial environments."""
    
    def __init__(self, safety_constraints: List[SafetyConstraint], tolerance: float = 0.1):
        super().__init__("safety_constraint_validator")
        self.safety_constraints = safety_constraints
        self.tolerance = tolerance  # Allowed violation rate
    
    def validate(self, data: Dict[str, Array], context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate safety constraints."""
        start_time = time.time()
        self.validation_count += 1
        
        try:
            states = data.get("observations") or data.get("states")
            actions = data.get("actions")
            
            if states is None or actions is None:
                return ValidationResult(
                    is_valid=False,
                    confidence=0.0,
                    error_message="Missing states or actions for safety validation",
                    validation_time=time.time() - start_time
                )
            
            # Ensure we have batch data
            if len(states.shape) == 1:
                states = states[None, :]
            if len(actions.shape) == 1:
                actions = actions[None, :]
            
            batch_size = states.shape[0]
            constraint_violations = {}
            total_violations = 0
            critical_violations = 0
            
            for constraint in self.safety_constraints:
                try:
                    # Check constraint for each sample in batch
                    violations = 0
                    for i in range(batch_size):
                        state = states[i]
                        action = actions[i] if i < actions.shape[0] else actions[-1]
                        
                        if not constraint.check_fn(state, action):
                            violations += 1
                            if constraint.critical:
                                critical_violations += 1
                    
                    violation_rate = violations / batch_size
                    constraint_violations[constraint.name] = {
                        "violations": violations,
                        "rate": violation_rate,
                        "critical": constraint.critical,
                    }
                    
                    total_violations += violations
                    
                except Exception as e:
                    # If constraint check fails, consider it a violation
                    constraint_violations[constraint.name] = {
                        "violations": batch_size,
                        "rate": 1.0,
                        "critical": constraint.critical,
                        "error": str(e),
                    }
                    total_violations += batch_size
                    if constraint.critical:
                        critical_violations += batch_size
            
            # Determine validity
            total_violation_rate = total_violations / (batch_size * len(self.safety_constraints))
            
            # Critical violations are not tolerated
            if critical_violations > 0:
                is_valid = False
                confidence = 0.0
                error_message = f"Critical safety violations detected: {critical_violations}"
            elif total_violation_rate <= self.tolerance:
                is_valid = True
                confidence = 1.0 - total_violation_rate
                error_message = None
            else:
                is_valid = False
                confidence = max(0.1, 1.0 - total_violation_rate)
                error_message = f"Safety violation rate {total_violation_rate:.2%} exceeds tolerance {self.tolerance:.2%}"
            
            if is_valid:
                self.success_count += 1
            
            self.last_validation_time = time.time()
            
            recommendations = []
            if total_violation_rate > 0:
                recommendations.append("Review safety constraint implementations")
                recommendations.append("Consider adjusting control policies")
                if critical_violations > 0:
                    recommendations.append("IMMEDIATE ACTION: Critical safety violations detected")
            
            return ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                error_message=error_message,
                details={
                    "constraint_violations": constraint_violations,
                    "total_violation_rate": total_violation_rate,
                    "critical_violations": critical_violations,
                },
                recommendations=recommendations,
                validation_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                error_message=f"Safety validation error: {str(e)}",
                validation_time=time.time() - start_time
            )

class StatisticalValidator(BaseValidator):
    """Validates data using statistical methods."""
    
    def __init__(self, reference_stats: Optional[Dict[str, Dict[str, float]]] = None):
        super().__init__("statistical_validator")
        self.reference_stats = reference_stats or {}
        self.learned_stats = defaultdict(lambda: {"values": [], "count": 0})
        self.min_samples = 100
    
    def validate(self, data: Dict[str, Array], context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate using statistical methods."""
        start_time = time.time()
        self.validation_count += 1
        
        anomalies = []
        confidence_scores = []
        
        try:
            for key, values in data.items():
                if not isinstance(values, (np.ndarray, jnp.ndarray)):
                    continue
                
                values_flat = np.array(values).flatten()
                
                # Update learned statistics
                self.learned_stats[key]["values"].extend(values_flat.tolist())
                self.learned_stats[key]["count"] += len(values_flat)
                
                # Keep only recent values to prevent memory issues
                if len(self.learned_stats[key]["values"]) > 10000:
                    self.learned_stats[key]["values"] = self.learned_stats[key]["values"][-5000:]
                
                # Calculate current statistics
                current_mean = np.mean(values_flat)
                current_std = np.std(values_flat)
                current_min = np.min(values_flat)
                current_max = np.max(values_flat)
                
                # Check against reference statistics if available
                if key in self.reference_stats:
                    ref_stats = self.reference_stats[key]
                    
                    # Check mean deviation
                    if "mean" in ref_stats:
                        mean_deviation = abs(current_mean - ref_stats["mean"])
                        expected_std = ref_stats.get("std", 1.0)
                        if mean_deviation > 3 * expected_std:
                            anomalies.append(f"{key}: mean deviation {mean_deviation:.3f} > 3σ")
                            confidence_scores.append(0.3)
                        else:
                            confidence_scores.append(1.0)
                    
                    # Check standard deviation
                    if "std" in ref_stats:
                        std_ratio = current_std / ref_stats["std"] if ref_stats["std"] > 0 else float('inf')
                        if std_ratio > 2.0 or std_ratio < 0.5:
                            anomalies.append(f"{key}: std ratio {std_ratio:.3f} outside [0.5, 2.0]")
                            confidence_scores.append(0.4)
                        else:
                            confidence_scores.append(1.0)
                
                # Check against learned statistics
                elif self.learned_stats[key]["count"] >= self.min_samples:
                    learned_values = np.array(self.learned_stats[key]["values"])
                    learned_mean = np.mean(learned_values)
                    learned_std = np.std(learned_values)
                    
                    # Z-score based anomaly detection
                    if learned_std > 0:
                        z_score = abs(current_mean - learned_mean) / learned_std
                        if z_score > 3:
                            anomalies.append(f"{key}: z-score {z_score:.3f} > 3")
                            confidence_scores.append(0.2)
                        else:
                            confidence_scores.append(1.0)
                else:
                    # Not enough samples for validation
                    confidence_scores.append(0.8)
            
            # Determine overall validity
            if not anomalies:
                is_valid = True
                confidence = np.mean(confidence_scores) if confidence_scores else 0.5
                error_message = None
            else:
                is_valid = len(anomalies) <= 1  # Allow one anomaly
                confidence = max(0.1, np.mean(confidence_scores) if confidence_scores else 0.1)
                error_message = "; ".join(anomalies)
            
            if is_valid:
                self.success_count += 1
            
            self.last_validation_time = time.time()
            
            return ValidationResult(
                is_valid=is_valid,
                confidence=confidence,
                error_message=error_message,
                details={"anomalies": anomalies} if anomalies else None,
                validation_time=time.time() - start_time
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                error_message=f"Statistical validation error: {str(e)}",
                validation_time=time.time() - start_time
            )

class ComprehensiveValidator:
    """Comprehensive validation system combining multiple validators."""
    
    def __init__(self):
        self.validators: List[BaseValidator] = []
        self.validation_history = deque(maxlen=1000)
        self.enable_parallel = True
        self._lock = threading.Lock()
        self.logger = logging.getLogger("comprehensive_validator")
    
    def add_validator(self, validator: BaseValidator) -> None:
        """Add a validator to the system."""
        with self._lock:
            self.validators.append(validator)
            self.logger.info(f"Added validator: {validator.name}")
    
    def remove_validator(self, validator_name: str) -> None:
        """Remove a validator by name."""
        with self._lock:
            self.validators = [v for v in self.validators if v.name != validator_name]
            self.logger.info(f"Removed validator: {validator_name}")
    
    def validate_all(self, data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, ValidationResult]:
        """Run all enabled validators."""
        start_time = time.time()
        results = {}
        
        active_validators = [v for v in self.validators if v.enabled]
        
        if self.enable_parallel and len(active_validators) > 1:
            # Parallel validation
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_validator = {
                    executor.submit(validator.validate, data, context): validator
                    for validator in active_validators
                }
                
                for future in concurrent.futures.as_completed(future_to_validator):
                    validator = future_to_validator[future]
                    try:
                        result = future.result(timeout=30)  # 30 second timeout
                        results[validator.name] = result
                    except Exception as e:
                        results[validator.name] = ValidationResult(
                            is_valid=False,
                            confidence=0.0,
                            error_message=f"Validation failed: {str(e)}"
                        )
        else:
            # Sequential validation
            for validator in active_validators:
                try:
                    result = validator.validate(data, context)
                    results[validator.name] = result
                except Exception as e:
                    results[validator.name] = ValidationResult(
                        is_valid=False,
                        confidence=0.0,
                        error_message=f"Validation failed: {str(e)}"
                    )
        
        # Store validation history
        validation_record = {
            "timestamp": time.time(),
            "results": results,
            "total_time": time.time() - start_time,
            "data_size": self._estimate_data_size(data),
        }
        
        with self._lock:
            self.validation_history.append(validation_record)
        
        return results
    
    def get_overall_validity(self, results: Dict[str, ValidationResult]) -> Tuple[bool, float]:
        """Get overall validity and confidence from individual results."""
        if not results:
            return False, 0.0
        
        valid_results = [r for r in results.values() if r.is_valid]
        invalid_results = [r for r in results.values() if not r.is_valid]
        
        # System is valid if majority of validators pass
        is_valid = len(valid_results) > len(invalid_results)
        
        # Confidence is weighted average
        total_confidence = sum(r.confidence for r in results.values())
        average_confidence = total_confidence / len(results)
        
        # Penalize if critical validators fail
        critical_failures = sum(1 for r in invalid_results if r.confidence < 0.1)
        if critical_failures > 0:
            average_confidence *= 0.5
        
        return is_valid, average_confidence
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation performance."""
        with self._lock:
            validator_stats = [v.get_stats() for v in self.validators]
            
            recent_validations = list(self.validation_history)[-100:]  # Last 100 validations
            
            if recent_validations:
                avg_time = np.mean([v["total_time"] for v in recent_validations])
                success_rates = []
                
                for validator_name in [v.name for v in self.validators]:
                    successes = sum(
                        1 for validation in recent_validations
                        if validation["results"].get(validator_name, {}).get("is_valid", False)
                    )
                    rate = successes / len(recent_validations)
                    success_rates.append(rate)
                
                overall_success_rate = np.mean(success_rates) if success_rates else 0.0
            else:
                avg_time = 0.0
                overall_success_rate = 0.0
            
            return {
                "total_validators": len(self.validators),
                "enabled_validators": len([v for v in self.validators if v.enabled]),
                "validator_stats": validator_stats,
                "recent_average_time": avg_time,
                "overall_success_rate": overall_success_rate,
                "total_validations": len(self.validation_history),
            }
    
    def _estimate_data_size(self, data: Any) -> int:
        """Estimate size of data being validated."""
        if isinstance(data, dict):
            return sum(
                arr.size if hasattr(arr, 'size') else len(str(arr))
                for arr in data.values()
            )
        elif hasattr(data, 'size'):
            return data.size
        else:
            return len(str(data))

# Factory functions for common validation setups

def create_industrial_env_validator(
    state_dim: int,
    action_dim: int,
    safety_constraints: List[SafetyConstraint],
    action_bounds: Tuple[float, float] = (-1.0, 1.0),
) -> ComprehensiveValidator:
    """Create validator for industrial environments."""
    validator = ComprehensiveValidator()
    
    # Shape validator
    shape_validator = DataShapeValidator({
        "observations": (state_dim,),
        "actions": (action_dim,),
    })
    validator.add_validator(shape_validator)
    
    # Range validator
    range_validator = DataRangeValidator({
        "actions": action_bounds,
        "observations": (-1e6, 1e6),  # Reasonable bounds for observations
    })
    validator.add_validator(range_validator)
    
    # Safety validator
    safety_validator = SafetyConstraintValidator(safety_constraints)
    validator.add_validator(safety_validator)
    
    # Statistical validator
    stats_validator = StatisticalValidator()
    validator.add_validator(stats_validator)
    
    return validator

def create_agent_validator(
    state_dim: int,
    action_dim: int,
) -> ComprehensiveValidator:
    """Create validator for agent training data."""
    validator = ComprehensiveValidator()
    
    # Shape validator for training data
    shape_validator = DataShapeValidator({
        "observations": (state_dim,),
        "actions": (action_dim,),
        "rewards": (),  # Scalar
        "terminals": (),  # Scalar
    })
    validator.add_validator(shape_validator)
    
    # Range validator for training data
    range_validator = DataRangeValidator({
        "actions": (-1.0, 1.0),
        "rewards": (-1000.0, 1000.0),  # Reasonable reward bounds
        "observations": (-1e6, 1e6),
    }, strict=False)  # Allow some out-of-range values in training data
    validator.add_validator(range_validator)
    
    # Statistical validator
    stats_validator = StatisticalValidator()
    validator.add_validator(stats_validator)
    
    return validator