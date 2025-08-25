"""Advanced circuit breaker with adaptive thresholds and predictive failure detection."""

import time
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
from functools import wraps

from ..monitoring.logger import get_logger
from ..monitoring.performance import get_performance_monitor


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class FailureMetrics:
    """Failure tracking metrics."""
    total_requests: int = 0
    failed_requests: int = 0
    success_requests: int = 0
    last_failure_time: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_types: Dict[str, int] = field(default_factory=dict)
    
    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        if not self.response_times:
            return 0.0
        return sum(self.response_times) / len(self.response_times)


class AdaptiveCircuitBreaker:
    """Advanced circuit breaker with adaptive thresholds and predictive capabilities."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: float = 0.5,
        recovery_timeout: float = 60.0,
        request_volume_threshold: int = 20,
        half_open_max_calls: int = 5,
        enable_prediction: bool = True,
        adaptive_threshold: bool = True
    ):
        """Initialize adaptive circuit breaker.
        
        Args:
            name: Circuit breaker identifier
            failure_threshold: Initial failure rate threshold (0-1)
            recovery_timeout: Time to wait before attempting recovery
            request_volume_threshold: Minimum requests before evaluation
            half_open_max_calls: Max calls in half-open state
            enable_prediction: Enable predictive failure detection
            adaptive_threshold: Enable adaptive threshold adjustment
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.request_volume_threshold = request_volume_threshold
        self.half_open_max_calls = half_open_max_calls
        self.enable_prediction = enable_prediction
        self.adaptive_threshold = adaptive_threshold
        
        # Circuit state
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        
        # Metrics tracking
        self.metrics = FailureMetrics()
        self.state_history = deque(maxlen=1000)
        self.threshold_history = deque(maxlen=100)
        
        # Adaptive threshold parameters
        self.min_threshold = 0.1
        self.max_threshold = 0.9
        self.adaptation_rate = 0.05
        
        # Predictive failure detection
        self.anomaly_detector = None
        self.prediction_confidence = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Logging and monitoring
        self.logger = get_logger(f"CircuitBreaker_{name}")
        self.performance_monitor = get_performance_monitor(f"circuit_{name}")
        
        self.logger.info(f"Initialized circuit breaker '{name}' with adaptive features")
        
        if self.enable_prediction:
            self._initialize_anomaly_detector()
    
    def _initialize_anomaly_detector(self) -> None:
        """Initialize anomaly detector for predictive failure detection."""
        try:
            from sklearn.ensemble import IsolationForest
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=50
            )
            self.logger.info("Initialized anomaly detector for predictive failure detection")
        except ImportError:
            self.logger.warning("sklearn not available, disabling predictive failure detection")
            self.enable_prediction = False
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to protect function with circuit breaker."""
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        return wrapper
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        
        with self._lock:
            # Check if circuit is open
            if self._should_block_request():
                self._handle_blocked_request()
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Failure rate: {self.metrics.failure_rate:.2f}, "
                    f"Threshold: {self.failure_threshold:.2f}"
                )
            
            # Record request attempt
            self.metrics.total_requests += 1
            start_time = time.time()
            
            try:
                # Execute the protected function
                with self.performance_monitor.time_operation(f"circuit_call_{self.name}"):
                    result = func(*args, **kwargs)
                
                # Record success
                response_time = time.time() - start_time
                self._record_success(response_time)
                
                return result
                
            except Exception as e:
                # Record failure
                response_time = time.time() - start_time
                self._record_failure(e, response_time)
                
                # Update circuit state
                self._update_circuit_state()
                
                # Re-raise the original exception
                raise
    
    def _should_block_request(self) -> bool:
        """Determine if request should be blocked."""
        
        current_time = time.time()
        
        # Circuit is OPEN
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if current_time - self.last_failure_time >= self.recovery_timeout:
                self._transition_to_half_open()
                return False
            return True
        
        # Circuit is HALF_OPEN
        if self.state == CircuitState.HALF_OPEN:
            # Limit calls in half-open state
            if self.half_open_calls >= self.half_open_max_calls:
                return True
            self.half_open_calls += 1
            return False
        
        # Circuit is CLOSED - check for predictive failure
        if self.enable_prediction and self._predict_failure():
            self.logger.warning(f"Predictive failure detected for circuit '{self.name}'")
            return True
        
        return False
    
    def _record_success(self, response_time: float) -> None:
        """Record successful request."""
        self.metrics.success_requests += 1
        self.metrics.consecutive_successes += 1
        self.metrics.consecutive_failures = 0
        self.metrics.response_times.append(response_time)
        
        # If we're in HALF_OPEN and have enough successes, close the circuit
        if (
            self.state == CircuitState.HALF_OPEN and
            self.metrics.consecutive_successes >= 3
        ):
            self._transition_to_closed()
        
        self.logger.debug(f"Circuit '{self.name}': Success recorded (response_time={response_time:.3f}s)")
    
    def _record_failure(self, exception: Exception, response_time: float) -> None:
        """Record failed request."""
        self.metrics.failed_requests += 1
        self.metrics.consecutive_failures += 1
        self.metrics.consecutive_successes = 0
        self.metrics.response_times.append(response_time)
        self.last_failure_time = time.time()
        
        # Track error types
        error_type = type(exception).__name__
        self.metrics.error_types[error_type] = self.metrics.error_types.get(error_type, 0) + 1
        
        self.logger.warning(
            f"Circuit '{self.name}': Failure recorded - {error_type} "
            f"(consecutive={self.metrics.consecutive_failures})"
        )
    
    def _update_circuit_state(self) -> None:
        """Update circuit state based on current metrics."""
        
        # Must have minimum volume of requests
        if self.metrics.total_requests < self.request_volume_threshold:
            return
        
        current_failure_rate = self.metrics.failure_rate
        current_threshold = self._get_adaptive_threshold()
        
        # Transition to OPEN if failure rate exceeds threshold
        if (
            current_failure_rate >= current_threshold and
            self.state != CircuitState.OPEN
        ):
            self._transition_to_open()
        
        # Record state change
        self.state_history.append({
            'timestamp': time.time(),
            'state': self.state.value,
            'failure_rate': current_failure_rate,
            'threshold': current_threshold
        })
    
    def _get_adaptive_threshold(self) -> float:
        """Get adaptive failure threshold."""
        
        if not self.adaptive_threshold:
            return self.failure_threshold
        
        # Adapt threshold based on recent performance
        if len(self.metrics.response_times) >= 20:
            recent_times = list(self.metrics.response_times)[-20:]
            avg_time = sum(recent_times) / len(recent_times)
            time_variance = np.var(recent_times)
            
            # Lower threshold if response times are unstable
            instability_factor = min(time_variance / (avg_time + 1e-6), 1.0)
            adapted_threshold = self.failure_threshold - (instability_factor * self.adaptation_rate)
            
            # Keep within bounds
            adapted_threshold = max(self.min_threshold, min(self.max_threshold, adapted_threshold))
            
            # Record threshold adaptation
            self.threshold_history.append(adapted_threshold)
            
            return adapted_threshold
        
        return self.failure_threshold
    
    def _predict_failure(self) -> bool:
        """Predict if a failure is likely to occur."""
        
        if not self.enable_prediction or self.anomaly_detector is None:
            return False
        
        # Need sufficient data for prediction
        if len(self.metrics.response_times) < 10:
            return False
        
        try:
            # Prepare feature vector for anomaly detection
            features = self._extract_prediction_features()
            
            # Predict anomaly
            prediction = self.anomaly_detector.predict([features])[0]
            
            # -1 indicates anomaly (potential failure)
            if prediction == -1:
                # Update prediction confidence
                decision_score = abs(self.anomaly_detector.decision_function([features])[0])
                self.prediction_confidence = decision_score
                
                return decision_score > 0.1  # Threshold for prediction confidence
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Failure prediction error: {e}")
            return False
    
    def _extract_prediction_features(self) -> List[float]:
        """Extract features for failure prediction."""
        recent_times = list(self.metrics.response_times)[-10:]
        
        features = [
            self.metrics.failure_rate,
            self.metrics.consecutive_failures,
            len(recent_times),
            np.mean(recent_times),
            np.std(recent_times),
            np.max(recent_times) - np.min(recent_times),  # Range
            time.time() - self.last_failure_time if self.last_failure_time > 0 else 1000,
        ]
        
        return features
    
    def _transition_to_open(self) -> None:
        """Transition circuit to OPEN state."""
        previous_state = self.state
        self.state = CircuitState.OPEN
        self.last_failure_time = time.time()
        
        self.logger.warning(
            f"Circuit '{self.name}': {previous_state.value} -> OPEN "
            f"(failure_rate={self.metrics.failure_rate:.2f}, "
            f"threshold={self._get_adaptive_threshold():.2f})"
        )
        
        self.performance_monitor.record_event("circuit_opened")
    
    def _transition_to_half_open(self) -> None:
        """Transition circuit to HALF_OPEN state."""
        previous_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.half_open_calls = 0
        
        self.logger.info(f"Circuit '{self.name}': {previous_state.value} -> HALF_OPEN")
        self.performance_monitor.record_event("circuit_half_opened")
    
    def _transition_to_closed(self) -> None:
        """Transition circuit to CLOSED state."""
        previous_state = self.state
        self.state = CircuitState.CLOSED
        
        # Reset metrics for fresh start
        self.metrics = FailureMetrics()
        
        self.logger.info(f"Circuit '{self.name}': {previous_state.value} -> CLOSED")
        self.performance_monitor.record_event("circuit_closed")
    
    def _handle_blocked_request(self) -> None:
        """Handle blocked request."""
        self.performance_monitor.record_event("request_blocked")
        
        self.logger.debug(f"Circuit '{self.name}': Request blocked (state={self.state.value})")
    
    def force_open(self) -> None:
        """Force circuit to OPEN state."""
        with self._lock:
            self._transition_to_open()
            self.logger.warning(f"Circuit '{self.name}': Forced to OPEN state")
    
    def force_closed(self) -> None:
        """Force circuit to CLOSED state."""
        with self._lock:
            self._transition_to_closed()
            self.logger.info(f"Circuit '{self.name}': Forced to CLOSED state")
    
    def reset_metrics(self) -> None:
        """Reset circuit metrics."""
        with self._lock:
            self.metrics = FailureMetrics()
            self.state_history.clear()
            self.threshold_history.clear()
            self.logger.info(f"Circuit '{self.name}': Metrics reset")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive circuit metrics."""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_threshold': self._get_adaptive_threshold(),
                'failure_rate': self.metrics.failure_rate,
                'total_requests': self.metrics.total_requests,
                'failed_requests': self.metrics.failed_requests,
                'success_requests': self.metrics.success_requests,
                'consecutive_failures': self.metrics.consecutive_failures,
                'consecutive_successes': self.metrics.consecutive_successes,
                'average_response_time': self.metrics.average_response_time,
                'last_failure_time': self.last_failure_time,
                'prediction_confidence': self.prediction_confidence,
                'error_types': dict(self.metrics.error_types),
                'state_transitions': len([h for h in self.state_history if h['state'] != 'closed']),
                'adaptive_features': {
                    'adaptive_threshold_enabled': self.adaptive_threshold,
                    'predictive_detection_enabled': self.enable_prediction,
                    'current_adaptive_threshold': self._get_adaptive_threshold(),
                    'threshold_variance': np.var(list(self.threshold_history)) if self.threshold_history else 0.0
                }
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get circuit health status."""
        metrics = self.get_metrics()
        
        # Calculate health score (0-1)
        health_score = 1.0
        
        # Penalize for high failure rate
        health_score -= metrics['failure_rate'] * 0.5
        
        # Penalize for being open
        if self.state == CircuitState.OPEN:
            health_score -= 0.3
        elif self.state == CircuitState.HALF_OPEN:
            health_score -= 0.1
        
        # Penalize for slow response times
        avg_time = metrics['average_response_time']
        if avg_time > 1.0:  # 1 second threshold
            health_score -= min(0.2, (avg_time - 1.0) * 0.1)
        
        health_score = max(0.0, min(1.0, health_score))
        
        return {
            'health_score': health_score,
            'status': 'healthy' if health_score > 0.7 else 'degraded' if health_score > 0.3 else 'unhealthy',
            'recommendations': self._generate_health_recommendations(metrics, health_score)
        }
    
    def _generate_health_recommendations(self, metrics: Dict[str, Any], health_score: float) -> List[str]:
        """Generate health recommendations."""
        recommendations = []
        
        if metrics['failure_rate'] > 0.2:
            recommendations.append("High failure rate detected - investigate underlying issues")
        
        if metrics['average_response_time'] > 2.0:
            recommendations.append("High response times - consider performance optimization")
        
        if self.state == CircuitState.OPEN:
            recommendations.append("Circuit is open - check service health and dependencies")
        
        if len(metrics['error_types']) > 3:
            recommendations.append("Multiple error types detected - review error handling")
        
        if health_score < 0.5:
            recommendations.append("Critical health issues - immediate attention required")
        
        return recommendations


class CircuitBreakerManager:
    """Manager for multiple circuit breakers."""
    
    def __init__(self):
        """Initialize circuit breaker manager."""
        self.breakers: Dict[str, AdaptiveCircuitBreaker] = {}
        self._lock = threading.RLock()
        
        self.logger = get_logger("CircuitBreakerManager")
    
    def get_breaker(
        self,
        name: str,
        **kwargs
    ) -> AdaptiveCircuitBreaker:
        """Get or create circuit breaker."""
        
        with self._lock:
            if name not in self.breakers:
                self.breakers[name] = AdaptiveCircuitBreaker(name=name, **kwargs)
                self.logger.info(f"Created circuit breaker '{name}'")
            
            return self.breakers[name]
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        with self._lock:
            return {
                name: breaker.get_metrics()
                for name, breaker in self.breakers.items()
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health based on all circuit breakers."""
        with self._lock:
            if not self.breakers:
                return {'status': 'unknown', 'health_score': 1.0}
            
            health_scores = []
            unhealthy_breakers = []
            
            for name, breaker in self.breakers.items():
                health = breaker.get_health_status()
                health_scores.append(health['health_score'])
                
                if health['status'] == 'unhealthy':
                    unhealthy_breakers.append(name)
            
            avg_health = sum(health_scores) / len(health_scores)
            
            # Overall status
            if avg_health > 0.8:
                status = 'healthy'
            elif avg_health > 0.5:
                status = 'degraded'
            else:
                status = 'unhealthy'
            
            return {
                'status': status,
                'health_score': avg_health,
                'total_breakers': len(self.breakers),
                'unhealthy_breakers': unhealthy_breakers,
                'breaker_states': {
                    name: breaker.state.value
                    for name, breaker in self.breakers.items()
                }
            }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Global circuit breaker manager instance
_circuit_manager = CircuitBreakerManager()


def circuit_breaker(
    name: str,
    failure_threshold: float = 0.5,
    recovery_timeout: float = 60.0,
    **kwargs
) -> AdaptiveCircuitBreaker:
    """Get circuit breaker decorator/context manager."""
    return _circuit_manager.get_breaker(
        name=name,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        **kwargs
    )


def get_circuit_manager() -> CircuitBreakerManager:
    """Get global circuit breaker manager."""
    return _circuit_manager