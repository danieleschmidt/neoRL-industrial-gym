"""Enhanced circuit breaker with adaptive thresholds and recovery strategies."""

import time
import threading
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered

@dataclass
class CircuitMetrics:
    """Circuit breaker metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeouts: int = 0
    circuit_opens: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    current_streak: int = 0  # Current success/failure streak
    
    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        return 1.0 - self.failure_rate

class AdaptiveCircuitBreaker:
    """Enhanced circuit breaker with adaptive thresholds and recovery strategies."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout_duration: float = 60.0,
        request_timeout: float = 30.0,
        sliding_window_size: int = 100,
        min_requests_threshold: int = 10,
        adaptive_threshold: bool = True,
        recovery_strategy: str = "exponential_backoff",  # exponential_backoff, linear, immediate
    ):
        """Initialize enhanced circuit breaker.
        
        Args:
            name: Circuit breaker identifier
            failure_threshold: Number of failures before opening circuit
            success_threshold: Number of successes before closing circuit
            timeout_duration: Duration to wait before trying half-open
            request_timeout: Maximum time to wait for a request
            sliding_window_size: Size of sliding window for metrics
            min_requests_threshold: Minimum requests before evaluating failure rate
            adaptive_threshold: Whether to adapt thresholds based on patterns
            recovery_strategy: Strategy for recovery timing
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout_duration = timeout_duration
        self.request_timeout = request_timeout
        self.sliding_window_size = sliding_window_size
        self.min_requests_threshold = min_requests_threshold
        self.adaptive_threshold = adaptive_threshold
        self.recovery_strategy = recovery_strategy
        
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self.last_failure_time = None
        self.state_change_time = time.time()
        
        # Sliding window for recent requests
        self.recent_requests = deque(maxlen=sliding_window_size)
        
        # Adaptive thresholds
        self.adaptive_failure_threshold = failure_threshold
        self.historical_failure_rates = deque(maxlen=50)
        
        # Recovery backoff
        self.consecutive_failures = 0
        self.base_timeout = timeout_duration
        
        self._lock = threading.Lock()
        self.logger = logging.getLogger(f"circuit_breaker.{name}")
        
        # Monitoring hooks
        self.on_state_change: Optional[Callable] = None
        self.on_failure: Optional[Callable] = None
        self.on_success: Optional[Callable] = None
    
    def call(self, func: Callable[[], Any], *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            # Check if circuit should be opened
            if self._should_trip():
                self._open_circuit()
            
            # Handle different states
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._half_open_circuit()
                else:
                    self._record_blocked_request()
                    raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")
            
            elif self.state == CircuitState.HALF_OPEN:
                # Allow limited requests in half-open state
                pass
        
        # Execute the function
        request_start_time = time.time()
        try:
            # Apply timeout
            result = self._execute_with_timeout(func, *args, **kwargs)
            
            # Record success
            self._record_success(time.time() - request_start_time)
            return result
            
        except Exception as e:
            # Record failure
            self._record_failure(time.time() - request_start_time, e)
            raise
    
    def _execute_with_timeout(self, func: Callable[[], Any], *args, **kwargs) -> Any:
        """Execute function with timeout protection."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Request timeout after {self.request_timeout}s")
        
        # Set up timeout (Unix systems only)
        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.request_timeout))
            
            try:
                return func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
        except AttributeError:
            # Windows doesn't support SIGALRM, fallback to basic execution
            return func(*args, **kwargs)
    
    def _record_success(self, response_time: float) -> None:
        """Record successful request."""
        with self._lock:
            current_time = time.time()
            
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = current_time
            
            # Update sliding window
            self.recent_requests.append({
                "timestamp": current_time,
                "success": True,
                "response_time": response_time,
            })
            
            # Update success streak
            if hasattr(self, '_last_result') and self._last_result:
                self.metrics.current_streak += 1
            else:
                self.metrics.current_streak = 1
            self._last_result = True
            
            # Reset consecutive failures
            self.consecutive_failures = 0
            
            # Check if we should close the circuit
            if self.state == CircuitState.HALF_OPEN:
                if self.metrics.current_streak >= self.success_threshold:
                    self._close_circuit()
            
            # Call success hook
            if self.on_success:
                self.on_success(self.name, response_time)
            
            self.logger.debug(f"Success recorded: response_time={response_time:.3f}s")
    
    def _record_failure(self, response_time: float, exception: Exception) -> None:
        """Record failed request."""
        with self._lock:
            current_time = time.time()
            
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = current_time
            
            # Check if it's a timeout
            if isinstance(exception, TimeoutError):
                self.metrics.timeouts += 1
            
            # Update sliding window
            self.recent_requests.append({
                "timestamp": current_time,
                "success": False,
                "response_time": response_time,
                "exception": str(exception),
            })
            
            # Update failure streak
            if hasattr(self, '_last_result') and not self._last_result:
                self.metrics.current_streak += 1
            else:
                self.metrics.current_streak = 1
            self._last_result = False
            
            # Increment consecutive failures
            self.consecutive_failures += 1
            
            # Adapt thresholds if enabled
            if self.adaptive_threshold:
                self._adapt_thresholds()
            
            # Call failure hook
            if self.on_failure:
                self.on_failure(self.name, exception, response_time)
            
            self.logger.warning(f"Failure recorded: {exception}, response_time={response_time:.3f}s")
    
    def _record_blocked_request(self) -> None:
        """Record a request that was blocked by open circuit."""
        with self._lock:
            self.metrics.total_requests += 1
            # Don't increment failed_requests for blocked requests
            
            self.logger.debug("Request blocked - circuit is OPEN")
    
    def _should_trip(self) -> bool:
        """Determine if circuit should trip to OPEN state."""
        if self.state != CircuitState.CLOSED:
            return False
        
        # Need minimum number of requests
        if len(self.recent_requests) < self.min_requests_threshold:
            return False
        
        # Calculate recent failure rate
        recent_failures = sum(1 for req in self.recent_requests if not req["success"])
        recent_failure_rate = recent_failures / len(self.recent_requests)
        
        # Use adaptive threshold or fixed threshold
        threshold = self.adaptive_failure_threshold / len(self.recent_requests)
        
        return recent_failure_rate >= threshold
    
    def _should_attempt_reset(self) -> bool:
        """Determine if circuit should attempt reset to HALF_OPEN state."""
        if self.state != CircuitState.OPEN:
            return False
        
        current_time = time.time()
        time_since_failure = current_time - (self.metrics.last_failure_time or 0)
        
        # Calculate timeout based on recovery strategy
        timeout = self._calculate_recovery_timeout()
        
        return time_since_failure >= timeout
    
    def _calculate_recovery_timeout(self) -> float:
        """Calculate timeout for recovery based on strategy."""
        if self.recovery_strategy == "immediate":
            return 0.0
        
        elif self.recovery_strategy == "linear":
            return self.base_timeout + (self.consecutive_failures * 10)
        
        elif self.recovery_strategy == "exponential_backoff":
            # Exponential backoff with jitter
            import random
            backoff = min(self.base_timeout * (2 ** self.consecutive_failures), 300)  # Max 5 minutes
            jitter = random.uniform(0.1, 0.3) * backoff
            return backoff + jitter
        
        else:
            return self.base_timeout
    
    def _adapt_thresholds(self) -> None:
        """Adapt failure thresholds based on historical patterns."""
        if not self.adaptive_threshold:
            return
        
        # Calculate recent failure rate
        if len(self.recent_requests) >= self.min_requests_threshold:
            recent_failures = sum(1 for req in self.recent_requests if not req["success"])
            failure_rate = recent_failures / len(self.recent_requests)
            self.historical_failure_rates.append(failure_rate)
        
        # Adjust threshold based on historical patterns
        if len(self.historical_failure_rates) >= 10:
            avg_failure_rate = sum(self.historical_failure_rates) / len(self.historical_failure_rates)
            
            # If historically high failure rate, be more permissive
            # If historically low failure rate, be more strict
            if avg_failure_rate > 0.1:  # 10% historical failure rate
                self.adaptive_failure_threshold = min(self.failure_threshold + 2, 15)
            elif avg_failure_rate < 0.02:  # 2% historical failure rate
                self.adaptive_failure_threshold = max(self.failure_threshold - 1, 2)
            else:
                self.adaptive_failure_threshold = self.failure_threshold
    
    def _open_circuit(self) -> None:
        """Transition to OPEN state."""
        if self.state != CircuitState.OPEN:
            old_state = self.state
            self.state = CircuitState.OPEN
            self.state_change_time = time.time()
            self.metrics.circuit_opens += 1
            
            self.logger.warning(f"Circuit breaker {self.name} opened: {old_state} -> OPEN")
            
            if self.on_state_change:
                self.on_state_change(self.name, old_state, CircuitState.OPEN)
    
    def _half_open_circuit(self) -> None:
        """Transition to HALF_OPEN state."""
        if self.state != CircuitState.HALF_OPEN:
            old_state = self.state
            self.state = CircuitState.HALF_OPEN
            self.state_change_time = time.time()
            self.metrics.current_streak = 0  # Reset streak for testing
            
            self.logger.info(f"Circuit breaker {self.name} half-opened: {old_state} -> HALF_OPEN")
            
            if self.on_state_change:
                self.on_state_change(self.name, old_state, CircuitState.HALF_OPEN)
    
    def _close_circuit(self) -> None:
        """Transition to CLOSED state."""
        if self.state != CircuitState.CLOSED:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.state_change_time = time.time()
            self.consecutive_failures = 0  # Reset backoff
            
            self.logger.info(f"Circuit breaker {self.name} closed: {old_state} -> CLOSED")
            
            if self.on_state_change:
                self.on_state_change(self.name, old_state, CircuitState.CLOSED)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics."""
        with self._lock:
            recent_window_size = min(len(self.recent_requests), 20)
            recent_requests = list(self.recent_requests)[-recent_window_size:]
            
            return {
                "name": self.name,
                "state": self.state.value,
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "timeouts": self.metrics.timeouts,
                "circuit_opens": self.metrics.circuit_opens,
                "failure_rate": self.metrics.failure_rate,
                "success_rate": self.metrics.success_rate,
                "current_streak": self.metrics.current_streak,
                "consecutive_failures": self.consecutive_failures,
                "adaptive_failure_threshold": self.adaptive_failure_threshold,
                "time_in_current_state": time.time() - self.state_change_time,
                "recent_requests_count": len(recent_requests),
                "recent_success_rate": (
                    sum(1 for r in recent_requests if r["success"]) / len(recent_requests)
                    if recent_requests else 1.0
                ),
            }
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        with self._lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.metrics = CircuitMetrics()
            self.recent_requests.clear()
            self.consecutive_failures = 0
            self.state_change_time = time.time()
            self.adaptive_failure_threshold = self.failure_threshold
            
            self.logger.info(f"Circuit breaker {self.name} reset: {old_state} -> CLOSED")
    
    def force_open(self) -> None:
        """Force circuit breaker to OPEN state."""
        with self._lock:
            self._open_circuit()
            self.logger.warning(f"Circuit breaker {self.name} force opened")
    
    def force_close(self) -> None:
        """Force circuit breaker to CLOSED state."""
        with self._lock:
            self._close_circuit()
            self.logger.info(f"Circuit breaker {self.name} force closed")


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, AdaptiveCircuitBreaker] = {}
        self._lock = threading.Lock()
    
    def get_breaker(self, name: str, **kwargs) -> AdaptiveCircuitBreaker:
        """Get or create a circuit breaker."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = AdaptiveCircuitBreaker(name=name, **kwargs)
            return self._breakers[name]
    
    def remove_breaker(self, name: str) -> None:
        """Remove a circuit breaker."""
        with self._lock:
            self._breakers.pop(name, None)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        with self._lock:
            return {name: breaker.get_metrics() for name, breaker in self._breakers.items()}
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


# Global registry
_registry = CircuitBreakerRegistry()

def get_circuit_breaker(name: str, **kwargs) -> AdaptiveCircuitBreaker:
    """Get or create a circuit breaker from global registry."""
    return _registry.get_breaker(name, **kwargs)

def circuit_breaker(name: str, **kwargs):
    """Decorator for circuit breaker protection."""
    def decorator(func):
        breaker = get_circuit_breaker(name, **kwargs)
        
        def wrapper(*args, **kwargs):
            return breaker.call(func, *args, **kwargs)
        
        wrapper.circuit_breaker = breaker
        return wrapper
    
    return decorator