"""Circuit breaker pattern implementation for fault tolerance."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Union
import threading


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit tripped, rejecting calls
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5              # Failures before opening
    recovery_timeout: float = 60.0          # Seconds before trying half-open
    success_threshold: int = 3              # Successes needed to close from half-open
    timeout: float = 30.0                   # Call timeout in seconds
    excluded_exceptions: tuple = ()         # Exceptions that don't count as failures


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


F = TypeVar('F', bound=Callable[..., Any])


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        """Initialize circuit breaker.
        
        Args:
            name: Circuit breaker identifier
            config: Configuration parameters
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        
        # State management
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._next_attempt_time: Optional[float] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "rejected_calls": 0,
            "state_changes": 0,
        }
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state
    
    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self._state == CircuitState.CLOSED
    
    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting calls)."""
        return self._state == CircuitState.OPEN
    
    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing recovery)."""
        return self._state == CircuitState.HALF_OPEN
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                **self._stats,
                "current_state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "next_attempt_time": self._next_attempt_time,
            }
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._next_attempt_time = None
            self._stats["state_changes"] += 1
    
    def force_open(self) -> None:
        """Force circuit breaker to open state."""
        with self._lock:
            if self._state != CircuitState.OPEN:
                self._state = CircuitState.OPEN
                self._next_attempt_time = time.time() + self.config.recovery_timeout
                self._stats["state_changes"] += 1
    
    def force_close(self) -> None:
        """Force circuit breaker to closed state."""
        with self._lock:
            if self._state != CircuitState.CLOSED:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                self._last_failure_time = None
                self._next_attempt_time = None
                self._stats["state_changes"] += 1
    
    def can_execute(self) -> bool:
        """Check if execution is allowed in current state."""
        with self._lock:
            current_time = time.time()
            
            if self._state == CircuitState.CLOSED:
                return True
            
            elif self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if (self._next_attempt_time and 
                    current_time >= self._next_attempt_time):
                    # Transition to half-open
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
                    self._stats["state_changes"] += 1
                    return True
                else:
                    return False
            
            elif self._state == CircuitState.HALF_OPEN:
                return True
            
            return False
    
    def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args: Function positional arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerError: If circuit is open
            Exception: Original function exceptions
        """
        with self._lock:
            self._stats["total_calls"] += 1
            
            if not self.can_execute():
                self._stats["rejected_calls"] += 1
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN. "
                    f"Next attempt at {self._next_attempt_time}"
                )
        
        # Execute function with timeout and exception handling
        try:
            start_time = time.time()
            
            # Simple timeout mechanism
            result = func(*args, **kwargs)
            
            execution_time = time.time() - start_time
            if execution_time > self.config.timeout:
                raise TimeoutError(
                    f"Function execution exceeded timeout "
                    f"({execution_time:.2f}s > {self.config.timeout}s)"
                )
            
            # Success handling
            self._on_success()
            return result
            
        except Exception as e:
            # Check if exception should be excluded
            if isinstance(e, self.config.excluded_exceptions):
                return self._on_success()
            else:
                self._on_failure(e)
                raise
    
    def _on_success(self) -> None:
        """Handle successful execution."""
        with self._lock:
            self._stats["successful_calls"] += 1
            
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                
                # Check if enough successes to close circuit
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._last_failure_time = None
                    self._next_attempt_time = None
                    self._stats["state_changes"] += 1
    
    def _on_failure(self, exception: Exception) -> None:
        """Handle failed execution."""
        with self._lock:
            self._stats["failed_calls"] += 1
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Failure in half-open state - go back to open
                self._state = CircuitState.OPEN
                self._next_attempt_time = (
                    time.time() + self.config.recovery_timeout
                )
                self._stats["state_changes"] += 1
                
            elif self._state == CircuitState.CLOSED:
                # Check if failure threshold exceeded
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._next_attempt_time = (
                        time.time() + self.config.recovery_timeout
                    )
                    self._stats["state_changes"] += 1
    
    def __call__(self, func: F) -> F:
        """Decorator interface for circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        
        return wrapper  # type: ignore
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CircuitBreaker(name='{self.name}', "
            f"state={self._state.value}, "
            f"failures={self._failure_count})"
        )


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""
    
    def __init__(self) -> None:
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def get_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get or create circuit breaker by name."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]
    
    def remove_breaker(self, name: str) -> bool:
        """Remove circuit breaker from registry."""
        with self._lock:
            if name in self._breakers:
                del self._breakers[name]
                return True
            return False
    
    def list_breakers(self) -> Dict[str, Dict[str, Any]]:
        """List all registered circuit breakers with stats."""
        with self._lock:
            return {
                name: breaker.get_stats()
                for name, breaker in self._breakers.items()
            }
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
    
    def get_unhealthy_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers that are not closed."""
        with self._lock:
            return {
                name: breaker
                for name, breaker in self._breakers.items()
                if not breaker.is_closed
            }


# Global registry instance
_circuit_breaker_registry = CircuitBreakerRegistry()


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 3,
    timeout: float = 30.0,
    excluded_exceptions: tuple = (),
) -> Callable[[F], F]:
    """Decorator to apply circuit breaker pattern to a function.
    
    Args:
        name: Circuit breaker identifier
        failure_threshold: Failures before opening
        recovery_timeout: Seconds before trying half-open
        success_threshold: Successes needed to close from half-open
        timeout: Call timeout in seconds
        excluded_exceptions: Exceptions that don't count as failures
        
    Returns:
        Decorated function with circuit breaker protection
    """
    config = CircuitBreakerConfig(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
        success_threshold=success_threshold,
        timeout=timeout,
        excluded_exceptions=excluded_exceptions,
    )
    
    breaker = _circuit_breaker_registry.get_breaker(name, config)
    
    def decorator(func: F) -> F:
        return breaker(func)
    
    return decorator


def get_circuit_breaker(name: str) -> Optional[CircuitBreaker]:
    """Get circuit breaker by name from global registry."""
    return _circuit_breaker_registry._breakers.get(name)


def list_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """List all registered circuit breakers."""
    return _circuit_breaker_registry.list_breakers()


def reset_circuit_breakers() -> None:
    """Reset all circuit breakers."""
    _circuit_breaker_registry.reset_all()