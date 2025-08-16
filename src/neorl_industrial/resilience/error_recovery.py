"""Error recovery and resilience mechanisms for industrial RL systems."""

import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

try:
    import jax.numpy as jnp
    import numpy as np
except ImportError:
    import numpy as np
    jnp = np

from ..core.types import Array
from ..exceptions import RecoveryError


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class ErrorRecord:
    """Record of an error and recovery attempt."""
    
    timestamp: datetime
    error_type: str
    error_message: str
    severity: ErrorSeverity
    component: str
    recovery_strategy: RecoveryStrategy
    recovery_successful: bool
    recovery_time_ms: float
    context: Dict[str, Any]


@dataclass
class RecoveryConfig:
    """Configuration for error recovery."""
    
    max_retries: int = 3
    retry_delay_ms: float = 100
    exponential_backoff: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_s: float = 60
    fallback_enabled: bool = True
    emergency_stop_threshold: int = 10


class RecoveryHandler(ABC):
    """Abstract base class for error recovery handlers."""
    
    def __init__(self, name: str, config: RecoveryConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"RecoveryHandler.{name}")
        
        # Error tracking
        self.error_history = deque(maxlen=1000)
        self.consecutive_errors = 0
        self.last_error_time = None
        
    @abstractmethod
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if this handler can recover from the error."""
        pass
        
    @abstractmethod
    def recover(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Attempt to recover from the error."""
        pass
        
    def record_error(self, error: Exception, context: Dict[str, Any], 
                    recovery_successful: bool, recovery_time_ms: float):
        """Record error and recovery attempt."""
        severity = self._determine_severity(error, context)
        strategy = self._determine_strategy(error, context)
        
        record = ErrorRecord(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            component=context.get("component", "unknown"),
            recovery_strategy=strategy,
            recovery_successful=recovery_successful,
            recovery_time_ms=recovery_time_ms,
            context=context
        )
        
        self.error_history.append(record)
        
        if recovery_successful:
            self.consecutive_errors = 0
        else:
            self.consecutive_errors += 1
            
        self.last_error_time = datetime.now()
        
    def _determine_severity(self, error: Exception, context: Dict[str, Any]) -> ErrorSeverity:
        """Determine error severity based on error type and context."""
        # Critical errors that affect safety
        critical_errors = (RuntimeError, MemoryError, KeyboardInterrupt)
        if isinstance(error, critical_errors):
            return ErrorSeverity.CRITICAL
            
        # High severity errors
        high_errors = (ValueError, TypeError, AttributeError)
        if isinstance(error, high_errors):
            return ErrorSeverity.HIGH
            
        # Check for safety-related context
        if context.get("safety_critical", False):
            return ErrorSeverity.CRITICAL
            
        return ErrorSeverity.MEDIUM
        
    def _determine_strategy(self, error: Exception, context: Dict[str, Any]) -> RecoveryStrategy:
        """Determine recovery strategy based on error and context."""
        severity = self._determine_severity(error, context)
        
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.EMERGENCY_STOP
        elif self.consecutive_errors >= self.config.circuit_breaker_threshold:
            return RecoveryStrategy.CIRCUIT_BREAKER
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return RecoveryStrategy.RETRY
        else:
            return RecoveryStrategy.FALLBACK


class RetryHandler(RecoveryHandler):
    """Handler for retry-based error recovery."""
    
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if error is retryable."""
        retryable_errors = (
            ConnectionError,
            TimeoutError,
            FileNotFoundError,
            PermissionError
        )
        return isinstance(error, retryable_errors) and self.consecutive_errors < self.config.max_retries
        
    def recover(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Attempt recovery through retries with exponential backoff."""
        start_time = time.time()
        
        for attempt in range(self.config.max_retries):
            try:
                # Calculate delay
                if self.config.exponential_backoff:
                    delay = self.config.retry_delay_ms * (2 ** attempt) / 1000
                else:
                    delay = self.config.retry_delay_ms / 1000
                    
                if attempt > 0:
                    self.logger.info(f"Retry attempt {attempt + 1}/{self.config.max_retries} after {delay:.2f}s")
                    time.sleep(delay)
                    
                # Retry the failed operation
                operation = context.get("operation")
                args = context.get("args", ())
                kwargs = context.get("kwargs", {})
                
                if operation:
                    result = operation(*args, **kwargs)
                    recovery_time = (time.time() - start_time) * 1000
                    self.record_error(error, context, True, recovery_time)
                    return True, result
                    
            except Exception as retry_error:
                self.logger.warning(f"Retry {attempt + 1} failed: {retry_error}")
                if attempt == self.config.max_retries - 1:
                    # Final attempt failed
                    recovery_time = (time.time() - start_time) * 1000
                    self.record_error(error, context, False, recovery_time)
                    return False, None
                    
        return False, None


class FallbackHandler(RecoveryHandler):
    """Handler for fallback-based error recovery."""
    
    def __init__(self, name: str, config: RecoveryConfig, fallback_functions: Dict[str, Callable]):
        super().__init__(name, config)
        self.fallback_functions = fallback_functions
        
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if fallback is available."""
        component = context.get("component")
        return (self.config.fallback_enabled and 
                component in self.fallback_functions)
        
    def recover(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Attempt recovery using fallback function."""
        start_time = time.time()
        
        try:
            component = context.get("component")
            fallback_fn = self.fallback_functions[component]
            
            # Extract arguments for fallback
            args = context.get("args", ())
            kwargs = context.get("kwargs", {})
            
            self.logger.info(f"Using fallback for component: {component}")
            result = fallback_fn(*args, **kwargs)
            
            recovery_time = (time.time() - start_time) * 1000
            self.record_error(error, context, True, recovery_time)
            
            return True, result
            
        except Exception as fallback_error:
            self.logger.error(f"Fallback failed: {fallback_error}")
            recovery_time = (time.time() - start_time) * 1000
            self.record_error(error, context, False, recovery_time)
            return False, None


class CircuitBreakerHandler(RecoveryHandler):
    """Handler implementing circuit breaker pattern."""
    
    def __init__(self, name: str, config: RecoveryConfig):
        super().__init__(name, config)
        self.circuit_open = False
        self.circuit_opened_time = None
        self.half_open_attempts = 0
        
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if circuit breaker should handle this error."""
        return self.consecutive_errors >= self.config.circuit_breaker_threshold
        
    def recover(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Handle error using circuit breaker pattern."""
        start_time = time.time()
        
        current_time = datetime.now()
        
        # Check if circuit should be closed
        if (self.circuit_open and self.circuit_opened_time and
            current_time - self.circuit_opened_time > timedelta(seconds=self.config.circuit_breaker_timeout_s)):
            self.logger.info("Circuit breaker entering half-open state")
            self.circuit_open = False
            self.half_open_attempts = 0
            
        # If circuit is open, fail immediately
        if self.circuit_open:
            self.logger.warning("Circuit breaker is open, failing immediately")
            recovery_time = (time.time() - start_time) * 1000
            self.record_error(error, context, False, recovery_time)
            return False, None
            
        # Try operation in half-open state
        try:
            operation = context.get("operation")
            args = context.get("args", ())
            kwargs = context.get("kwargs", {})
            
            if operation:
                result = operation(*args, **kwargs)
                
                # Success - close circuit
                self.consecutive_errors = 0
                self.half_open_attempts = 0
                
                recovery_time = (time.time() - start_time) * 1000
                self.record_error(error, context, True, recovery_time)
                
                self.logger.info("Circuit breaker closed after successful operation")
                return True, result
                
        except Exception as circuit_error:
            self.half_open_attempts += 1
            
            # Open circuit if half-open attempts fail
            if self.half_open_attempts >= 3:
                self.circuit_open = True
                self.circuit_opened_time = current_time
                self.logger.error("Circuit breaker opened due to continued failures")
                
            recovery_time = (time.time() - start_time) * 1000
            self.record_error(error, context, False, recovery_time)
            return False, None
            
        return False, None


class GracefulDegradationHandler(RecoveryHandler):
    """Handler for graceful degradation recovery."""
    
    def __init__(self, name: str, config: RecoveryConfig, degraded_functions: Dict[str, Callable]):
        super().__init__(name, config)
        self.degraded_functions = degraded_functions
        self.degradation_active = set()
        
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if graceful degradation is available."""
        component = context.get("component")
        return component in self.degraded_functions
        
    def recover(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Recover using graceful degradation."""
        start_time = time.time()
        
        try:
            component = context.get("component")
            degraded_fn = self.degraded_functions[component]
            
            # Activate degradation mode
            self.degradation_active.add(component)
            
            args = context.get("args", ())
            kwargs = context.get("kwargs", {})
            
            self.logger.warning(f"Activating degraded mode for component: {component}")
            result = degraded_fn(*args, **kwargs)
            
            recovery_time = (time.time() - start_time) * 1000
            self.record_error(error, context, True, recovery_time)
            
            return True, result
            
        except Exception as degradation_error:
            self.logger.error(f"Graceful degradation failed: {degradation_error}")
            recovery_time = (time.time() - start_time) * 1000
            self.record_error(error, context, False, recovery_time)
            return False, None
            
    def restore_component(self, component: str):
        """Restore component from degraded mode."""
        if component in self.degradation_active:
            self.degradation_active.remove(component)
            self.logger.info(f"Restored component from degraded mode: {component}")


class EmergencyStopHandler(RecoveryHandler):
    """Handler for emergency stop procedures."""
    
    def __init__(self, name: str, config: RecoveryConfig, emergency_procedures: Dict[str, Callable]):
        super().__init__(name, config)
        self.emergency_procedures = emergency_procedures
        self.emergency_active = False
        
    def can_handle(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if emergency stop is required."""
        severity = self._determine_severity(error, context)
        total_recent_errors = len([r for r in self.error_history 
                                 if r.timestamp > datetime.now() - timedelta(minutes=1)])
        
        return (severity == ErrorSeverity.CRITICAL or 
                total_recent_errors >= self.config.emergency_stop_threshold)
        
    def recover(self, error: Exception, context: Dict[str, Any]) -> Tuple[bool, Any]:
        """Execute emergency stop procedures."""
        start_time = time.time()
        
        try:
            self.emergency_active = True
            
            # Execute emergency procedures
            for procedure_name, procedure in self.emergency_procedures.items():
                try:
                    self.logger.critical(f"Executing emergency procedure: {procedure_name}")
                    procedure(error, context)
                except Exception as proc_error:
                    self.logger.error(f"Emergency procedure {procedure_name} failed: {proc_error}")
                    
            recovery_time = (time.time() - start_time) * 1000
            self.record_error(error, context, True, recovery_time)
            
            # Emergency stop doesn't recover - it stops safely
            return False, None
            
        except Exception as emergency_error:
            self.logger.critical(f"Emergency stop failed: {emergency_error}")
            recovery_time = (time.time() - start_time) * 1000
            self.record_error(error, context, False, recovery_time)
            return False, None


class ErrorRecoveryManager:
    """Central manager for error recovery in industrial RL systems."""
    
    def __init__(self, config: Optional[RecoveryConfig] = None):
        self.config = config or RecoveryConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize recovery handlers
        self.handlers: List[RecoveryHandler] = []
        
        # Default fallback functions
        self.default_fallbacks = {
            "agent_prediction": self._safe_action_fallback,
            "environment_step": self._safe_environment_fallback,
            "data_loading": self._cached_data_fallback
        }
        
        # Default emergency procedures
        self.emergency_procedures = {
            "log_error": self._log_emergency,
            "notify_operators": self._notify_operators,
            "safe_shutdown": self._safe_shutdown
        }
        
        self._initialize_handlers()
        
    def _initialize_handlers(self):
        """Initialize default recovery handlers."""
        self.handlers = [
            RetryHandler("retry", self.config),
            FallbackHandler("fallback", self.config, self.default_fallbacks),
            CircuitBreakerHandler("circuit_breaker", self.config),
            GracefulDegradationHandler("degradation", self.config, {}),
            EmergencyStopHandler("emergency", self.config, self.emergency_procedures)
        ]
        
    def add_fallback(self, component: str, fallback_fn: Callable):
        """Add fallback function for component."""
        self.default_fallbacks[component] = fallback_fn
        
        # Update fallback handler
        for handler in self.handlers:
            if isinstance(handler, FallbackHandler):
                handler.fallback_functions[component] = fallback_fn
                
    def add_degraded_function(self, component: str, degraded_fn: Callable):
        """Add degraded function for component."""
        for handler in self.handlers:
            if isinstance(handler, GracefulDegradationHandler):
                handler.degraded_functions[component] = degraded_fn
                
    def add_emergency_procedure(self, name: str, procedure: Callable):
        """Add emergency procedure."""
        self.emergency_procedures[name] = procedure
        
        # Update emergency handler
        for handler in self.handlers:
            if isinstance(handler, EmergencyStopHandler):
                handler.emergency_procedures[name] = procedure
                
    def recover_from_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any]:
        """Attempt to recover from error using available handlers."""
        if context is None:
            context = {}
            
        self.logger.error(f"Attempting recovery from error: {type(error).__name__}: {error}")
        
        # Try handlers in order of preference
        for handler in self.handlers:
            if handler.can_handle(error, context):
                self.logger.info(f"Using recovery handler: {handler.name}")
                
                try:
                    success, result = handler.recover(error, context)
                    
                    if success:
                        self.logger.info(f"Recovery successful using {handler.name}")
                        return True, result
                    else:
                        self.logger.warning(f"Recovery failed using {handler.name}")
                        
                except Exception as recovery_error:
                    self.logger.error(f"Recovery handler {handler.name} threw error: {recovery_error}")
                    
        # No handler could recover
        self.logger.critical(f"All recovery attempts failed for error: {error}")
        return False, None
        
    def _safe_action_fallback(self, *args, **kwargs) -> Array:
        """Safe fallback action (zero action)."""
        action_dim = kwargs.get("action_dim", 6)  # Default action dimension
        return jnp.zeros(action_dim)
        
    def _safe_environment_fallback(self, *args, **kwargs) -> Tuple[Array, float, bool, bool, Dict]:
        """Safe fallback for environment step."""
        state_dim = kwargs.get("state_dim", 20)  # Default state dimension
        safe_obs = jnp.zeros(state_dim)
        return safe_obs, 0.0, True, False, {"fallback_used": True}
        
    def _cached_data_fallback(self, *args, **kwargs) -> Any:
        """Fallback to cached data."""
        # Return empty dataset structure
        return {
            "observations": jnp.array([]),
            "actions": jnp.array([]),
            "rewards": jnp.array([]),
            "next_observations": jnp.array([]),
            "dones": jnp.array([])
        }
        
    def _log_emergency(self, error: Exception, context: Dict[str, Any]):
        """Log emergency error details."""
        self.logger.critical(f"EMERGENCY ERROR: {type(error).__name__}: {error}")
        self.logger.critical(f"Context: {context}")
        
    def _notify_operators(self, error: Exception, context: Dict[str, Any]):
        """Notify human operators of emergency."""
        # In a real system, this would send alerts, emails, etc.
        self.logger.critical("EMERGENCY: Human operator notification required")
        
    def _safe_shutdown(self, error: Exception, context: Dict[str, Any]):
        """Perform safe shutdown procedures."""
        self.logger.critical("Initiating safe shutdown procedures")
        # In a real system, this would safely stop all operations
        
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics."""
        all_errors = []
        for handler in self.handlers:
            all_errors.extend(handler.error_history)
            
        total_errors = len(all_errors)
        successful_recoveries = sum(1 for error in all_errors if error.recovery_successful)
        
        recovery_rate = successful_recoveries / max(total_errors, 1)
        
        # Statistics by handler
        handler_stats = {}
        for handler in self.handlers:
            handler_errors = list(handler.error_history)
            handler_successes = sum(1 for error in handler_errors if error.recovery_successful)
            
            handler_stats[handler.name] = {
                "total_errors": len(handler_errors),
                "successful_recoveries": handler_successes,
                "recovery_rate": handler_successes / max(len(handler_errors), 1),
                "consecutive_errors": handler.consecutive_errors
            }
            
        # Recent error trends
        recent_errors = [e for e in all_errors 
                        if e.timestamp > datetime.now() - timedelta(hours=1)]
        
        return {
            "overall": {
                "total_errors": total_errors,
                "successful_recoveries": successful_recoveries,
                "recovery_rate": recovery_rate,
                "recent_errors_1h": len(recent_errors)
            },
            "by_handler": handler_stats,
            "active_emergencies": any(
                isinstance(h, EmergencyStopHandler) and h.emergency_active 
                for h in self.handlers
            ),
            "circuit_breakers_open": [
                h.name for h in self.handlers 
                if isinstance(h, CircuitBreakerHandler) and h.circuit_open
            ]
        }