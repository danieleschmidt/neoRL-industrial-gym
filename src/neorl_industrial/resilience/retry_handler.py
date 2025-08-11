"""Retry handling with exponential backoff and jitter."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, List, Optional, Type, TypeVar, Union
import logging

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class RetryPolicy:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0             # Base delay in seconds
    max_delay: float = 60.0             # Maximum delay in seconds
    exponential_base: float = 2.0       # Exponential backoff multiplier
    jitter: bool = True                 # Add random jitter
    retriable_exceptions: tuple = (Exception,)  # Exceptions to retry on
    non_retriable_exceptions: tuple = ()        # Exceptions to never retry


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""
    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Retry exhausted after {attempts} attempts. "
            f"Last error: {last_exception}"
        )


class RetryHandler:
    """Advanced retry handler with exponential backoff."""
    
    def __init__(self, policy: Optional[RetryPolicy] = None) -> None:
        """Initialize retry handler.
        
        Args:
            policy: Retry policy configuration
        """
        self.policy = policy or RetryPolicy()
        self._attempt_history: List[dict] = []
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should trigger a retry.
        
        Args:
            exception: Exception that occurred
            attempt: Current attempt number (1-based)
            
        Returns:
            True if should retry, False otherwise
        """
        # Check if we've exceeded max attempts
        if attempt >= self.policy.max_attempts:
            return False
        
        # Check non-retriable exceptions first
        if isinstance(exception, self.policy.non_retriable_exceptions):
            return False
        
        # Check if exception is retriable
        return isinstance(exception, self.policy.retriable_exceptions)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry attempt.
        
        Args:
            attempt: Current attempt number (1-based)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = self.policy.base_delay * (
            self.policy.exponential_base ** (attempt - 1)
        )
        
        # Cap at maximum delay
        delay = min(delay, self.policy.max_delay)
        
        # Add jitter to avoid thundering herd
        if self.policy.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay)  # Ensure minimum delay
        
        return delay
    
    def execute(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function positional arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            RetryExhaustedError: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(1, self.policy.max_attempts + 1):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                
                # Record successful attempt
                self._attempt_history.append({
                    "attempt": attempt,
                    "success": True,
                    "duration": time.time() - start_time,
                    "timestamp": start_time,
                })
                
                return result
                
            except Exception as e:
                last_exception = e
                execution_time = time.time() - start_time
                
                # Record failed attempt
                self._attempt_history.append({
                    "attempt": attempt,
                    "success": False,
                    "exception": str(e),
                    "exception_type": type(e).__name__,
                    "duration": execution_time,
                    "timestamp": start_time,
                })
                
                # Check if we should retry
                if not self.should_retry(e, attempt):
                    logger.info(
                        f"Not retrying {type(e).__name__} on attempt {attempt}"
                    )
                    raise e
                
                # Log retry attempt
                if attempt < self.policy.max_attempts:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt} failed with {type(e).__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {self.policy.max_attempts} retry attempts failed"
                    )
        
        # All attempts exhausted
        raise RetryExhaustedError(self.policy.max_attempts, last_exception)
    
    def get_attempt_history(self) -> List[dict]:
        """Get history of retry attempts."""
        return self._attempt_history.copy()
    
    def clear_history(self) -> None:
        """Clear attempt history."""
        self._attempt_history.clear()
    
    def __call__(self, func: F) -> F:
        """Decorator interface for retry handler."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute(func, *args, **kwargs)
        
        return wrapper  # type: ignore


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retriable_exceptions: tuple = (Exception,),
    non_retriable_exceptions: tuple = (),
) -> Callable[[F], F]:
    """Decorator for retry with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Multiplier for exponential backoff
        jitter: Whether to add random jitter to delays
        retriable_exceptions: Exceptions that should trigger retries
        non_retriable_exceptions: Exceptions that should never be retried
        
    Returns:
        Decorated function with retry logic
    """
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        retriable_exceptions=retriable_exceptions,
        non_retriable_exceptions=non_retriable_exceptions,
    )
    
    handler = RetryHandler(policy)
    
    def decorator(func: F) -> F:
        return handler(func)
    
    return decorator


class ConditionalRetry:
    """Conditional retry based on return value or exception."""
    
    def __init__(
        self,
        should_retry_func: Callable[[Any], bool],
        policy: Optional[RetryPolicy] = None,
    ) -> None:
        """Initialize conditional retry.
        
        Args:
            should_retry_func: Function to determine if result should trigger retry
            policy: Retry policy configuration
        """
        self.should_retry_func = should_retry_func
        self.policy = policy or RetryPolicy()
        self._attempt_history: List[dict] = []
    
    def execute(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Execute function with conditional retry logic."""
        last_result = None
        last_exception = None
        
        for attempt in range(1, self.policy.max_attempts + 1):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                
                # Check if result should trigger retry
                if not self.should_retry_func(result):
                    # Success - don't retry
                    self._attempt_history.append({
                        "attempt": attempt,
                        "success": True,
                        "result": result,
                        "duration": time.time() - start_time,
                        "timestamp": start_time,
                    })
                    return result
                else:
                    # Result indicates retry needed
                    last_result = result
                    self._attempt_history.append({
                        "attempt": attempt,
                        "success": False,
                        "result": result,
                        "reason": "conditional_retry",
                        "duration": time.time() - start_time,
                        "timestamp": start_time,
                    })
                    
                    if attempt < self.policy.max_attempts:
                        delay = self._calculate_delay(attempt)
                        logger.warning(
                            f"Conditional retry triggered on attempt {attempt}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                
            except Exception as e:
                last_exception = e
                execution_time = time.time() - start_time
                
                self._attempt_history.append({
                    "attempt": attempt,
                    "success": False,
                    "exception": str(e),
                    "exception_type": type(e).__name__,
                    "duration": execution_time,
                    "timestamp": start_time,
                })
                
                # Standard exception retry logic
                if not isinstance(e, self.policy.retriable_exceptions):
                    raise e
                
                if attempt < self.policy.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt} failed with {type(e).__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )
                    time.sleep(delay)
        
        # All attempts exhausted
        if last_exception:
            raise RetryExhaustedError(self.policy.max_attempts, last_exception)
        else:
            # Return last result even if conditional retry wanted to continue
            return last_result
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        delay = self.policy.base_delay * (
            self.policy.exponential_base ** (attempt - 1)
        )
        delay = min(delay, self.policy.max_delay)
        
        if self.policy.jitter:
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0.1, delay)
        
        return delay


def retry_until(
    condition: Callable[[Any], bool],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
) -> Callable[[F], F]:
    """Decorator to retry until a condition is met.
    
    Args:
        condition: Function that returns True when result is acceptable
        max_attempts: Maximum retry attempts
        base_delay: Base delay between retries
        max_delay: Maximum delay between retries
        
    Returns:
        Decorated function with conditional retry logic
    """
    def should_not_retry(result):
        return condition(result)
    
    policy = RetryPolicy(
        max_attempts=max_attempts,
        base_delay=base_delay,
        max_delay=max_delay,
    )
    
    conditional_retry = ConditionalRetry(
        should_retry_func=lambda x: not should_not_retry(x),
        policy=policy,
    )
    
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return conditional_retry.execute(func, *args, **kwargs)
        return wrapper  # type: ignore
    
    return decorator


class RetryStats:
    """Statistics tracking for retry operations."""
    
    def __init__(self) -> None:
        self.total_attempts = 0
        self.successful_first_attempts = 0
        self.retry_successes = 0
        self.total_failures = 0
        self.exception_counts: dict = {}
        
    def record_attempt(
        self,
        attempt_num: int,
        success: bool,
        exception_type: Optional[str] = None
    ) -> None:
        """Record a retry attempt."""
        self.total_attempts += 1
        
        if success:
            if attempt_num == 1:
                self.successful_first_attempts += 1
            else:
                self.retry_successes += 1
        else:
            if exception_type:
                self.exception_counts[exception_type] = (
                    self.exception_counts.get(exception_type, 0) + 1
                )
            if attempt_num >= 3:  # Assuming max 3 attempts
                self.total_failures += 1
    
    def get_stats(self) -> dict:
        """Get retry statistics."""
        total_operations = (
            self.successful_first_attempts + 
            self.retry_successes + 
            self.total_failures
        )
        
        if total_operations == 0:
            return {"message": "No retry operations recorded"}
        
        return {
            "total_operations": total_operations,
            "successful_first_attempts": self.successful_first_attempts,
            "retry_successes": self.retry_successes,
            "total_failures": self.total_failures,
            "first_attempt_success_rate": (
                self.successful_first_attempts / total_operations * 100
            ),
            "eventual_success_rate": (
                (self.successful_first_attempts + self.retry_successes) / 
                total_operations * 100
            ),
            "exception_breakdown": self.exception_counts,
        }
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.total_attempts = 0
        self.successful_first_attempts = 0
        self.retry_successes = 0
        self.total_failures = 0
        self.exception_counts.clear()


# Global retry statistics instance
retry_stats = RetryStats()