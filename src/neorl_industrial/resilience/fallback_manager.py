"""Fallback strategies for graceful degradation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class FallbackStrategy(Enum):
    """Types of fallback strategies."""
    DEFAULT_VALUE = "default_value"
    CACHED_VALUE = "cached_value"
    ALTERNATIVE_FUNCTION = "alternative_function"
    CIRCUIT_BREAKER = "circuit_breaker"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    strategy: FallbackStrategy
    fallback_value: Any = None
    cache_ttl: float = 300.0  # Cache TTL in seconds
    alternative_func: Optional[Callable] = None
    degradation_mode: Optional[str] = None
    trigger_exceptions: tuple = (Exception,)
    log_fallbacks: bool = True


class FallbackProvider(ABC):
    """Abstract base class for fallback providers."""
    
    @abstractmethod
    def get_fallback(self, *args, **kwargs) -> Any:
        """Get fallback value or result."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if fallback is available."""
        pass


class DefaultValueProvider(FallbackProvider):
    """Provides default values as fallback."""
    
    def __init__(self, default_value: Any):
        self.default_value = default_value
    
    def get_fallback(self, *args, **kwargs) -> Any:
        """Return the default value."""
        return self.default_value
    
    def is_available(self) -> bool:
        """Default values are always available."""
        return True


class CachedValueProvider(FallbackProvider):
    """Provides cached values as fallback."""
    
    def __init__(self, cache_ttl: float = 300.0):
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def cache_result(self, key: str, result: Any) -> None:
        """Cache a result for fallback use."""
        import time
        self._cache[key] = {
            "value": result,
            "timestamp": time.time(),
        }
    
    def get_fallback(self, cache_key: str, *args, **kwargs) -> Any:
        """Get cached value if available and not expired."""
        import time
        
        if cache_key not in self._cache:
            raise ValueError(f"No cached value for key: {cache_key}")
        
        cached_data = self._cache[cache_key]
        age = time.time() - cached_data["timestamp"]
        
        if age > self.cache_ttl:
            raise ValueError(f"Cached value expired for key: {cache_key}")
        
        logger.info(f"Using cached fallback for key: {cache_key} (age: {age:.1f}s)")
        return cached_data["value"]
    
    def is_available(self) -> bool:
        """Check if any cached values are available."""
        import time
        current_time = time.time()
        
        return any(
            current_time - data["timestamp"] <= self.cache_ttl
            for data in self._cache.values()
        )
    
    def clear_cache(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        import time
        current_time = time.time()
        
        expired_count = sum(
            1 for data in self._cache.values()
            if current_time - data["timestamp"] > self.cache_ttl
        )
        
        return {
            "total_entries": len(self._cache),
            "expired_entries": expired_count,
            "valid_entries": len(self._cache) - expired_count,
            "cache_ttl": self.cache_ttl,
        }


class AlternativeFunctionProvider(FallbackProvider):
    """Provides alternative function as fallback."""
    
    def __init__(self, alternative_func: Callable):
        self.alternative_func = alternative_func
    
    def get_fallback(self, *args, **kwargs) -> Any:
        """Execute alternative function."""
        logger.info(f"Using alternative function fallback: {self.alternative_func.__name__}")
        return self.alternative_func(*args, **kwargs)
    
    def is_available(self) -> bool:
        """Alternative function is available if callable."""
        return callable(self.alternative_func)


class DegradedModeProvider(FallbackProvider):
    """Provides degraded functionality as fallback."""
    
    def __init__(self, degradation_modes: Dict[str, Callable]):
        self.degradation_modes = degradation_modes
    
    def get_fallback(self, mode: str, *args, **kwargs) -> Any:
        """Execute degraded mode function."""
        if mode not in self.degradation_modes:
            raise ValueError(f"Unknown degradation mode: {mode}")
        
        logger.warning(f"Operating in degraded mode: {mode}")
        return self.degradation_modes[mode](*args, **kwargs)
    
    def is_available(self) -> bool:
        """Degraded mode is available if modes are configured."""
        return bool(self.degradation_modes)


class FallbackManager:
    """Manages multiple fallback strategies in priority order."""
    
    def __init__(self):
        self._fallback_chain: List[tuple[FallbackProvider, dict]] = []
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "fallback_calls": 0,
            "fallback_by_provider": {},
        }
    
    def add_fallback(
        self,
        provider: FallbackProvider,
        config: Optional[dict] = None
    ) -> None:
        """Add fallback provider to the chain.
        
        Args:
            provider: Fallback provider instance
            config: Optional configuration for this provider
        """
        self._fallback_chain.append((provider, config or {}))
    
    def execute_with_fallback(
        self,
        func: Callable[..., Any],
        *args,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Execute function with fallback chain.
        
        Args:
            func: Primary function to execute
            *args: Function arguments
            cache_key: Optional cache key for caching providers
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
        """
        self._stats["total_calls"] += 1
        
        # Try primary function first
        try:
            result = func(*args, **kwargs)
            
            # Cache successful result if we have caching providers
            if cache_key:
                self._cache_result(cache_key, result)
            
            self._stats["successful_calls"] += 1
            return result
            
        except Exception as e:
            logger.warning(f"Primary function failed: {e}. Trying fallbacks...")
            
            # Try each fallback in order
            for provider, config in self._fallback_chain:
                try:
                    if not provider.is_available():
                        continue
                    
                    # Get fallback result based on provider type
                    if isinstance(provider, CachedValueProvider) and cache_key:
                        result = provider.get_fallback(cache_key, *args, **kwargs)
                    elif isinstance(provider, DegradedModeProvider):
                        mode = config.get("mode", "default")
                        result = provider.get_fallback(mode, *args, **kwargs)
                    else:
                        result = provider.get_fallback(*args, **kwargs)
                    
                    # Track fallback usage
                    provider_name = type(provider).__name__
                    self._stats["fallback_calls"] += 1
                    self._stats["fallback_by_provider"][provider_name] = (
                        self._stats["fallback_by_provider"].get(provider_name, 0) + 1
                    )
                    
                    logger.info(f"Fallback successful using {provider_name}")
                    return result
                    
                except Exception as fallback_error:
                    logger.warning(
                        f"Fallback {type(provider).__name__} failed: {fallback_error}"
                    )
                    continue
            
            # All fallbacks failed
            logger.error("All fallback strategies failed")
            raise e
    
    def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache result in caching providers."""
        for provider, _ in self._fallback_chain:
            if isinstance(provider, CachedValueProvider):
                provider.cache_result(cache_key, result)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get fallback usage statistics."""
        total_calls = self._stats["total_calls"]
        if total_calls == 0:
            return {"message": "No fallback calls recorded"}
        
        return {
            **self._stats,
            "success_rate": (self._stats["successful_calls"] / total_calls * 100),
            "fallback_rate": (self._stats["fallback_calls"] / total_calls * 100),
        }
    
    def reset_stats(self) -> None:
        """Reset fallback statistics."""
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "fallback_calls": 0,
            "fallback_by_provider": {},
        }
    
    def clear_caches(self) -> None:
        """Clear all cached values in caching providers."""
        for provider, _ in self._fallback_chain:
            if isinstance(provider, CachedValueProvider):
                provider.clear_cache()
    
    def __call__(self, func: F) -> F:
        """Decorator interface for fallback manager."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self.execute_with_fallback(func, *args, **kwargs)
        
        return wrapper  # type: ignore


def create_fallback_chain(
    default_value: Any = None,
    cache_ttl: float = 300.0,
    alternative_func: Optional[Callable] = None,
    degradation_modes: Optional[Dict[str, Callable]] = None,
) -> FallbackManager:
    """Create a fallback chain with common providers.
    
    Args:
        default_value: Default value to return as last fallback
        cache_ttl: Cache TTL for cached value provider
        alternative_func: Alternative function to use as fallback
        degradation_modes: Dictionary of degraded mode functions
        
    Returns:
        Configured fallback manager
    """
    manager = FallbackManager()
    
    # Add cached value provider first (fastest)
    if cache_ttl > 0:
        manager.add_fallback(CachedValueProvider(cache_ttl))
    
    # Add alternative function fallback
    if alternative_func:
        manager.add_fallback(AlternativeFunctionProvider(alternative_func))
    
    # Add degraded mode provider
    if degradation_modes:
        manager.add_fallback(DegradedModeProvider(degradation_modes))
    
    # Add default value as last resort
    if default_value is not None:
        manager.add_fallback(DefaultValueProvider(default_value))
    
    return manager


def fallback_decorator(
    default_value: Any = None,
    cache_ttl: float = 300.0,
    alternative_func: Optional[Callable] = None,
    degradation_modes: Optional[Dict[str, Callable]] = None,
    cache_key_func: Optional[Callable[..., str]] = None,
) -> Callable[[F], F]:
    """Decorator to add fallback behavior to a function.
    
    Args:
        default_value: Default value to return as fallback
        cache_ttl: Cache TTL for caching successful results
        alternative_func: Alternative function to use as fallback
        degradation_modes: Degraded mode functions
        cache_key_func: Function to generate cache keys from args/kwargs
        
    Returns:
        Decorated function with fallback behavior
    """
    manager = create_fallback_chain(
        default_value=default_value,
        cache_ttl=cache_ttl,
        alternative_func=alternative_func,
        degradation_modes=degradation_modes,
    )
    
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key if function provided
            cache_key = None
            if cache_key_func:
                try:
                    cache_key = cache_key_func(*args, **kwargs)
                except Exception:
                    cache_key = f"{func.__name__}_{hash((args, tuple(kwargs.items())))}"
            
            return manager.execute_with_fallback(func, *args, cache_key=cache_key, **kwargs)
        
        # Expose manager for stats and control
        wrapper._fallback_manager = manager  # type: ignore
        return wrapper  # type: ignore
    
    return decorator


class SafetyFallbackManager(FallbackManager):
    """Specialized fallback manager for safety-critical operations."""
    
    def __init__(self):
        super().__init__()
        self._safety_mode = False
        self._emergency_fallback: Optional[Callable] = None
    
    def set_emergency_fallback(self, emergency_func: Callable) -> None:
        """Set emergency fallback for safety-critical failures."""
        self._emergency_fallback = emergency_func
    
    def enter_safety_mode(self) -> None:
        """Enter safety mode - only use most conservative fallbacks."""
        self._safety_mode = True
        logger.critical("Entering safety mode - using conservative fallbacks only")
    
    def exit_safety_mode(self) -> None:
        """Exit safety mode."""
        self._safety_mode = False
        logger.info("Exiting safety mode")
    
    def execute_with_fallback(
        self,
        func: Callable[..., Any],
        *args,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Execute with safety-aware fallback logic."""
        try:
            return super().execute_with_fallback(func, *args, cache_key=cache_key, **kwargs)
        except Exception as e:
            # If in safety mode and we have emergency fallback, use it
            if self._safety_mode and self._emergency_fallback:
                logger.critical(f"Using emergency fallback due to: {e}")
                return self._emergency_fallback(*args, **kwargs)
            raise


# Global safety fallback manager for critical systems
safety_fallback_manager = SafetyFallbackManager()