"""Advanced caching system for industrial RL with adaptive patterns."""

import functools
import hashlib
import pickle
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union, Tuple

import numpy as np
import jax.numpy as jnp

from ..monitoring.logger import get_logger


class AdaptiveCache:
    """Adaptive LRU cache that learns access patterns and optimizes storage."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: Optional[int] = None,
        hit_ratio_threshold: float = 0.8,
        eviction_batch_size: int = 100,
        persistence_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize adaptive cache.
        
        Args:
            max_size: Maximum number of cache entries
            ttl_seconds: Time-to-live for cache entries (None for no expiry)
            hit_ratio_threshold: Minimum hit ratio to maintain
            eviction_batch_size: Number of items to evict at once
            persistence_path: Path to persist cache to disk
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hit_ratio_threshold = hit_ratio_threshold
        self.eviction_batch_size = eviction_batch_size
        self.persistence_path = Path(persistence_path) if persistence_path else None
        
        # Cache storage
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.access_times = {}
        self.access_frequencies = {}
        
        # Logger
        self.logger = get_logger("cache")
        
        # Load from persistence if available
        if self.persistence_path and self.persistence_path.exists():
            self._load_from_disk()
            
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        # Hash key for consistent storage
        hashed_key = self._hash_key(key)
        
        if hashed_key in self.cache:
            entry = self.cache[hashed_key]
            
            # Check TTL
            if self.ttl_seconds and time.time() - entry['timestamp'] > self.ttl_seconds:
                del self.cache[hashed_key]
                self.misses += 1
                return None
                
            # Update access patterns
            self._update_access_patterns(hashed_key)
            
            # Move to end (most recently used)
            self.cache.move_to_end(hashed_key)
            
            self.hits += 1
            return entry['value']
        else:
            self.misses += 1
            return None
            
    def put(self, key: str, value: Any) -> None:
        """Put value into cache."""
        hashed_key = self._hash_key(key)
        
        # Create entry
        entry = {
            'value': value,
            'timestamp': time.time(),
            'access_count': 1,
            'size': self._estimate_size(value),
        }
        
        # Check if we need to evict
        if len(self.cache) >= self.max_size:
            self._adaptive_evict()
            
        # Store entry
        self.cache[hashed_key] = entry
        self._update_access_patterns(hashed_key)
        
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.access_times.clear()
        self.access_frequencies.clear()
        
    def _hash_key(self, key: str) -> str:
        """Create consistent hash for cache key."""
        return hashlib.md5(key.encode()).hexdigest()
        
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        try:
            if isinstance(value, (np.ndarray, jnp.ndarray)):
                return value.nbytes
            else:
                return len(pickle.dumps(value))
        except Exception:
            return 1000  # Default estimate
            
    def _update_access_patterns(self, key: str) -> None:
        """Update access pattern tracking."""
        current_time = time.time()
        
        # Update frequency
        self.access_frequencies[key] = self.access_frequencies.get(key, 0) + 1
        
        # Update timing
        if key in self.access_times:
            # Calculate time since last access
            time_delta = current_time - self.access_times[key]
            # Use this for temporal locality analysis
            
        self.access_times[key] = current_time
        
    def _adaptive_evict(self) -> None:
        """Perform adaptive eviction based on access patterns."""
        if not self.cache:
            return
            
        # Calculate scores for each entry
        current_time = time.time()
        scores = {}
        
        for key, entry in self.cache.items():
            age = current_time - entry['timestamp']
            frequency = self.access_frequencies.get(key, 1)
            size = entry['size']
            
            # Combined score (lower is better for eviction)
            # Considers recency, frequency, and size
            score = (age * size) / (frequency + 1)
            scores[key] = score
            
        # Sort by score (highest scores evicted first)
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        
        # Evict batch
        evict_count = min(self.eviction_batch_size, len(sorted_keys))
        for i in range(evict_count):
            key = sorted_keys[i]
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
            if key in self.access_frequencies:
                del self.access_frequencies[key]
            self.evictions += 1
            
        self.logger.debug(f"Evicted {evict_count} entries from cache")
        
    def _save_to_disk(self) -> None:
        """Persist cache to disk."""
        if not self.persistence_path:
            return
            
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            
            cache_data = {
                'cache': dict(self.cache),
                'access_frequencies': self.access_frequencies,
                'stats': {
                    'hits': self.hits,
                    'misses': self.misses,
                    'evictions': self.evictions,
                }
            }
            
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
            self.logger.info(f"Cache persisted to {self.persistence_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to persist cache: {e}")
            
    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        try:
            with open(self.persistence_path, 'rb') as f:
                cache_data = pickle.load(f)
                
            self.cache = OrderedDict(cache_data['cache'])
            self.access_frequencies = cache_data['access_frequencies']
            
            stats = cache_data.get('stats', {})
            self.hits = stats.get('hits', 0)
            self.misses = stats.get('misses', 0)  
            self.evictions = stats.get('evictions', 0)
            
            self.logger.info(f"Cache loaded from {self.persistence_path}")
            
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_ratio = self.hits / max(total_requests, 1)
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_ratio': hit_ratio,
            'efficiency': 'GOOD' if hit_ratio >= self.hit_ratio_threshold else 'POOR',
            'memory_usage': sum(entry['size'] for entry in self.cache.values()),
        }
        
    def optimize(self) -> None:
        """Optimize cache based on access patterns."""
        stats = self.get_stats()
        
        # Adjust cache size based on hit ratio
        if stats['hit_ratio'] < self.hit_ratio_threshold and len(self.cache) < self.max_size:
            # Poor hit ratio but space available - might need better eviction
            pass
        elif stats['hit_ratio'] > 0.95 and len(self.cache) == self.max_size:
            # Excellent hit ratio at capacity - consider increasing size
            self.max_size = int(self.max_size * 1.2)
            self.logger.info(f"Increased cache size to {self.max_size}")
            
        # Persist optimized cache
        if self.persistence_path:
            self._save_to_disk()


def cached_computation(
    cache_instance: Optional[AdaptiveCache] = None,
    ttl_seconds: Optional[int] = None,
    key_func: Optional[Callable] = None,
) -> Callable:
    """Decorator for caching function results.
    
    Args:
        cache_instance: Cache instance to use
        ttl_seconds: Override TTL for this function
        key_func: Custom key generation function
        
    Returns:
        Decorated function with caching
    """
    def decorator(func: Callable) -> Callable:
        # Use global cache if none provided
        nonlocal cache_instance
        if cache_instance is None:
            cache_instance = AdaptiveCache(max_size=500, ttl_seconds=ttl_seconds)
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                
                for arg in args:
                    if isinstance(arg, (np.ndarray, jnp.ndarray)):
                        key_parts.append(f"array_{arg.shape}_{hash(arg.tobytes())}")
                    else:
                        key_parts.append(str(hash(arg)))
                        
                for k, v in sorted(kwargs.items()):
                    if isinstance(v, (np.ndarray, jnp.ndarray)):
                        key_parts.append(f"{k}_array_{v.shape}_{hash(v.tobytes())}")
                    else:
                        key_parts.append(f"{k}_{hash(v)}")
                        
                cache_key = "_".join(key_parts)
            
            # Try cache first
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
                
            # Compute and cache result
            result = func(*args, **kwargs)
            cache_instance.put(cache_key, result)
            
            return result
            
        # Add cache management methods to wrapped function
        wrapper.cache_stats = cache_instance.get_stats
        wrapper.cache_clear = cache_instance.clear
        wrapper.cache_optimize = cache_instance.optimize
        
        return wrapper
    return decorator


class NetworkWeightCache:
    """Specialized cache for neural network weights and states."""
    
    def __init__(self, max_memory_mb: int = 1000):
        """Initialize network weight cache.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.current_memory = 0
        self.logger = get_logger("weight_cache")
        
    def store_weights(self, model_id: str, weights: Dict[str, Any], metadata: Optional[Dict] = None) -> None:
        """Store network weights with metadata."""
        # Estimate memory usage
        memory_usage = 0
        for param in jax.tree_util.tree_leaves(weights):
            if hasattr(param, 'nbytes'):
                memory_usage += param.nbytes
                
        # Evict if necessary
        while self.current_memory + memory_usage > self.max_memory_bytes and self.cache:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            removed_entry = self.cache.pop(oldest_key)
            self.current_memory -= removed_entry['memory_usage']
            self.logger.debug(f"Evicted weights for {oldest_key}")
            
        # Store weights
        self.cache[model_id] = {
            'weights': weights,
            'metadata': metadata or {},
            'timestamp': time.time(),
            'memory_usage': memory_usage,
        }
        
        self.current_memory += memory_usage
        self.logger.debug(f"Cached weights for {model_id} ({memory_usage / 1024 / 1024:.1f}MB)")
        
    def get_weights(self, model_id: str) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Retrieve network weights and metadata."""
        if model_id in self.cache:
            entry = self.cache[model_id]
            return entry['weights'], entry['metadata']
        return None
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        return {
            'current_mb': self.current_memory / 1024 / 1024,
            'max_mb': self.max_memory_bytes / 1024 / 1024,
            'utilization': self.current_memory / self.max_memory_bytes,
            'cached_models': len(self.cache),
        }


# Global cache instances
_global_cache = None
_weight_cache = None


def get_global_cache() -> AdaptiveCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = AdaptiveCache(
            max_size=1000,
            ttl_seconds=3600,  # 1 hour
            persistence_path="./cache/global_cache.pkl"
        )
    return _global_cache


def get_weight_cache() -> NetworkWeightCache:
    """Get global weight cache instance."""
    global _weight_cache
    if _weight_cache is None:
        _weight_cache = NetworkWeightCache(max_memory_mb=2000)  # 2GB
    return _weight_cache