"""Adaptive caching system for high-performance industrial RL."""

import hashlib
import logging
import pickle
import time
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

try:
    import jax.numpy as jnp
    import numpy as np
except ImportError:
    import numpy as np
    jnp = np

from ..core.types import Array


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # ML-based adaptive policy
    SIZE_AWARE = "size_aware"  # Size-aware LRU


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"      # Fast in-memory cache
    L2_COMPRESSED = "l2_compressed"  # Compressed in-memory cache
    L3_DISK = "l3_disk"          # Disk-based cache
    L4_DISTRIBUTED = "l4_distributed"  # Distributed cache


@dataclass
class CacheConfig:
    """Configuration for adaptive caching system."""
    
    max_memory_mb: int = 1024
    max_disk_mb: int = 10240
    default_ttl_seconds: int = 3600
    compression_enabled: bool = True
    persistent_cache: bool = True
    cache_directory: str = "/tmp/neorl_cache"
    eviction_policy: CachePolicy = CachePolicy.ADAPTIVE
    prefetch_enabled: bool = True
    stats_collection: bool = True


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    
    key: str
    value: Any
    size_bytes: int
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int] = None
    compressed: bool = False
    level: CacheLevel = CacheLevel.L1_MEMORY


@dataclass
class CacheStats:
    """Cache performance statistics."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    total_size_bytes: int = 0
    avg_access_time_ms: float = 0.0
    hit_rate: float = 0.0
    memory_utilization: float = 0.0


class CacheStorage(ABC):
    """Abstract base class for cache storage backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve cache entry by key."""
        pass
        
    @abstractmethod
    def put(self, entry: CacheEntry) -> bool:
        """Store cache entry."""
        pass
        
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cache entry."""
        pass
        
    @abstractmethod
    def size(self) -> int:
        """Get current cache size in bytes."""
        pass
        
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass


class MemoryStorage(CacheStorage):
    """In-memory cache storage with thread safety."""
    
    def __init__(self, max_size_bytes: int):
        self.max_size_bytes = max_size_bytes
        self.storage: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
        self.current_size = 0
        
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry and update access time."""
        with self.lock:
            if key in self.storage:
                entry = self.storage[key]
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                # Move to end (most recently used)
                self.storage.move_to_end(key)
                return entry
            return None
            
    def put(self, entry: CacheEntry) -> bool:
        """Store cache entry with size management."""
        with self.lock:
            # Remove existing entry if present
            if entry.key in self.storage:
                old_entry = self.storage[entry.key]
                self.current_size -= old_entry.size_bytes
                del self.storage[entry.key]
                
            # Check if we need to evict entries
            while (self.current_size + entry.size_bytes > self.max_size_bytes and 
                   len(self.storage) > 0):
                # Remove least recently used
                old_key, old_entry = self.storage.popitem(last=False)
                self.current_size -= old_entry.size_bytes
                
            # Add new entry
            self.storage[entry.key] = entry
            self.current_size += entry.size_bytes
            return True
            
    def delete(self, key: str) -> bool:
        """Delete cache entry."""
        with self.lock:
            if key in self.storage:
                entry = self.storage[key]
                self.current_size -= entry.size_bytes
                del self.storage[key]
                return True
            return False
            
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return self.current_size
            
    def clear(self) -> None:
        """Clear all entries."""
        with self.lock:
            self.storage.clear()
            self.current_size = 0


class CompressedStorage(CacheStorage):
    """Compressed in-memory storage for larger datasets."""
    
    def __init__(self, max_size_bytes: int):
        self.max_size_bytes = max_size_bytes
        self.storage: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        self.current_size = 0
        
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get and decompress cache entry."""
        with self.lock:
            if key in self.storage:
                entry = self.storage[key]
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                # Decompress if needed
                if entry.compressed:
                    try:
                        entry.value = self._decompress(entry.value)
                        entry.compressed = False
                    except Exception as e:
                        logging.error(f"Decompression failed: {e}")
                        return None
                        
                return entry
            return None
            
    def put(self, entry: CacheEntry) -> bool:
        """Compress and store cache entry."""
        with self.lock:
            # Compress large entries
            if entry.size_bytes > 1024:  # Compress entries > 1KB
                try:
                    compressed_value = self._compress(entry.value)
                    compressed_size = len(compressed_value)
                    
                    if compressed_size < entry.size_bytes * 0.8:  # Only if 20%+ reduction
                        entry.value = compressed_value
                        entry.size_bytes = compressed_size
                        entry.compressed = True
                except Exception as e:
                    logging.warning(f"Compression failed: {e}")
                    
            # Store entry
            if entry.key in self.storage:
                old_entry = self.storage[entry.key]
                self.current_size -= old_entry.size_bytes
                
            self.storage[entry.key] = entry
            self.current_size += entry.size_bytes
            
            return True
            
    def delete(self, key: str) -> bool:
        """Delete compressed entry."""
        with self.lock:
            if key in self.storage:
                entry = self.storage[key]
                self.current_size -= entry.size_bytes
                del self.storage[key]
                return True
            return False
            
    def size(self) -> int:
        """Get current size."""
        with self.lock:
            return self.current_size
            
    def clear(self) -> None:
        """Clear all entries."""
        with self.lock:
            self.storage.clear()
            self.current_size = 0
            
    def _compress(self, data: Any) -> bytes:
        """Compress data using pickle and gzip."""
        import gzip
        pickled = pickle.dumps(data)
        return gzip.compress(pickled)
        
    def _decompress(self, compressed_data: bytes) -> Any:
        """Decompress data."""
        import gzip
        pickled = gzip.decompress(compressed_data)
        return pickle.loads(pickled)


class AdaptiveEvictionPolicy:
    """Machine learning-based adaptive cache eviction policy."""
    
    def __init__(self):
        self.access_patterns = defaultdict(list)
        self.prediction_weights = {
            'recency': 0.4,
            'frequency': 0.3,
            'size': 0.2,
            'pattern': 0.1
        }
        self.learning_rate = 0.01
        
    def predict_value(self, entry: CacheEntry) -> float:
        """Predict cache value for entry."""
        now = datetime.now()
        
        # Recency score (higher = more recent)
        time_diff = (now - entry.last_accessed).total_seconds()
        recency_score = 1.0 / (1.0 + time_diff / 3600)  # Decay over hours
        
        # Frequency score
        frequency_score = min(1.0, entry.access_count / 10.0)
        
        # Size score (smaller is better for eviction candidates)
        size_score = 1.0 / (1.0 + entry.size_bytes / 1024.0)  # Decay by KB
        
        # Pattern score based on access history
        pattern_score = self._calculate_pattern_score(entry.key)
        
        # Weighted combination
        value = (
            self.prediction_weights['recency'] * recency_score +
            self.prediction_weights['frequency'] * frequency_score +
            self.prediction_weights['size'] * size_score +
            self.prediction_weights['pattern'] * pattern_score
        )
        
        return value
        
    def _calculate_pattern_score(self, key: str) -> float:
        """Calculate score based on access patterns."""
        if key not in self.access_patterns:
            return 0.5  # Neutral score
            
        pattern = self.access_patterns[key]
        if len(pattern) < 2:
            return 0.5
            
        # Calculate access regularity
        intervals = [pattern[i+1] - pattern[i] for i in range(len(pattern)-1)]
        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            interval_variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
            
            # Regular access patterns get higher scores
            regularity = 1.0 / (1.0 + interval_variance / avg_interval if avg_interval > 0 else 1.0)
            return regularity
            
        return 0.5
        
    def record_access(self, key: str):
        """Record access for pattern learning."""
        self.access_patterns[key].append(time.time())
        
        # Keep only recent history
        cutoff_time = time.time() - 86400  # 24 hours
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff_time
        ]
        
    def update_weights(self, hit_rate: float, target_hit_rate: float = 0.8):
        """Update prediction weights based on performance."""
        error = target_hit_rate - hit_rate
        
        # Simple gradient update
        adjustment = self.learning_rate * error
        
        # Increase recency weight if hit rate is low
        self.prediction_weights['recency'] += adjustment * 0.5
        self.prediction_weights['frequency'] += adjustment * 0.3
        
        # Normalize weights
        total_weight = sum(self.prediction_weights.values())
        for key in self.prediction_weights:
            self.prediction_weights[key] /= total_weight


class AdaptiveCache:
    """High-performance adaptive cache with multiple levels."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize storage levels
        self.l1_storage = MemoryStorage(config.max_memory_mb * 1024 * 1024 // 2)
        self.l2_storage = CompressedStorage(config.max_memory_mb * 1024 * 1024 // 2)
        
        # Eviction policy
        if config.eviction_policy == CachePolicy.ADAPTIVE:
            self.eviction_policy = AdaptiveEvictionPolicy()
        else:
            self.eviction_policy = None
            
        # Statistics
        self.stats = CacheStats()
        self.stats_lock = threading.Lock()
        
        # Prefetching
        self.prefetch_queue = []
        self.prefetch_patterns = defaultdict(list)
        
        self.logger.info("Adaptive cache initialized")
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with adaptive retrieval."""
        start_time = time.time()
        
        # Generate cache key hash
        cache_key = self._generate_key(key)
        
        # Try L1 cache first
        entry = self.l1_storage.get(cache_key)
        if entry is not None:
            self._record_hit(time.time() - start_time)
            self._record_access_pattern(key)
            return entry.value
            
        # Try L2 cache
        entry = self.l2_storage.get(cache_key)
        if entry is not None:
            # Promote to L1 if frequently accessed
            if entry.access_count > 3:
                self._promote_to_l1(entry)
                
            self._record_hit(time.time() - start_time)
            self._record_access_pattern(key)
            return entry.value
            
        # Cache miss
        self._record_miss(time.time() - start_time)
        return None
        
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Store value in cache with intelligent placement."""
        cache_key = self._generate_key(key)
        
        # Calculate value size
        try:
            size_bytes = len(pickle.dumps(value))
        except Exception:
            size_bytes = 1024  # Default estimate
            
        # Create cache entry
        entry = CacheEntry(
            key=cache_key,
            value=value,
            size_bytes=size_bytes,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            ttl_seconds=ttl_seconds or self.config.default_ttl_seconds
        )
        
        # Decide cache level based on size and policy
        if size_bytes < 10 * 1024:  # Small items go to L1
            success = self.l1_storage.put(entry)
            entry.level = CacheLevel.L1_MEMORY
        else:  # Large items go to L2
            success = self.l2_storage.put(entry)
            entry.level = CacheLevel.L2_COMPRESSED
            
        if success:
            self._update_stats()
            self._trigger_prefetch(key)
            
        return success
        
    def delete(self, key: str) -> bool:
        """Delete value from all cache levels."""
        cache_key = self._generate_key(key)
        
        l1_deleted = self.l1_storage.delete(cache_key)
        l2_deleted = self.l2_storage.delete(cache_key)
        
        return l1_deleted or l2_deleted
        
    def clear(self) -> None:
        """Clear all cache levels."""
        self.l1_storage.clear()
        self.l2_storage.clear()
        
        with self.stats_lock:
            self.stats = CacheStats()
            
        self.logger.info("Cache cleared")
        
    def _generate_key(self, key: str) -> str:
        """Generate deterministic cache key."""
        if isinstance(key, str):
            return hashlib.md5(key.encode()).hexdigest()
        else:
            return hashlib.md5(str(key).encode()).hexdigest()
            
    def _promote_to_l1(self, entry: CacheEntry) -> bool:
        """Promote frequently accessed entry to L1."""
        try:
            # Remove from L2
            self.l2_storage.delete(entry.key)
            
            # Add to L1
            entry.level = CacheLevel.L1_MEMORY
            return self.l1_storage.put(entry)
        except Exception as e:
            self.logger.error(f"Failed to promote entry: {e}")
            return False
            
    def _record_hit(self, access_time: float):
        """Record cache hit statistics."""
        with self.stats_lock:
            self.stats.hits += 1
            self.stats.total_requests += 1
            self._update_access_time(access_time)
            
        # Update eviction policy
        if self.eviction_policy:
            self.eviction_policy.record_access(f"hit_{time.time()}")
            
    def _record_miss(self, access_time: float):
        """Record cache miss statistics."""
        with self.stats_lock:
            self.stats.misses += 1
            self.stats.total_requests += 1
            self._update_access_time(access_time)
            
    def _update_access_time(self, access_time: float):
        """Update average access time."""
        if self.stats.total_requests == 1:
            self.stats.avg_access_time_ms = access_time * 1000
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats.avg_access_time_ms = (
                alpha * access_time * 1000 + 
                (1 - alpha) * self.stats.avg_access_time_ms
            )
            
    def _update_stats(self):
        """Update comprehensive cache statistics."""
        with self.stats_lock:
            if self.stats.total_requests > 0:
                self.stats.hit_rate = self.stats.hits / self.stats.total_requests
                
            total_size = self.l1_storage.size() + self.l2_storage.size()
            max_size = self.config.max_memory_mb * 1024 * 1024
            self.stats.memory_utilization = total_size / max_size
            self.stats.total_size_bytes = total_size
            
    def _record_access_pattern(self, key: str):
        """Record access pattern for prefetching."""
        if not self.config.prefetch_enabled:
            return
            
        current_time = time.time()
        self.prefetch_patterns[key].append(current_time)
        
        # Keep only recent history
        cutoff_time = current_time - 3600  # 1 hour
        self.prefetch_patterns[key] = [
            t for t in self.prefetch_patterns[key] if t > cutoff_time
        ]
        
    def _trigger_prefetch(self, key: str):
        """Trigger intelligent prefetching based on patterns."""
        if not self.config.prefetch_enabled:
            return
            
        # Analyze access patterns to predict next accesses
        # This is a simplified implementation
        pattern = self.prefetch_patterns.get(key, [])
        if len(pattern) >= 3:
            # Calculate average interval
            intervals = [pattern[i+1] - pattern[i] for i in range(len(pattern)-1)]
            avg_interval = sum(intervals) / len(intervals)
            
            # If pattern is regular, add to prefetch queue
            if avg_interval < 3600:  # Within 1 hour
                self.prefetch_queue.append((key, time.time() + avg_interval))
                
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.stats_lock:
            return {
                "performance": {
                    "hit_rate": self.stats.hit_rate,
                    "total_requests": self.stats.total_requests,
                    "hits": self.stats.hits,
                    "misses": self.stats.misses,
                    "avg_access_time_ms": self.stats.avg_access_time_ms
                },
                "memory": {
                    "utilization_percent": self.stats.memory_utilization * 100,
                    "total_size_bytes": self.stats.total_size_bytes,
                    "l1_size_bytes": self.l1_storage.size(),
                    "l2_size_bytes": self.l2_storage.size()
                },
                "configuration": {
                    "max_memory_mb": self.config.max_memory_mb,
                    "eviction_policy": self.config.eviction_policy.value,
                    "compression_enabled": self.config.compression_enabled,
                    "prefetch_enabled": self.config.prefetch_enabled
                }
            }
            
    def optimize_performance(self) -> Dict[str, Any]:
        """Analyze and optimize cache performance."""
        stats = self.get_statistics()
        recommendations = []
        
        # Analyze hit rate
        if stats["performance"]["hit_rate"] < 0.7:
            recommendations.append("Low hit rate - consider increasing cache size")
            
        # Analyze memory utilization
        if stats["memory"]["utilization_percent"] > 90:
            recommendations.append("High memory usage - consider enabling compression or increasing cache size")
        elif stats["memory"]["utilization_percent"] < 30:
            recommendations.append("Low memory usage - cache size may be oversized")
            
        # Analyze access time
        if stats["performance"]["avg_access_time_ms"] > 10:
            recommendations.append("High access time - consider optimizing cache structure")
            
        # Update adaptive policy
        if self.eviction_policy and hasattr(self.eviction_policy, 'update_weights'):
            self.eviction_policy.update_weights(stats["performance"]["hit_rate"])
            
        return {
            "current_performance": stats,
            "recommendations": recommendations,
            "optimizations_applied": ["adaptive_weights_updated"] if self.eviction_policy else []
        }


# Global cache instance for easy access
_global_cache: Optional[AdaptiveCache] = None
_cache_lock = threading.Lock()


def get_global_cache(config: Optional[CacheConfig] = None) -> AdaptiveCache:
    """Get or create global cache instance."""
    global _global_cache
    
    with _cache_lock:
        if _global_cache is None:
            if config is None:
                config = CacheConfig()
            _global_cache = AdaptiveCache(config)
            
        return _global_cache


def cache_function_result(ttl_seconds: Optional[int] = None):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_data = f"{func.__name__}_{args}_{sorted(kwargs.items())}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            cache = get_global_cache()
            cached_result = cache.get(cache_key)
            
            if cached_result is not None:
                return cached_result
                
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl_seconds)
            
            return result
            
        return wrapper
    return decorator


# Example usage decorator
@cache_function_result(ttl_seconds=3600)
def expensive_computation(data: Array) -> Array:
    """Example of cached expensive computation."""
    # Simulate expensive computation
    if jnp is not None:
        return jnp.sum(data ** 2, axis=-1)
    else:
        return sum(x ** 2 for x in data)