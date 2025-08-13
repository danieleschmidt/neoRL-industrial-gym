"""Intelligent caching system with adaptive policies and predictive prefetching."""

import time
import threading
import pickle
import hashlib
from typing import Any, Dict, Optional, Callable, List, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, OrderedDict
from abc import ABC, abstractmethod
import numpy as np
import logging
from pathlib import Path
import weakref
import gzip
import json

from ..monitoring.dashboard import record_metric

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    ttl: Optional[float] = None
    priority: int = 0
    computation_time: float = 0.0  # Time it took to compute this value
    
    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    @property
    def age(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.timestamp
    
    @property
    def access_frequency(self) -> float:
        """Calculate access frequency (accesses per second)."""
        age = max(self.age, 1.0)  # Avoid division by zero
        return self.access_count / age

class CacheEvictionPolicy(ABC):
    """Base class for cache eviction policies."""
    
    @abstractmethod
    def select_for_eviction(self, entries: Dict[str, CacheEntry]) -> List[str]:
        """Select keys for eviction."""
        pass

class LRUEvictionPolicy(CacheEvictionPolicy):
    """Least Recently Used eviction policy."""
    
    def select_for_eviction(self, entries: Dict[str, CacheEntry]) -> List[str]:
        # Sort by last access time (oldest first)
        sorted_entries = sorted(entries.items(), key=lambda x: x[1].last_access)
        return [key for key, _ in sorted_entries]

class LFUEvictionPolicy(CacheEvictionPolicy):
    """Least Frequently Used eviction policy."""
    
    def select_for_eviction(self, entries: Dict[str, CacheEntry]) -> List[str]:
        # Sort by access frequency (lowest first)
        sorted_entries = sorted(entries.items(), key=lambda x: x[1].access_frequency)
        return [key for key, _ in sorted_entries]

class AdaptiveEvictionPolicy(CacheEvictionPolicy):
    """Adaptive eviction policy combining multiple factors."""
    
    def __init__(self, recency_weight: float = 0.3, frequency_weight: float = 0.3, 
                 size_weight: float = 0.2, priority_weight: float = 0.2):
        self.recency_weight = recency_weight
        self.frequency_weight = frequency_weight
        self.size_weight = size_weight
        self.priority_weight = priority_weight
    
    def select_for_eviction(self, entries: Dict[str, CacheEntry]) -> List[str]:
        if not entries:
            return []
        
        # Calculate scores for each entry
        scores = {}
        current_time = time.time()
        
        # Normalize factors
        max_age = max(entry.age for entry in entries.values())
        max_freq = max(entry.access_frequency for entry in entries.values())
        max_size = max(entry.size_bytes for entry in entries.values())
        max_priority = max(entry.priority for entry in entries.values())
        
        for key, entry in entries.items():
            # Higher score = more likely to be evicted
            recency_score = entry.age / max_age if max_age > 0 else 0
            frequency_score = 1 - (entry.access_frequency / max_freq) if max_freq > 0 else 1
            size_score = entry.size_bytes / max_size if max_size > 0 else 0
            priority_score = 1 - (entry.priority / max_priority) if max_priority > 0 else 1
            
            total_score = (
                self.recency_weight * recency_score +
                self.frequency_weight * frequency_score +
                self.size_weight * size_score +
                self.priority_weight * priority_score
            )
            
            scores[key] = total_score
        
        # Sort by score (highest first = most likely to evict)
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return sorted_keys

class IntelligentCache:
    """Intelligent cache with adaptive policies and predictive prefetching."""
    
    def __init__(
        self,
        max_size_mb: float = 1024,  # 1GB default
        max_entries: int = 10000,
        eviction_policy: Optional[CacheEvictionPolicy] = None,
        enable_persistence: bool = True,
        persistence_path: Optional[str] = None,
        enable_compression: bool = True,
        enable_prefetching: bool = True,
        prefetch_threshold: float = 0.7,  # Prefetch when cache is 70% full
    ):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_entries = max_entries
        self.eviction_policy = eviction_policy or AdaptiveEvictionPolicy()
        self.enable_persistence = enable_persistence
        self.enable_compression = enable_compression
        self.enable_prefetching = enable_prefetching
        self.prefetch_threshold = prefetch_threshold
        
        # Storage
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order = OrderedDict()  # For LRU tracking
        self._current_size_bytes = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "prefetches": 0,
            "persisted": 0,
            "restored": 0,
        }
        
        # Access pattern tracking for prediction
        self.access_patterns = defaultdict(list)  # key -> [timestamps]
        self.access_sequences = []  # List of recent accesses for sequence prediction
        self.max_sequence_length = 1000
        
        # Persistence
        if enable_persistence:
            self.persistence_path = Path(persistence_path or "cache_persistence")
            self.persistence_path.mkdir(exist_ok=True)
            self._load_persistent_cache()
        
        # Background threads
        self._stop_background = threading.Event()
        self._maintenance_thread = threading.Thread(target=self._maintenance_worker, daemon=True)
        self._maintenance_thread.start()
        
        self.logger = logging.getLogger("intelligent_cache")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check expiration
                if entry.is_expired:
                    self._remove_entry(key)
                    self.stats["misses"] += 1
                    return default
                
                # Update access information
                entry.access_count += 1
                entry.last_access = time.time()
                self._access_order.move_to_end(key)
                
                # Track access pattern
                self._track_access(key)
                
                self.stats["hits"] += 1
                return entry.value
            else:
                self.stats["misses"] += 1
                return default
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        priority: int = 0,
        computation_time: float = 0.0,
    ) -> None:
        """Put value in cache."""
        with self._lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Check if we need to make room
            self._ensure_capacity(size_bytes)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes,
                ttl=ttl,
                priority=priority,
                computation_time=computation_time,
            )
            
            # Remove existing entry if present
            if key in self._cache:
                self._remove_entry(key)
            
            # Add new entry
            self._cache[key] = entry
            self._access_order[key] = True
            self._current_size_bytes += size_bytes
            
            # Track access pattern
            self._track_access(key)
            
            # Record metrics
            record_metric("cache_size_mb", self._current_size_bytes / (1024 * 1024), 
                         tags={"component": "cache"})
            record_metric("cache_entries", len(self._cache), tags={"component": "cache"})
            
            # Trigger prefetching if enabled
            if self.enable_prefetching:
                self._trigger_prefetching()
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._current_size_bytes = 0
            self.access_patterns.clear()
            self.access_sequences.clear()
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update size tracking."""
        if key in self._cache:
            entry = self._cache[key]
            self._current_size_bytes -= entry.size_bytes
            del self._cache[key]
            self._access_order.pop(key, None)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes."""
        try:
            if self.enable_compression:
                serialized = pickle.dumps(value)
                compressed = gzip.compress(serialized)
                return len(compressed)
            else:
                return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            return self._estimate_size(value)
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate size when pickle fails."""
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, (list, tuple)):
            return sum(self._estimate_size(item) for item in value)
        elif isinstance(value, dict):
            return sum(self._estimate_size(k) + self._estimate_size(v) for k, v in value.items())
        elif isinstance(value, str):
            return len(value.encode('utf-8'))
        else:
            return 100  # Default estimate
    
    def _ensure_capacity(self, new_size: int) -> None:
        """Ensure cache has capacity for new entry."""
        # Check entry limit
        while len(self._cache) >= self.max_entries:
            self._evict_entries(1)
        
        # Check size limit
        while self._current_size_bytes + new_size > self.max_size_bytes:
            # Evict until we have enough space
            target_size = self.max_size_bytes - new_size
            bytes_to_free = self._current_size_bytes - target_size
            self._evict_by_size(bytes_to_free)
    
    def _evict_entries(self, count: int) -> None:
        """Evict specified number of entries."""
        if not self._cache:
            return
        
        keys_to_evict = self.eviction_policy.select_for_eviction(self._cache)
        
        for i, key in enumerate(keys_to_evict):
            if i >= count:
                break
            if key in self._cache:
                self._remove_entry(key)
                self.stats["evictions"] += 1
    
    def _evict_by_size(self, target_bytes: int) -> None:
        """Evict entries until target bytes are freed."""
        if not self._cache:
            return
        
        freed_bytes = 0
        keys_to_evict = self.eviction_policy.select_for_eviction(self._cache)
        
        for key in keys_to_evict:
            if freed_bytes >= target_bytes:
                break
            if key in self._cache:
                freed_bytes += self._cache[key].size_bytes
                self._remove_entry(key)
                self.stats["evictions"] += 1
    
    def _track_access(self, key: str) -> None:
        """Track access patterns for prediction."""
        current_time = time.time()
        
        # Track access times for this key
        self.access_patterns[key].append(current_time)
        
        # Keep only recent accesses (last hour)
        cutoff_time = current_time - 3600  # 1 hour
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t >= cutoff_time
        ]
        
        # Track access sequences
        self.access_sequences.append((key, current_time))
        if len(self.access_sequences) > self.max_sequence_length:
            self.access_sequences.pop(0)
    
    def _trigger_prefetching(self) -> None:
        """Trigger predictive prefetching if cache utilization is below threshold."""
        utilization = self._current_size_bytes / self.max_size_bytes
        
        if utilization < self.prefetch_threshold:
            # Run prefetching in background
            threading.Thread(target=self._run_prefetching, daemon=True).start()
    
    def _run_prefetching(self) -> None:
        """Run predictive prefetching."""
        try:
            predicted_keys = self._predict_next_accesses()
            
            for key in predicted_keys[:5]:  # Limit prefetching
                if key not in self._cache:
                    # Would need callback to compute value
                    # For now, just record that we would prefetch
                    self.stats["prefetches"] += 1
                    self.logger.debug(f"Would prefetch key: {key}")
        
        except Exception as e:
            self.logger.error(f"Prefetching error: {e}")
    
    def _predict_next_accesses(self) -> List[str]:
        """Predict which keys are likely to be accessed next."""
        predictions = []
        current_time = time.time()
        
        # Method 1: Frequency-based prediction
        key_frequencies = {}
        for key, timestamps in self.access_patterns.items():
            if timestamps:
                key_frequencies[key] = len(timestamps) / (current_time - timestamps[0] + 1)
        
        # Method 2: Temporal pattern prediction
        if len(self.access_sequences) >= 10:
            recent_sequence = [item[0] for item in self.access_sequences[-10:]]
            # Simple pattern: if we see A->B frequently, predict B after A
            last_key = recent_sequence[-1]
            pattern_predictions = self._find_sequence_patterns(recent_sequence, last_key)
            predictions.extend(pattern_predictions)
        
        # Method 3: Periodic pattern prediction
        for key, timestamps in self.access_patterns.items():
            if len(timestamps) >= 3:
                intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
                avg_interval = np.mean(intervals)
                time_since_last = current_time - timestamps[-1]
                
                # If we're due for next access based on pattern
                if time_since_last >= avg_interval * 0.8:
                    predictions.append(key)
        
        # Combine and rank predictions
        prediction_scores = defaultdict(float)
        for key in predictions:
            prediction_scores[key] += 1
        
        for key, freq in key_frequencies.items():
            prediction_scores[key] += freq
        
        # Return top predictions
        sorted_predictions = sorted(prediction_scores.keys(), 
                                  key=lambda k: prediction_scores[k], reverse=True)
        return sorted_predictions
    
    def _find_sequence_patterns(self, sequence: List[str], current_key: str) -> List[str]:
        """Find keys that typically follow the current key."""
        patterns = defaultdict(int)
        
        for i in range(len(sequence) - 1):
            if sequence[i] == current_key:
                next_key = sequence[i + 1]
                patterns[next_key] += 1
        
        # Return keys sorted by frequency
        return sorted(patterns.keys(), key=lambda k: patterns[k], reverse=True)
    
    def _maintenance_worker(self) -> None:
        """Background maintenance worker."""
        while not self._stop_background.wait(60):  # Run every minute
            try:
                self._cleanup_expired()
                self._persist_cache()
                self._update_cache_metrics()
            except Exception as e:
                self.logger.error(f"Maintenance error: {e}")
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() 
                if entry.is_expired
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
    
    def _persist_cache(self) -> None:
        """Persist important cache entries to disk."""
        if not self.enable_persistence:
            return
        
        try:
            # Select entries worth persisting (high value, frequently accessed)
            persistent_entries = {}
            
            for key, entry in self._cache.items():
                # Criteria for persistence
                if (entry.access_count >= 5 and 
                    entry.computation_time > 1.0 and  # Expensive to compute
                    not entry.is_expired):
                    
                    persistent_entries[key] = {
                        "value": entry.value,
                        "timestamp": entry.timestamp,
                        "access_count": entry.access_count,
                        "ttl": entry.ttl,
                        "priority": entry.priority,
                    }
            
            if persistent_entries:
                persist_file = self.persistence_path / "cache_snapshot.json.gz"
                with gzip.open(persist_file, 'wt') as f:
                    json.dump(persistent_entries, f, default=str)
                
                self.stats["persisted"] += len(persistent_entries)
        
        except Exception as e:
            self.logger.error(f"Persistence error: {e}")
    
    def _load_persistent_cache(self) -> None:
        """Load persistent cache entries."""
        if not self.enable_persistence:
            return
        
        try:
            persist_file = self.persistence_path / "cache_snapshot.json.gz"
            if persist_file.exists():
                with gzip.open(persist_file, 'rt') as f:
                    persistent_entries = json.load(f)
                
                for key, data in persistent_entries.items():
                    # Check if entry is still valid
                    if data.get("ttl"):
                        age = time.time() - data["timestamp"]
                        if age > data["ttl"]:
                            continue
                    
                    # Restore entry
                    self.put(
                        key=key,
                        value=data["value"],
                        ttl=data.get("ttl"),
                        priority=data.get("priority", 0),
                    )
                    
                    # Restore access count
                    if key in self._cache:
                        self._cache[key].access_count = data.get("access_count", 0)
                
                self.stats["restored"] += len(persistent_entries)
                self.logger.info(f"Restored {len(persistent_entries)} cache entries")
        
        except Exception as e:
            self.logger.error(f"Cache restoration error: {e}")
    
    def _update_cache_metrics(self) -> None:
        """Update cache performance metrics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        record_metric("cache_hit_rate", hit_rate, tags={"component": "cache"})
        record_metric("cache_utilization", self._current_size_bytes / self.max_size_bytes, 
                     tags={"component": "cache"})
        record_metric("cache_evictions", self.stats["evictions"], tags={"component": "cache"})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            "hit_rate": hit_rate,
            "entries": len(self._cache),
            "size_mb": self._current_size_bytes / (1024 * 1024),
            "utilization": self._current_size_bytes / self.max_size_bytes,
            "avg_entry_size": self._current_size_bytes / len(self._cache) if self._cache else 0,
        }
    
    def shutdown(self) -> None:
        """Shutdown cache and cleanup resources."""
        self._stop_background.set()
        if self._maintenance_thread.is_alive():
            self._maintenance_thread.join(timeout=5)
        
        if self.enable_persistence:
            self._persist_cache()

# Global cache instance
_global_cache = None

def get_cache() -> IntelligentCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = IntelligentCache()
    return _global_cache

def cached(ttl: Optional[float] = None, priority: int = 0):
    """Decorator for caching function results."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_data = {
                "func": func.__name__,
                "args": args,
                "kwargs": kwargs,
            }
            key = hashlib.sha256(str(key_data).encode()).hexdigest()
            
            cache = get_cache()
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Compute result
            start_time = time.time()
            result = func(*args, **kwargs)
            computation_time = time.time() - start_time
            
            # Store in cache
            cache.put(key, result, ttl=ttl, priority=priority, computation_time=computation_time)
            
            return result
        
        return wrapper
    return decorator