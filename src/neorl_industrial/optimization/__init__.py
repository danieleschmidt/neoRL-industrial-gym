"""Performance optimization utilities for industrial RL."""

from .caching import (
    AdaptiveCache,
    cached_computation,
    get_global_cache,
    get_weight_cache,
    NetworkWeightCache,
)

from .performance import (
    PerformanceOptimizer,
    get_performance_optimizer,
    auto_optimize_function,
    benchmark_function,
    DataloaderOptimizer,
)

from .auto_tuning import (
    AutoTuner,
    HyperparameterSpace,
    OptimizationConfig,
    create_default_search_space,
    auto_tune_agent,
)

from .memory_optimization import (
    MemoryOptimizer,
    MemoryMonitor,
    AdaptiveBatchSizer,
    StreamingDataLoader,
    MemoryConfig,
    get_memory_optimizer,
    optimize_memory,
    get_memory_report,
)

__all__ = [
    # Caching
    "AdaptiveCache",
    "cached_computation", 
    "get_global_cache",
    "get_weight_cache",
    "NetworkWeightCache",
    
    # Performance
    "PerformanceOptimizer",
    "get_performance_optimizer",
    "auto_optimize_function",
    "benchmark_function",
    "DataloaderOptimizer",
    
    # Auto-tuning
    "AutoTuner",
    "HyperparameterSpace",
    "OptimizationConfig",
    "create_default_search_space",
    "auto_tune_agent",
    
    # Memory optimization
    "MemoryOptimizer",
    "MemoryMonitor",
    "AdaptiveBatchSizer",
    "StreamingDataLoader",
    "MemoryConfig",
    "get_memory_optimizer",
    "optimize_memory",
    "get_memory_report",
]