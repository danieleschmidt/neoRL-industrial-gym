# Performance Benchmarking Guide

This document describes the comprehensive performance benchmarking framework for neoRL-industrial-gym, designed to monitor and optimize performance across all system components.

## Overview

Performance benchmarking is critical for industrial RL applications where real-time constraints and resource efficiency directly impact safety and operational effectiveness. Our benchmarking framework covers:

- **Environment Simulation Performance**: Step execution times and throughput
- **RL Algorithm Performance**: Training speed and memory efficiency  
- **JAX Compilation Performance**: JIT compilation and execution optimization
- **Safety Validation Overhead**: Cost of safety constraint checking
- **Memory Usage Patterns**: Memory allocation and garbage collection
- **Regression Detection**: Automated performance monitoring

## Benchmarking Framework

### Core Components

#### 1. Benchmark Suite (`scripts/benchmark_suite.py`)

Comprehensive benchmarking tool that measures:

```bash
# Run all benchmarks
python scripts/benchmark_suite.py

# Run specific categories
python scripts/benchmark_suite.py --categories jax environment rl

# Compare with baseline
python scripts/benchmark_suite.py --baseline benchmark_results/baseline_results.json --compare-baseline

# Save new baseline
python scripts/benchmark_suite.py --save-baseline
```

#### 2. Performance Categories

**JAX Performance**:
- JIT compilation times
- Execution speed for different array sizes
- Memory usage during computation
- GPU vs CPU performance comparison

**Environment Simulation**:
- Environment step execution time
- State transition computation
- Safety constraint validation overhead
- Reward calculation performance

**RL Algorithm Performance**:
- Policy network forward pass
- Value function updates
- Gradient computation and application
- Batch processing efficiency

**Memory Performance**:
- Memory allocation patterns
- Peak memory usage
- Garbage collection impact
- Memory leaks detection

### Benchmark Execution

#### Basic Usage

```python
from scripts.benchmark_suite import PerformanceBenchmarker

# Initialize benchmarker
benchmarker = PerformanceBenchmarker()

# Run specific benchmark
result = benchmarker.benchmark_function(
    func=my_function,
    name="function_benchmark",
    category="custom",
    iterations=100,
    warmup=10
)

# Run all benchmarks
benchmarker.run_all_benchmarks()

# Generate and save report
benchmarker.save_results()
benchmarker.print_summary()
```

#### Advanced Configuration

```python
# Custom benchmark with metadata
benchmarker.benchmark_function(
    func=train_policy,
    name="policy_training_benchmark",
    category="rl_training",
    iterations=50,
    warmup=5,
    # Custom parameters
    batch_size=256,
    learning_rate=3e-4,
    environment="ChemicalReactor-v0"
)
```

## Performance Metrics

### Key Performance Indicators (KPIs)

| Metric | Description | Target | Alert Threshold |
|--------|-------------|--------|-----------------|
| **Environment Step Time** | Time per simulation step | < 1ms | > 5ms |
| **Policy Forward Pass** | Neural network inference | < 0.1ms | > 1ms |
| **Safety Validation** | Constraint checking overhead | < 10% | > 25% |
| **Memory Usage** | Peak memory per episode | < 100MB | > 500MB |
| **Compilation Time** | JAX JIT compilation | < 5s | > 30s |
| **Training Throughput** | Steps processed per second | > 1000 steps/s | < 100 steps/s |

### Measurement Precision

```python
# High-precision timing
import time
start = time.perf_counter()
result = function_under_test()
duration = time.perf_counter() - start

# Memory tracking
import tracemalloc
tracemalloc.start()
result = function_under_test()
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

# CPU usage monitoring
import psutil
process = psutil.Process()
cpu_before = process.cpu_percent()
result = function_under_test()
cpu_after = process.cpu_percent()
```

## Automated Performance Monitoring

### CI/CD Integration

#### GitHub Actions Benchmark Workflow

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2 AM

jobs:
  performance-benchmarks:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install psutil
      
      - name: Download baseline results
        uses: actions/download-artifact@v3
        with:
          name: baseline-results
          path: benchmark_results/
        continue-on-error: true
      
      - name: Run benchmarks
        run: |
          python scripts/benchmark_suite.py \
            --baseline benchmark_results/baseline_results.json \
            --compare-baseline
      
      - name: Upload benchmark results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark_results/
      
      - name: Update baseline (main branch only)
        if: github.ref == 'refs/heads/main'
        run: |
          python scripts/benchmark_suite.py --save-baseline
      
      - name: Upload new baseline
        if: github.ref == 'refs/heads/main'
        uses: actions/upload-artifact@v3
        with:
          name: baseline-results
          path: benchmark_results/baseline_results.json
```

### Performance Regression Detection

```python
def detect_performance_regression(current_results, baseline_results, threshold=0.1):
    """Detect performance regressions."""
    regressions = []
    
    for benchmark_name, current in current_results.items():
        baseline = baseline_results.get(benchmark_name)
        if not baseline:
            continue
            
        # Check throughput regression
        throughput_change = (current.throughput - baseline.throughput) / baseline.throughput
        if throughput_change < -threshold:  # 10% slower
            regressions.append({
                'benchmark': benchmark_name,
                'metric': 'throughput',
                'change_percent': throughput_change * 100,
                'severity': 'high' if throughput_change < -0.2 else 'medium'
            })
        
        # Check memory regression
        memory_change = (current.memory_peak_mb - baseline.memory_peak_mb) / baseline.memory_peak_mb
        if memory_change > 0.2:  # 20% more memory
            regressions.append({
                'benchmark': benchmark_name,
                'metric': 'memory',
                'change_percent': memory_change * 100,
                'severity': 'medium'
            })
    
    return regressions
```

## Industrial Environment Benchmarks

### Environment-Specific Performance Tests

```python
def benchmark_chemical_reactor():
    """Benchmark ChemicalReactor-v0 performance."""
    import neorl_industrial as ni
    
    env = ni.make('ChemicalReactor-v0')
    
    # Warmup
    for _ in range(10):
        env.reset()
        for _ in range(100):
            action = env.action_space.sample()
            env.step(action)
    
    # Benchmark
    start_time = time.perf_counter()
    total_steps = 0
    
    for episode in range(100):
        env.reset()
        for step in range(1000):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_steps += 1
            
            if done:
                break
    
    duration = time.perf_counter() - start_time
    throughput = total_steps / duration
    
    return {
        'environment': 'ChemicalReactor-v0',
        'throughput_steps_per_sec': throughput,
        'avg_episode_length': total_steps / 100,
        'total_duration': duration
    }
```

### Safety Validation Performance

```python
def benchmark_safety_validation():
    """Benchmark safety constraint validation overhead."""
    
    def step_with_safety(env, action):
        """Environment step with safety validation."""
        obs, reward, done, info = env.step(action)
        
        # Safety constraint checks
        temp = obs[0]
        pressure = obs[1]
        
        # Temperature constraint (0°C to 100°C)
        temp_safe = 273.15 <= temp <= 373.15
        
        # Pressure constraint (0 to 10 bar)
        pressure_safe = 0.0 <= pressure <= 10.0
        
        if not (temp_safe and pressure_safe):
            info['safety_violation'] = True
            done = True
            reward = -100  # Safety penalty
        
        return obs, reward, done, info
    
    def step_without_safety(env, action):
        """Environment step without safety validation."""
        return env.step(action)
    
    # Benchmark both versions
    env = create_test_env()
    
    # With safety
    safety_time = benchmark_function(
        lambda: run_episode(env, step_with_safety),
        iterations=100
    )
    
    # Without safety
    no_safety_time = benchmark_function(
        lambda: run_episode(env, step_without_safety),
        iterations=100
    )
    
    overhead_percent = ((safety_time - no_safety_time) / no_safety_time) * 100
    
    return {
        'safety_validation_overhead_percent': overhead_percent,
        'acceptable': overhead_percent < 10  # Less than 10% overhead
    }
```

## Performance Optimization Guidelines

### JAX Optimization

```python
# Use JIT compilation for computational kernels
@jax.jit
def compute_q_values(states, actions, params):
    """JIT-compiled Q-value computation."""
    return network_forward(params, states, actions)

# Batch operations for better GPU utilization
def batch_train_step(batch_states, batch_actions, batch_rewards):
    """Vectorized training step."""
    # Use vmap for automatic vectorization
    batched_update = jax.vmap(single_update, in_axes=(0, 0, 0))
    return batched_update(batch_states, batch_actions, batch_rewards)

# Pre-compile functions during initialization
def initialize_agent():
    """Initialize agent with pre-compiled functions."""
    dummy_state = jnp.zeros((1, state_dim))
    dummy_action = jnp.zeros((1, action_dim))
    
    # Trigger compilation
    _ = compute_q_values(dummy_state, dummy_action, initial_params)
```

### Memory Optimization

```python
# Use memory-efficient data structures
def create_replay_buffer(capacity):
    """Memory-efficient replay buffer."""
    return {
        'states': np.zeros((capacity, state_dim), dtype=np.float32),
        'actions': np.zeros((capacity, action_dim), dtype=np.float32),
        'rewards': np.zeros(capacity, dtype=np.float32),
        'dones': np.zeros(capacity, dtype=bool),
    }

# Clear JAX caches periodically
def clear_jax_caches():
    """Clear JAX compilation caches to free memory."""
    jax.clear_caches()

# Use gradient checkpointing for large models
from jax.experimental import checkpoint

@checkpoint
def large_network_layer(x, params):
    """Checkpointed layer to save memory."""
    return forward_pass(x, params)
```

## Benchmark Result Analysis

### Performance Trends

```python
def analyze_performance_trends(benchmark_history):
    """Analyze performance trends over time."""
    import matplotlib.pyplot as plt
    
    # Extract throughput trends
    dates = [result['timestamp'] for result in benchmark_history]
    throughputs = [result['avg_throughput'] for result in benchmark_history]
    
    plt.figure(figsize=(12, 6))
    plt.plot(dates, throughputs, marker='o')
    plt.title('Performance Trend Analysis')
    plt.xlabel('Date')
    plt.ylabel('Throughput (ops/sec)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('performance_trends.png')
    
    # Detect trend direction
    if len(throughputs) >= 2:
        recent_avg = np.mean(throughputs[-3:])  # Last 3 measurements
        historical_avg = np.mean(throughputs[:-3])  # All but last 3
        
        trend = "improving" if recent_avg > historical_avg else "declining"
        change_percent = ((recent_avg - historical_avg) / historical_avg) * 100
        
        return {
            'trend': trend,
            'change_percent': change_percent,
            'recent_avg': recent_avg,
            'historical_avg': historical_avg
        }
```

### Performance Report Generation

```python
def generate_performance_report(benchmark_results):
    """Generate comprehensive performance report."""
    
    report = {
        'summary': {
            'total_benchmarks': len(benchmark_results),
            'avg_throughput': np.mean([r.throughput for r in benchmark_results]),
            'peak_memory_mb': max([r.memory_peak_mb for r in benchmark_results]),
            'total_duration': sum([r.duration for r in benchmark_results])
        },
        'by_category': {},
        'recommendations': []
    }
    
    # Group by category
    categories = {}
    for result in benchmark_results:
        if result.category not in categories:
            categories[result.category] = []
        categories[result.category].append(result)
    
    # Analyze each category
    for category, results in categories.items():
        throughputs = [r.throughput for r in results]
        memories = [r.memory_peak_mb for r in results]
        
        report['by_category'][category] = {
            'count': len(results),
            'avg_throughput': np.mean(throughputs),
            'std_throughput': np.std(throughputs),
            'avg_memory': np.mean(memories),
            'max_memory': max(memories)
        }
        
        # Generate recommendations
        if np.mean(throughputs) < 100:  # Low throughput
            report['recommendations'].append(
                f"Consider optimizing {category} - low throughput detected"
            )
        
        if max(memories) > 1000:  # High memory usage
            report['recommendations'].append(
                f"Consider memory optimization for {category} - high memory usage"
            )
    
    return report
```

## Best Practices

### Benchmarking Guidelines

1. **Consistent Environment**: Run benchmarks on consistent hardware
2. **Warmup Runs**: Always include warmup iterations to account for JIT compilation
3. **Multiple Iterations**: Use sufficient iterations for statistical significance
4. **Isolation**: Run benchmarks in isolated environments
5. **Documentation**: Document any performance optimization changes

### Performance Targets

```python
PERFORMANCE_TARGETS = {
    'environment_step_time_ms': 1.0,
    'policy_inference_time_ms': 0.1,
    'safety_validation_overhead_percent': 10.0,
    'memory_usage_per_episode_mb': 100.0,
    'training_throughput_steps_per_sec': 1000.0
}

def check_performance_targets(benchmark_results):
    """Check if benchmark results meet performance targets."""
    violations = []
    
    for result in benchmark_results:
        # Extract relevant metrics and compare with targets
        # Add to violations if targets are not met
        pass
    
    return violations
```

### Continuous Monitoring

1. **Automated Benchmarks**: Run benchmarks on every commit
2. **Performance Dashboards**: Visualize performance trends
3. **Alert System**: Notify on performance regressions
4. **Regular Reviews**: Monthly performance review meetings

This comprehensive benchmarking framework ensures that neoRL-industrial-gym maintains high performance standards suitable for industrial applications while providing the tools necessary to identify and address performance issues proactively.