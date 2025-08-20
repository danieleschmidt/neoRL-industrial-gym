#!/usr/bin/env python3
"""Performance benchmarks for neoRL Industrial."""

import time
import numpy as np
import psutil
import json
from pathlib import Path
import sys

try:
    import neorl_industrial as ni
    print("✓ Package imported successfully")
except ImportError as e:
    print(f"✗ Failed to import package: {e}")
    sys.exit(1)


def measure_memory():
    """Get current memory usage."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB


def benchmark_environment_creation(n_envs=10):
    """Benchmark environment creation speed."""
    print(f"\n📊 Benchmarking environment creation ({n_envs} environments)...")
    
    start_mem = measure_memory()
    start_time = time.time()
    
    envs = []
    for i in range(n_envs):
        env = ni.make('ChemicalReactor-v0')
        envs.append(env)
    
    end_time = time.time()
    end_mem = measure_memory()
    
    avg_time = (end_time - start_time) / n_envs
    mem_per_env = (end_mem - start_mem) / n_envs
    
    print(f"  ✓ Average creation time: {avg_time:.4f}s")
    print(f"  ✓ Memory per environment: {mem_per_env:.2f}MB")
    
    return {
        'avg_creation_time': avg_time,
        'memory_per_env': mem_per_env,
        'total_time': end_time - start_time
    }


def benchmark_agent_creation(n_agents=5):
    """Benchmark agent creation speed."""
    print(f"\n🤖 Benchmarking agent creation ({n_agents} agents)...")
    
    start_mem = measure_memory()
    start_time = time.time()
    
    agents = []
    for i in range(n_agents):
        agent = ni.CQLAgent(state_dim=12, action_dim=3)
        agents.append(agent)
    
    end_time = time.time()
    end_mem = measure_memory()
    
    avg_time = (end_time - start_time) / n_agents
    mem_per_agent = (end_mem - start_mem) / n_agents
    
    print(f"  ✓ Average creation time: {avg_time:.4f}s")
    print(f"  ✓ Memory per agent: {mem_per_agent:.2f}MB")
    
    return {
        'avg_creation_time': avg_time,
        'memory_per_agent': mem_per_agent,
        'total_time': end_time - start_time
    }


def benchmark_dataset_loading():
    """Benchmark dataset loading speed."""
    print(f"\n📁 Benchmarking dataset loading...")
    
    env = ni.make('ChemicalReactor-v0')
    
    start_time = time.time()
    dataset = env.get_dataset(quality='medium')
    end_time = time.time()
    
    n_samples = len(dataset['observations'])
    load_time = end_time - start_time
    samples_per_sec = n_samples / load_time
    
    print(f"  ✓ Dataset size: {n_samples} samples")
    print(f"  ✓ Load time: {load_time:.4f}s")
    print(f"  ✓ Load speed: {samples_per_sec:.0f} samples/sec")
    
    return {
        'dataset_size': n_samples,
        'load_time': load_time,
        'samples_per_sec': samples_per_sec
    }


def benchmark_environment_steps(n_steps=1000):
    """Benchmark environment stepping speed."""
    print(f"\n🏃 Benchmarking environment steps ({n_steps} steps)...")
    
    env = ni.make('ChemicalReactor-v0')
    obs, _ = env.reset()
    
    start_time = time.time()
    
    for i in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    steps_per_sec = n_steps / total_time
    
    print(f"  ✓ Total time: {total_time:.4f}s")
    print(f"  ✓ Steps per second: {steps_per_sec:.0f}")
    
    return {
        'total_time': total_time,
        'steps_per_sec': steps_per_sec
    }


def benchmark_training_speed():
    """Benchmark agent training speed."""
    print(f"\n🎯 Benchmarking training speed...")
    
    env = ni.make('ChemicalReactor-v0')
    agent = ni.CQLAgent(state_dim=12, action_dim=3)
    dataset = env.get_dataset(quality='medium')
    
    # Quick training test
    start_time = time.time()
    metrics = agent.train(dataset, n_epochs=1, batch_size=128)
    end_time = time.time()
    
    training_time = end_time - start_time
    
    print(f"  ✓ Training time (1 epoch): {training_time:.4f}s")
    print(f"  ✓ Final actor loss: {metrics.get('actor_loss', 'N/A')}")
    
    return {
        'training_time': training_time,
        'metrics': str(metrics)
    }


def benchmark_inference_speed():
    """Benchmark inference speed after training."""
    print(f"\n⚡ Benchmarking inference speed...")
    
    env = ni.make('ChemicalReactor-v0')
    agent = ni.CQLAgent(state_dim=12, action_dim=3)
    dataset = env.get_dataset(quality='medium')
    
    # Train agent first
    agent.train(dataset, n_epochs=1, batch_size=128)
    
    # Test different batch sizes
    batch_sizes = [1, 10, 100]
    results = {}
    
    for batch_size in batch_sizes:
        obs = env.reset()[0]
        batch_obs = np.array([obs] * batch_size)
        
        # Warm up
        _ = agent.predict(batch_obs, deterministic=True)
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):  # 10 runs for average
            actions = agent.predict(batch_obs, deterministic=True)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        throughput = batch_size / avg_time
        
        print(f"  ✓ Batch size {batch_size}: {avg_time:.4f}s, {throughput:.0f} samples/sec")
        
        results[f'batch_{batch_size}'] = {
            'avg_time': avg_time,
            'throughput': throughput
        }
    
    return results


def run_comprehensive_benchmark():
    """Run comprehensive benchmark suite."""
    print("🚀 neoRL Industrial Performance Benchmark Suite")
    print("=" * 50)
    
    start_time = time.time()
    start_mem = measure_memory()
    
    results = {}
    
    try:
        results['environment_creation'] = benchmark_environment_creation()
        results['agent_creation'] = benchmark_agent_creation()
        results['dataset_loading'] = benchmark_dataset_loading()
        results['environment_steps'] = benchmark_environment_steps()
        results['training_speed'] = benchmark_training_speed()
        results['inference_speed'] = benchmark_inference_speed()
        
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        results['error'] = str(e)
    
    end_time = time.time()
    end_mem = measure_memory()
    
    results['benchmark_summary'] = {
        'total_time': end_time - start_time,
        'peak_memory': end_mem,
        'memory_increase': end_mem - start_mem
    }
    
    print(f"\n📈 Benchmark Summary:")
    print(f"  ✓ Total benchmark time: {end_time - start_time:.2f}s")
    print(f"  ✓ Peak memory usage: {end_mem:.2f}MB")
    print(f"  ✓ Memory increase: {end_mem - start_mem:.2f}MB")
    
    # Save results
    results_file = Path("benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"  ✓ Results saved to: {results_file}")
    
    return results


if __name__ == "__main__":
    results = run_comprehensive_benchmark()
    
    # Quick performance assessment
    summary = results.get('benchmark_summary', {})
    if summary.get('total_time', 0) < 60:
        print("\n🎉 Performance: EXCELLENT (< 1 minute)")
    elif summary.get('total_time', 0) < 180:
        print("\n✅ Performance: GOOD (< 3 minutes)")
    else:
        print("\n⚠️  Performance: NEEDS OPTIMIZATION (> 3 minutes)")