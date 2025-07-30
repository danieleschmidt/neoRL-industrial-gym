#!/usr/bin/env python3
"""
Comprehensive benchmarking suite for neoRL-industrial-gym.

This script provides performance benchmarking capabilities for:
- Environment simulation performance
- RL algorithm training speed
- JAX compilation and execution times
- Memory usage profiling
- Safety constraint validation overhead
"""

import argparse
import json
import time
import sys
import tracemalloc
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

import numpy as np
import psutil
import jax
import jax.numpy as jnp
from jax import jit


@dataclass
class BenchmarkResult:
    """Structure for benchmark results."""
    name: str
    category: str
    duration: float
    memory_peak_mb: float
    memory_current_mb: float
    cpu_percent: float
    iterations: int
    throughput: float
    metadata: Dict[str, Any]


class PerformanceBenchmarker:
    """Comprehensive performance benchmarking suite."""
    
    def __init__(self, output_dir: Path = Path("benchmark_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        self.results: List[BenchmarkResult] = []
        self.baseline_results: Optional[Dict[str, BenchmarkResult]] = None
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_baseline(self, baseline_file: Path) -> None:
        """Load baseline results for comparison."""
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
                self.baseline_results = {
                    result['name']: BenchmarkResult(**result)
                    for result in baseline_data['results']
                }
            self.logger.info(f"Loaded baseline results from {baseline_file}")
    
    def benchmark_function(
        self, 
        func, 
        name: str, 
        category: str,
        iterations: int = 100,
        warmup: int = 10,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark a function with comprehensive metrics."""
        
        # Warmup runs
        for _ in range(warmup):
            try:
                func(**kwargs)
            except Exception as e:
                self.logger.warning(f"Warmup failed for {name}: {e}")
        
        # Start monitoring
        process = psutil.Process()
        tracemalloc.start()
        
        # Benchmark runs
        start_time = time.perf_counter()
        start_cpu = process.cpu_percent()
        
        for _ in range(iterations):
            func(**kwargs)
        
        end_time = time.perf_counter()
        end_cpu = process.cpu_percent()
        
        current_memory, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate metrics
        duration = end_time - start_time
        throughput = iterations / duration if duration > 0 else 0
        avg_cpu = (start_cpu + end_cpu) / 2
        
        result = BenchmarkResult(
            name=name,
            category=category,
            duration=duration,
            memory_peak_mb=peak_memory / 1024 / 1024,
            memory_current_mb=current_memory / 1024 / 1024,
            cpu_percent=avg_cpu,
            iterations=iterations,
            throughput=throughput,
            metadata=kwargs
        )
        
        self.results.append(result)
        self.logger.info(f"Benchmarked {name}: {throughput:.2f} ops/sec")
        
        return result
    
    def benchmark_jax_compilation(self) -> None:
        """Benchmark JAX compilation performance."""
        
        def simple_computation(x):
            return jnp.sum(x ** 2)
        
        def complex_computation(x):
            for _ in range(10):
                x = jnp.sin(x) + jnp.cos(x)
            return jnp.mean(x)
        
        # Test different array sizes
        array_sizes = [100, 1000, 10000, 100000]
        
        for size in array_sizes:
            x = jnp.ones(size)
            
            # Benchmark compilation time
            start_time = time.perf_counter()
            jitted_simple = jit(simple_computation)
            jitted_simple(x).block_until_ready()
            compile_time = time.perf_counter() - start_time
            
            # Benchmark execution time
            def run_jitted():
                return jitted_simple(x).block_until_ready()
            
            self.benchmark_function(
                run_jitted,
                f"jax_simple_execution_{size}",
                "jax_performance",
                iterations=100,
                array_size=size,
                compile_time=compile_time
            )
    
    def benchmark_environment_simulation(self) -> None:
        """Benchmark environment simulation performance."""
        
        # Mock environment simulation
        def simulate_chemical_reactor(steps: int = 1000):
            """Simulate chemical reactor environment."""
            state = np.random.randn(12).astype(np.float32)
            
            for _ in range(steps):
                action = np.random.randn(3).astype(np.float32)
                # Simulate simple dynamics
                state = state + 0.1 * action[:len(state)//4*4].reshape(-1, 4).mean(axis=1)
                state = np.clip(state, -10, 10)
                
                # Safety constraint check
                temp_safe = 273 <= state[0] <= 373
                pressure_safe = 0 <= state[1] <= 10
                
                if not (temp_safe and pressure_safe):
                    break
            
            return state
        
        def simulate_robot_assembly(steps: int = 1000):
            """Simulate robot assembly environment."""
            state = np.random.randn(24).astype(np.float32)
            
            for _ in range(steps):
                action = np.random.randn(7).astype(np.float32)
                # Simulate robot dynamics
                state[:7] = state[:7] + 0.1 * action
                state[7:14] = state[7:14] + 0.05 * action
                state[14:21] = state[14:21] + 0.02 * action
                state = np.clip(state, -5, 5)
            
            return state
        
        # Benchmark different environments
        environments = [
            ("chemical_reactor", simulate_chemical_reactor, 1000),
            ("robot_assembly", simulate_robot_assembly, 1000),
        ]
        
        for env_name, sim_func, steps in environments:
            self.benchmark_function(
                sim_func,
                f"env_simulation_{env_name}",
                "environment_simulation",
                iterations=50,
                steps=steps
            )
    
    def benchmark_rl_training(self) -> None:
        """Benchmark RL algorithm training components."""
        
        def mock_policy_update(batch_size: int = 256):
            """Mock policy update computation."""
            states = jnp.array(np.random.randn(batch_size, 10))
            actions = jnp.array(np.random.randn(batch_size, 3))
            rewards = jnp.array(np.random.randn(batch_size))
            
            # Mock Q-learning update
            q_values = jnp.sum(states * actions[:, :states.shape[1]], axis=1)
            td_error = rewards - q_values
            loss = jnp.mean(td_error ** 2)
            
            return loss
        
        def mock_safety_constraint_check(batch_size: int = 256):
            """Mock safety constraint validation."""
            states = np.random.randn(batch_size, 12)
            
            violations = 0
            for state in states:
                # Temperature constraint
                if not (273 <= state[0] <= 373):
                    violations += 1
                # Pressure constraint  
                if not (0 <= state[1] <= 10):
                    violations += 1
            
            return violations / len(states)
        
        # Benchmark training components
        batch_sizes = [64, 256, 1024]
        
        for batch_size in batch_sizes:
            self.benchmark_function(
                mock_policy_update,
                f"policy_update_batch_{batch_size}",
                "rl_training",
                iterations=100,
                batch_size=batch_size
            )
            
            self.benchmark_function(
                mock_safety_constraint_check,
                f"safety_check_batch_{batch_size}",
                "safety_validation",
                iterations=100,
                batch_size=batch_size
            )
    
    def benchmark_memory_usage(self) -> None:
        """Benchmark memory usage patterns."""
        
        def create_large_arrays(size: int = 1000000):
            """Create and manipulate large arrays."""
            arrays = []
            for i in range(10):
                arr = np.random.randn(size // 10).astype(np.float32)
                arrays.append(arr)
            
            # Simulate processing
            result = np.concatenate(arrays)
            processed = np.fft.fft(result)
            return np.real(processed).mean()
        
        memory_sizes = [100000, 1000000, 10000000]
        
        for size in memory_sizes:
            self.benchmark_function(
                create_large_arrays,
                f"memory_usage_{size//1000}k",
                "memory_performance",
                iterations=10,
                size=size
            )
    
    def run_all_benchmarks(self) -> None:
        """Run all benchmark suites."""
        self.logger.info("Starting comprehensive benchmark suite...")
        
        try:
            self.benchmark_jax_compilation()
            self.benchmark_environment_simulation()
            self.benchmark_rl_training()
            self.benchmark_memory_usage()
        except Exception as e:
            self.logger.error(f"Benchmark suite failed: {e}")
            traceback.print_exc()
    
    def compare_with_baseline(self) -> Dict[str, Any]:
        """Compare current results with baseline."""
        if not self.baseline_results:
            self.logger.warning("No baseline results loaded")
            return {}
        
        comparisons = {}
        
        for result in self.results:
            baseline = self.baseline_results.get(result.name)
            if baseline:
                throughput_change = (
                    (result.throughput - baseline.throughput) / baseline.throughput * 100
                )
                memory_change = (
                    (result.memory_peak_mb - baseline.memory_peak_mb) / baseline.memory_peak_mb * 100
                )
                
                comparisons[result.name] = {
                    "throughput_change_percent": throughput_change,
                    "memory_change_percent": memory_change,
                    "performance_regression": throughput_change < -10,  # >10% slower
                    "memory_regression": memory_change > 20,  # >20% more memory
                }
        
        return comparisons
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        
        # Group results by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        # Calculate category statistics
        category_stats = {}
        for category, results in categories.items():
            throughputs = [r.throughput for r in results]
            memory_peaks = [r.memory_peak_mb for r in results]
            
            category_stats[category] = {
                "count": len(results),
                "avg_throughput": np.mean(throughputs),
                "max_throughput": np.max(throughputs),
                "avg_memory_peak": np.mean(memory_peaks),
                "max_memory_peak": np.max(memory_peaks),
            }
        
        # System information
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "jax_backend": jax.default_backend(),
            "jax_devices": len(jax.devices()),
        }
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": system_info,
            "category_stats": category_stats,
            "results": [asdict(result) for result in self.results],
            "comparisons": self.compare_with_baseline(),
        }
        
        return report
    
    def save_results(self, filename: str = None) -> Path:
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        output_file = self.output_dir / filename
        report = self.generate_report()
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_file}")
        return output_file
    
    def print_summary(self) -> None:
        """Print benchmark summary to console."""
        
        if not self.results:
            print("No benchmark results available")
            return
        
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        
        # Group by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        for category, results in categories.items():
            print(f"\n{category.upper().replace('_', ' ')}")
            print("-" * 40)
            
            for result in sorted(results, key=lambda x: x.throughput, reverse=True):
                print(f"  {result.name:30} {result.throughput:8.2f} ops/sec  "
                      f"{result.memory_peak_mb:6.1f} MB")
        
        # Performance regressions
        comparisons = self.compare_with_baseline()
        if comparisons:
            regressions = [
                name for name, comp in comparisons.items()
                if comp.get('performance_regression') or comp.get('memory_regression')
            ]
            
            if regressions:
                print(f"\n⚠️  PERFORMANCE REGRESSIONS DETECTED:")
                for name in regressions:
                    comp = comparisons[name]
                    print(f"  {name}: "
                          f"throughput {comp['throughput_change_percent']:+.1f}%, "
                          f"memory {comp['memory_change_percent']:+.1f}%")
        
        print("\n" + "="*80)


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="neoRL-industrial-gym benchmark suite")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results"),
                       help="Output directory for results")
    parser.add_argument("--baseline", type=Path,
                       help="Baseline results file for comparison")
    parser.add_argument("--compare-baseline", action="store_true",
                       help="Compare with baseline and exit with error on regression")
    parser.add_argument("--save-baseline", action="store_true",
                       help="Save results as new baseline")
    parser.add_argument("--categories", nargs="+", 
                       choices=["jax", "environment", "rl", "memory", "all"],
                       default=["all"],
                       help="Benchmark categories to run")
    
    args = parser.parse_args()
    
    # Initialize benchmarker
    benchmarker = PerformanceBenchmarker(args.output_dir)
    
    # Load baseline if provided
    if args.baseline:
        benchmarker.load_baseline(args.baseline)
    
    # Run benchmarks
    if "all" in args.categories:
        benchmarker.run_all_benchmarks()
    else:
        if "jax" in args.categories:
            benchmarker.benchmark_jax_compilation()
        if "environment" in args.categories:
            benchmarker.benchmark_environment_simulation()
        if "rl" in args.categories:
            benchmarker.benchmark_rl_training()
        if "memory" in args.categories:
            benchmarker.benchmark_memory_usage()
    
    # Save results
    results_file = benchmarker.save_results()
    
    # Save as baseline if requested
    if args.save_baseline:
        baseline_file = args.output_dir / "baseline_results.json"
        results_file.rename(baseline_file)
        print(f"Saved as baseline: {baseline_file}")
    
    # Print summary
    benchmarker.print_summary()
    
    # Check for regressions
    if args.compare_baseline and benchmarker.baseline_results:
        comparisons = benchmarker.compare_with_baseline()
        regressions = [
            name for name, comp in comparisons.items()
            if comp.get('performance_regression') or comp.get('memory_regression')
        ]
        
        if regressions:
            print(f"\n❌ Performance regressions detected in {len(regressions)} benchmarks")
            sys.exit(1)
        else:
            print(f"\n✅ No performance regressions detected")
            sys.exit(0)


if __name__ == "__main__":
    main()