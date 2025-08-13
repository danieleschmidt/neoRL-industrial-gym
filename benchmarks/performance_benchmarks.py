"""Performance benchmarks for neoRL-industrial-gym."""

import time
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import neorl_industrial as ni
from neorl_industrial.monitoring.dashboard import record_metric

@dataclass
class BenchmarkResult:
    """Benchmark result data."""
    name: str
    duration: float
    throughput: float
    memory_peak_mb: float
    memory_avg_mb: float
    iterations: int
    metadata: Dict[str, Any]

class PerformanceBenchmark:
    """Performance benchmark suite for neoRL-industrial."""
    
    def __init__(self, output_dir: Path = None):
        """Initialize benchmark suite."""
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = []
        self.logger = logging.getLogger("performance_benchmark")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all performance benchmarks."""
        self.logger.info("🚀 Starting performance benchmarks...")
        
        benchmarks = [
            ("Environment Creation", self.benchmark_environment_creation),
            ("Environment Steps", self.benchmark_environment_steps),
            ("Dataset Generation", self.benchmark_dataset_generation),
            ("Agent Creation", self.benchmark_agent_creation),
            ("Agent Training", self.benchmark_agent_training),
            ("Agent Inference", self.benchmark_agent_inference),
            ("Monitoring System", self.benchmark_monitoring_system),
            ("Caching System", self.benchmark_caching_system),
            ("Validation System", self.benchmark_validation_system),
            ("Memory Usage", self.benchmark_memory_usage),
            ("Concurrent Operations", self.benchmark_concurrent_operations),
        ]
        
        for name, benchmark_func in benchmarks:
            self.logger.info(f"📊 Running {name} benchmark...")
            try:
                result = benchmark_func()
                self.results.append(result)
                self.logger.info(f"✅ {name}: {result.throughput:.2f} ops/sec, {result.duration:.3f}s")
            except Exception as e:
                self.logger.error(f"❌ {name} failed: {e}")
                # Create failed result
                failed_result = BenchmarkResult(
                    name=name,
                    duration=0.0,
                    throughput=0.0,
                    memory_peak_mb=0.0,
                    memory_avg_mb=0.0,
                    iterations=0,
                    metadata={"error": str(e)}
                )
                self.results.append(failed_result)
        
        # Generate report
        self.generate_report()
        
        return self.results
    
    def benchmark_environment_creation(self) -> BenchmarkResult:
        """Benchmark environment creation speed."""
        env_types = ['ChemicalReactor-v0', 'PowerGrid-v0', 'RobotAssembly-v0']
        iterations = 50
        
        memory_tracker = []
        
        start_time = time.time()
        memory_start = psutil.Process().memory_info().rss / 1024 / 1024
        
        for i in range(iterations):
            for env_type in env_types:
                env = ni.make(env_type)
                memory_tracker.append(psutil.Process().memory_info().rss / 1024 / 1024)
                del env  # Explicit cleanup
                gc.collect()
        
        end_time = time.time()
        memory_end = psutil.Process().memory_info().rss / 1024 / 1024
        
        duration = end_time - start_time
        total_operations = iterations * len(env_types)
        throughput = total_operations / duration
        
        return BenchmarkResult(
            name="Environment Creation",
            duration=duration,
            throughput=throughput,
            memory_peak_mb=max(memory_tracker),
            memory_avg_mb=np.mean(memory_tracker),
            iterations=total_operations,
            metadata={
                "env_types": env_types,
                "memory_growth_mb": memory_end - memory_start,
            }
        )
    
    def benchmark_environment_steps(self) -> BenchmarkResult:
        """Benchmark environment step performance."""
        env = ni.make('ChemicalReactor-v0')
        env.reset()
        
        iterations = 1000
        memory_tracker = []
        step_times = []
        
        start_time = time.time()
        
        for i in range(iterations):
            step_start = time.time()
            
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_times.append(time.time() - step_start)
            memory_tracker.append(psutil.Process().memory_info().rss / 1024 / 1024)
            
            if terminated or truncated:
                env.reset()
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = iterations / duration
        
        return BenchmarkResult(
            name="Environment Steps",
            duration=duration,
            throughput=throughput,
            memory_peak_mb=max(memory_tracker),
            memory_avg_mb=np.mean(memory_tracker),
            iterations=iterations,
            metadata={
                "avg_step_time": np.mean(step_times),
                "min_step_time": np.min(step_times),
                "max_step_time": np.max(step_times),
                "step_time_std": np.std(step_times),
            }
        )
    
    def benchmark_dataset_generation(self) -> BenchmarkResult:
        """Benchmark dataset generation performance."""
        env = ni.make('ChemicalReactor-v0')
        
        qualities = ['expert', 'medium', 'mixed', 'random']
        memory_tracker = []
        generation_times = []
        
        start_time = time.time()
        
        for quality in qualities:
            gen_start = time.time()
            
            dataset = env.get_dataset(quality=quality)
            
            generation_times.append(time.time() - gen_start)
            memory_tracker.append(psutil.Process().memory_info().rss / 1024 / 1024)
            
            # Validate dataset
            assert len(dataset['observations']) > 0
            assert len(dataset['actions']) > 0
            
            del dataset  # Cleanup
            gc.collect()
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = len(qualities) / duration
        
        return BenchmarkResult(
            name="Dataset Generation",
            duration=duration,
            throughput=throughput,
            memory_peak_mb=max(memory_tracker),
            memory_avg_mb=np.mean(memory_tracker),
            iterations=len(qualities),
            metadata={
                "qualities_tested": qualities,
                "avg_generation_time": np.mean(generation_times),
                "generation_times": generation_times,
            }
        )
    
    def benchmark_agent_creation(self) -> BenchmarkResult:
        """Benchmark agent creation performance."""
        agent_types = [ni.CQLAgent, ni.IQLAgent, ni.TD3BCAgent]
        state_dims = [12, 32, 24]  # Different environment dimensions
        action_dims = [3, 8, 7]
        
        iterations = 20
        memory_tracker = []
        creation_times = []
        
        start_time = time.time()
        
        for i in range(iterations):
            for agent_type, state_dim, action_dim in zip(agent_types, state_dims, action_dims):
                create_start = time.time()
                
                agent = agent_type(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    safety_critic=True
                )
                
                creation_times.append(time.time() - create_start)
                memory_tracker.append(psutil.Process().memory_info().rss / 1024 / 1024)
                
                del agent  # Cleanup
                gc.collect()
        
        end_time = time.time()
        duration = end_time - start_time
        total_operations = iterations * len(agent_types)
        throughput = total_operations / duration
        
        return BenchmarkResult(
            name="Agent Creation",
            duration=duration,
            throughput=throughput,
            memory_peak_mb=max(memory_tracker),
            memory_avg_mb=np.mean(memory_tracker),
            iterations=total_operations,
            metadata={
                "agent_types": [cls.__name__ for cls in agent_types],
                "avg_creation_time": np.mean(creation_times),
                "configurations_tested": len(agent_types),
            }
        )
    
    def benchmark_agent_training(self) -> BenchmarkResult:
        """Benchmark agent training performance."""
        env = ni.make('ChemicalReactor-v0')
        agent = ni.CQLAgent(state_dim=12, action_dim=3)
        
        # Create small dataset for training
        dataset = env.get_dataset(quality='random')
        
        # Use subset for benchmark
        subset_size = min(1000, len(dataset['observations']))
        small_dataset = {
            key: values[:subset_size] for key, values in dataset.items()
        }
        
        memory_tracker = []
        
        start_time = time.time()
        memory_start = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Train for a few epochs
        training_result = agent.train(
            small_dataset,
            n_epochs=5,
            batch_size=64,
            eval_freq=10  # No evaluation for speed
        )
        
        end_time = time.time()
        memory_end = psutil.Process().memory_info().rss / 1024 / 1024
        
        duration = end_time - start_time
        samples_processed = subset_size * 5  # 5 epochs
        throughput = samples_processed / duration
        
        return BenchmarkResult(
            name="Agent Training",
            duration=duration,
            throughput=throughput,
            memory_peak_mb=memory_end,
            memory_avg_mb=(memory_start + memory_end) / 2,
            iterations=5,  # epochs
            metadata={
                "samples_processed": samples_processed,
                "dataset_size": subset_size,
                "batch_size": 64,
                "memory_growth_mb": memory_end - memory_start,
                "final_loss": training_result.get('final_metrics', {}).get('loss', 0),
            }
        )
    
    def benchmark_agent_inference(self) -> BenchmarkResult:
        """Benchmark agent inference performance."""
        env = ni.make('ChemicalReactor-v0')
        agent = ni.CQLAgent(state_dim=12, action_dim=3)
        
        # Quick training to enable inference
        dataset = env.get_dataset(quality='random')
        subset_size = min(500, len(dataset['observations']))
        small_dataset = {
            key: values[:subset_size] for key, values in dataset.items()
        }
        agent.train(small_dataset, n_epochs=1, batch_size=32)
        
        # Benchmark inference
        iterations = 1000
        obs = env.reset()[0]
        
        inference_times = []
        memory_tracker = []
        
        start_time = time.time()
        
        for i in range(iterations):
            inf_start = time.time()
            
            action = agent.predict(obs, deterministic=True)
            
            inference_times.append(time.time() - inf_start)
            memory_tracker.append(psutil.Process().memory_info().rss / 1024 / 1024)
            
            # Vary observations
            obs = np.random.randn(12).astype(np.float32)
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = iterations / duration
        
        return BenchmarkResult(
            name="Agent Inference",
            duration=duration,
            throughput=throughput,
            memory_peak_mb=max(memory_tracker),
            memory_avg_mb=np.mean(memory_tracker),
            iterations=iterations,
            metadata={
                "avg_inference_time_ms": np.mean(inference_times) * 1000,
                "min_inference_time_ms": np.min(inference_times) * 1000,
                "max_inference_time_ms": np.max(inference_times) * 1000,
                "inference_std_ms": np.std(inference_times) * 1000,
            }
        )
    
    def benchmark_monitoring_system(self) -> BenchmarkResult:
        """Benchmark monitoring system performance."""
        from neorl_industrial.monitoring.dashboard import MonitoringDashboard, SystemMetric
        
        dashboard = MonitoringDashboard()
        iterations = 1000
        
        memory_tracker = []
        record_times = []
        
        start_time = time.time()
        
        for i in range(iterations):
            record_start = time.time()
            
            metric = SystemMetric(
                name=f"test_metric_{i % 10}",
                value=np.random.random(),
                timestamp=time.time(),
                tags={"component": "benchmark", "iteration": str(i)}
            )
            
            dashboard.record_metric(metric)
            
            record_times.append(time.time() - record_start)
            memory_tracker.append(psutil.Process().memory_info().rss / 1024 / 1024)
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = iterations / duration
        
        return BenchmarkResult(
            name="Monitoring System",
            duration=duration,
            throughput=throughput,
            memory_peak_mb=max(memory_tracker),
            memory_avg_mb=np.mean(memory_tracker),
            iterations=iterations,
            metadata={
                "avg_record_time_ms": np.mean(record_times) * 1000,
                "metrics_stored": len(dashboard.metrics_history),
                "dashboard_size": len(dashboard.metrics_history),
            }
        )
    
    def benchmark_caching_system(self) -> BenchmarkResult:
        """Benchmark caching system performance."""
        from neorl_industrial.optimization.intelligent_caching import IntelligentCache
        
        cache = IntelligentCache(max_size_mb=100, max_entries=1000)
        iterations = 1000
        
        memory_tracker = []
        operation_times = []
        
        # Test data
        test_data = {
            f"key_{i}": np.random.randn(100, 100) for i in range(50)
        }
        
        start_time = time.time()
        
        for i in range(iterations):
            op_start = time.time()
            
            if i % 3 == 0:  # Put operation
                key = f"key_{i % 50}"
                value = test_data[key]
                cache.put(key, value)
            else:  # Get operation
                key = f"key_{i % 50}"
                value = cache.get(key)
            
            operation_times.append(time.time() - op_start)
            memory_tracker.append(psutil.Process().memory_info().rss / 1024 / 1024)
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = iterations / duration
        
        cache_stats = cache.get_stats()
        
        return BenchmarkResult(
            name="Caching System",
            duration=duration,
            throughput=throughput,
            memory_peak_mb=max(memory_tracker),
            memory_avg_mb=np.mean(memory_tracker),
            iterations=iterations,
            metadata={
                "avg_operation_time_ms": np.mean(operation_times) * 1000,
                "cache_hit_rate": cache_stats["hit_rate"],
                "cache_size_mb": cache_stats["size_mb"],
                "cache_entries": cache_stats["entries"],
            }
        )
    
    def benchmark_validation_system(self) -> BenchmarkResult:
        """Benchmark validation system performance."""
        from neorl_industrial.validation.comprehensive_validator import create_industrial_env_validator
        from neorl_industrial.core.types import SafetyConstraint
        
        # Create test constraint
        def test_constraint(state, action):
            return np.all(state >= -1) and np.all(state <= 1)
        
        constraint = SafetyConstraint(
            name="test_constraint",
            check_fn=test_constraint,
            penalty=-10.0
        )
        
        validator = create_industrial_env_validator(
            state_dim=12,
            action_dim=3,
            safety_constraints=[constraint]
        )
        
        iterations = 100
        memory_tracker = []
        validation_times = []
        
        start_time = time.time()
        
        for i in range(iterations):
            val_start = time.time()
            
            # Create test data
            test_data = {
                "observations": np.random.uniform(-0.5, 0.5, (10, 12)),
                "actions": np.random.uniform(-0.5, 0.5, (10, 3)),
            }
            
            results = validator.validate_all(test_data)
            
            validation_times.append(time.time() - val_start)
            memory_tracker.append(psutil.Process().memory_info().rss / 1024 / 1024)
        
        end_time = time.time()
        duration = end_time - start_time
        throughput = iterations / duration
        
        return BenchmarkResult(
            name="Validation System",
            duration=duration,
            throughput=throughput,
            memory_peak_mb=max(memory_tracker),
            memory_avg_mb=np.mean(memory_tracker),
            iterations=iterations,
            metadata={
                "avg_validation_time_ms": np.mean(validation_times) * 1000,
                "validators_used": len(validator.validators),
            }
        )
    
    def benchmark_memory_usage(self) -> BenchmarkResult:
        """Benchmark memory usage patterns."""
        memory_measurements = []
        operations = []
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Baseline measurement
        memory_measurements.append(initial_memory)
        operations.append("baseline")
        
        # Environment creation
        env = ni.make('ChemicalReactor-v0')
        memory_measurements.append(psutil.Process().memory_info().rss / 1024 / 1024)
        operations.append("env_creation")
        
        # Dataset generation
        dataset = env.get_dataset(quality='mixed')
        memory_measurements.append(psutil.Process().memory_info().rss / 1024 / 1024)
        operations.append("dataset_generation")
        
        # Agent creation
        agent = ni.CQLAgent(state_dim=12, action_dim=3)
        memory_measurements.append(psutil.Process().memory_info().rss / 1024 / 1024)
        operations.append("agent_creation")
        
        # Training
        subset_size = min(500, len(dataset['observations']))
        small_dataset = {key: values[:subset_size] for key, values in dataset.items()}
        agent.train(small_dataset, n_epochs=2, batch_size=32)
        memory_measurements.append(psutil.Process().memory_info().rss / 1024 / 1024)
        operations.append("training")
        
        # Cleanup
        del dataset, small_dataset, agent, env
        gc.collect()
        memory_measurements.append(psutil.Process().memory_info().rss / 1024 / 1024)
        operations.append("cleanup")
        
        end_time = time.time()
        duration = end_time - start_time
        
        peak_memory = max(memory_measurements)
        memory_growth = memory_measurements[-1] - memory_measurements[0]
        
        return BenchmarkResult(
            name="Memory Usage",
            duration=duration,
            throughput=len(operations) / duration,
            memory_peak_mb=peak_memory,
            memory_avg_mb=np.mean(memory_measurements),
            iterations=len(operations),
            metadata={
                "memory_timeline": list(zip(operations, memory_measurements)),
                "memory_growth_mb": memory_growth,
                "peak_memory_mb": peak_memory,
                "cleanup_effectiveness": memory_measurements[0] - memory_measurements[-1],
            }
        )
    
    def benchmark_concurrent_operations(self) -> BenchmarkResult:
        """Benchmark concurrent operations performance."""
        iterations = 50
        num_workers = 4
        
        def worker_task(worker_id: int, task_count: int) -> Dict[str, Any]:
            """Task for concurrent execution."""
            env = ni.make('ChemicalReactor-v0')
            results = []
            
            for i in range(task_count):
                env.reset()
                
                for j in range(10):  # 10 steps per task
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if terminated or truncated:
                        break
                
                results.append({"worker": worker_id, "task": i, "steps": j+1})
            
            return {
                "worker_id": worker_id,
                "completed_tasks": len(results),
                "total_steps": sum(r["steps"] for r in results),
            }
        
        memory_tracker = []
        
        start_time = time.time()
        memory_start = psutil.Process().memory_info().rss / 1024 / 1024
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            tasks_per_worker = iterations // num_workers
            
            futures = [
                executor.submit(worker_task, worker_id, tasks_per_worker)
                for worker_id in range(num_workers)
            ]
            
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                memory_tracker.append(psutil.Process().memory_info().rss / 1024 / 1024)
        
        end_time = time.time()
        memory_end = psutil.Process().memory_info().rss / 1024 / 1024
        
        duration = end_time - start_time
        total_tasks = sum(r["completed_tasks"] for r in results)
        total_steps = sum(r["total_steps"] for r in results)
        throughput = total_tasks / duration
        
        return BenchmarkResult(
            name="Concurrent Operations",
            duration=duration,
            throughput=throughput,
            memory_peak_mb=max(memory_tracker) if memory_tracker else memory_end,
            memory_avg_mb=np.mean(memory_tracker) if memory_tracker else memory_end,
            iterations=total_tasks,
            metadata={
                "num_workers": num_workers,
                "total_steps": total_steps,
                "avg_steps_per_task": total_steps / total_tasks if total_tasks > 0 else 0,
                "memory_growth_mb": memory_end - memory_start,
                "parallel_efficiency": throughput / num_workers,
            }
        )
    
    def generate_report(self) -> None:
        """Generate comprehensive benchmark report."""
        # Create text report
        report_text = self._generate_text_report()
        
        # Save text report
        text_file = self.output_dir / "benchmark_report.txt"
        with open(text_file, "w") as f:
            f.write(report_text)
        
        # Save JSON results
        json_data = {
            "timestamp": time.time(),
            "results": [
                {
                    "name": r.name,
                    "duration": r.duration,
                    "throughput": r.throughput,
                    "memory_peak_mb": r.memory_peak_mb,
                    "memory_avg_mb": r.memory_avg_mb,
                    "iterations": r.iterations,
                    "metadata": r.metadata,
                }
                for r in self.results
            ]
        }
        
        json_file = self.output_dir / "benchmark_results.json"
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=2, default=str)
        
        # Generate plots
        self._generate_plots()
        
        self.logger.info(f"📊 Benchmark report saved to {self.output_dir}")
    
    def _generate_text_report(self) -> str:
        """Generate text report."""
        report = ["="*80]
        report.append("🚀 neoRL-industrial-gym Performance Benchmark Report")
        report.append("="*80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total benchmarks: {len(self.results)}")
        report.append("")
        
        # Summary table
        report.append("📊 Performance Summary:")
        report.append("-" * 80)
        report.append(f"{'Benchmark':<25} {'Throughput':<15} {'Duration':<12} {'Memory Peak':<12}")
        report.append("-" * 80)
        
        for result in self.results:
            report.append(
                f"{result.name:<25} "
                f"{result.throughput:>10.2f} ops/s "
                f"{result.duration:>8.3f}s "
                f"{result.memory_peak_mb:>8.1f} MB"
            )
        
        report.append("")
        
        # Detailed results
        for result in self.results:
            report.append(f"🔍 {result.name}")
            report.append("-" * 40)
            report.append(f"  Duration: {result.duration:.3f} seconds")
            report.append(f"  Throughput: {result.throughput:.2f} operations/second")
            report.append(f"  Iterations: {result.iterations}")
            report.append(f"  Memory Peak: {result.memory_peak_mb:.1f} MB")
            report.append(f"  Memory Average: {result.memory_avg_mb:.1f} MB")
            
            if result.metadata:
                report.append("  Metadata:")
                for key, value in result.metadata.items():
                    if isinstance(value, float):
                        report.append(f"    {key}: {value:.3f}")
                    else:
                        report.append(f"    {key}: {value}")
            
            report.append("")
        
        return "\n".join(report)
    
    def _generate_plots(self) -> None:
        """Generate performance plots."""
        try:
            import matplotlib.pyplot as plt
            
            # Throughput comparison
            names = [r.name for r in self.results]
            throughputs = [r.throughput for r in self.results]
            
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.bar(range(len(names)), throughputs)
            plt.xticks(range(len(names)), names, rotation=45, ha='right')
            plt.ylabel('Throughput (ops/sec)')
            plt.title('Benchmark Throughput Comparison')
            plt.tight_layout()
            
            # Memory usage
            memory_peaks = [r.memory_peak_mb for r in self.results]
            
            plt.subplot(2, 2, 2)
            plt.bar(range(len(names)), memory_peaks, color='orange')
            plt.xticks(range(len(names)), names, rotation=45, ha='right')
            plt.ylabel('Peak Memory (MB)')
            plt.title('Memory Usage Comparison')
            plt.tight_layout()
            
            # Duration comparison
            durations = [r.duration for r in self.results]
            
            plt.subplot(2, 2, 3)
            plt.bar(range(len(names)), durations, color='green')
            plt.xticks(range(len(names)), names, rotation=45, ha='right')
            plt.ylabel('Duration (seconds)')
            plt.title('Execution Time Comparison')
            plt.tight_layout()
            
            # Combined efficiency plot (throughput / memory)
            efficiency = [t / m if m > 0 else 0 for t, m in zip(throughputs, memory_peaks)]
            
            plt.subplot(2, 2, 4)
            plt.bar(range(len(names)), efficiency, color='purple')
            plt.xticks(range(len(names)), names, rotation=45, ha='right')
            plt.ylabel('Efficiency (ops/sec/MB)')
            plt.title('Memory Efficiency')
            plt.tight_layout()
            
            plt.savefig(self.output_dir / "benchmark_plots.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            self.logger.warning("Matplotlib not available, skipping plots")
        except Exception as e:
            self.logger.error(f"Plot generation failed: {e}")

def main():
    """Run performance benchmarks."""
    output_dir = Path("benchmark_results")
    
    benchmark = PerformanceBenchmark(output_dir)
    results = benchmark.run_all_benchmarks()
    
    print("\n" + "="*80)
    print("🎯 BENCHMARK SUMMARY")
    print("="*80)
    print(f"Total benchmarks: {len(results)}")
    print(f"Results saved to: {output_dir}")
    
    # Print top performers
    successful_results = [r for r in results if r.throughput > 0]
    if successful_results:
        top_throughput = max(successful_results, key=lambda r: r.throughput)
        most_efficient = max(successful_results, key=lambda r: r.throughput / max(r.memory_peak_mb, 1))
        
        print(f"🚀 Highest throughput: {top_throughput.name} ({top_throughput.throughput:.2f} ops/sec)")
        print(f"⚡ Most efficient: {most_efficient.name} ({most_efficient.throughput / max(most_efficient.memory_peak_mb, 1):.3f} ops/sec/MB)")
    
    print("="*80)

if __name__ == "__main__":
    main()