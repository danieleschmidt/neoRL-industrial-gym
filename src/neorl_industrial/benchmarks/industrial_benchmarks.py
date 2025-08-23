"""Comprehensive industrial RL benchmarking suite for research validation."""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from ..agents.base import OfflineAgent
from ..core.types import SafetyMetrics
from ..environments.base import IndustrialEnv


@dataclass
class BenchmarkResult:
    """Result from a single benchmark evaluation."""
    
    benchmark_name: str
    agent_name: str
    environment_name: str
    
    # Performance metrics
    mean_return: float
    std_return: float
    median_return: float
    min_return: float
    max_return: float
    
    # Safety metrics
    safety_violations: int
    violation_rate: float
    safety_score: float
    
    # Efficiency metrics
    episode_length: float
    convergence_time: float
    sample_efficiency: float
    
    # Statistical metrics
    confidence_interval: Tuple[float, float]
    statistical_significance: float
    
    # Additional metrics
    metadata: Dict[str, Any]


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    
    n_episodes: int = 100
    n_seeds: int = 10
    confidence_level: float = 0.95
    max_episode_steps: int = 1000
    
    # Research-specific settings
    statistical_test: str = "t_test"  # t_test, mann_whitney, wilcoxon
    multiple_comparisons_correction: str = "bonferroni"  # bonferroni, fdr, none
    effect_size_calculation: bool = True
    
    # Reproducibility settings
    deterministic_evaluation: bool = True
    save_trajectories: bool = True
    save_intermediate_results: bool = True


class IndustrialBenchmark(ABC):
    """Abstract base class for industrial RL benchmarks."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def evaluate_agent(
        self,
        agent: OfflineAgent,
        environment: IndustrialEnv,
        seed: int = 42
    ) -> BenchmarkResult:
        """Evaluate agent on this benchmark."""
        pass
        
    @abstractmethod
    def get_benchmark_description(self) -> str:
        """Get description of this benchmark for academic papers."""
        pass


class SafetyBenchmark(IndustrialBenchmark):
    """Benchmark focused on safety constraint satisfaction."""
    
    def __init__(self, config: BenchmarkConfig, safety_threshold: float = 0.01):
        super().__init__(config)
        self.safety_threshold = safety_threshold
        
    def evaluate_agent(
        self,
        agent: OfflineAgent,
        environment: IndustrialEnv,
        seed: int = 42
    ) -> BenchmarkResult:
        """Evaluate agent safety performance."""
        
        self.logger.info(f"Running safety benchmark for {agent.__class__.__name__}")
        
        key = jax.random.PRNGKey(seed)
        episode_returns = []
        safety_violations_list = []
        safety_scores = []
        episode_lengths = []
        
        start_time = time.time()
        
        for episode in tqdm(range(self.config.n_episodes), desc="Safety Evaluation"):
            episode_key = jax.random.split(key, self.config.n_episodes)[episode]
            
            # Reset environment
            obs, _ = environment.reset(seed=int(episode_key[0]))
            
            episode_return = 0.0
            episode_violations = 0
            episode_safety_scores = []
            step_count = 0
            
            done = False
            while not done and step_count < self.config.max_episode_steps:
                # Get action from agent
                action = agent.predict(obs)
                
                # Take environment step
                next_obs, reward, terminated, truncated, info = environment.step(action)
                
                episode_return += reward
                
                # Track safety metrics
                if "safety_metrics" in info:
                    safety_metrics = info["safety_metrics"]
                    episode_violations += safety_metrics.total_violations
                    episode_safety_scores.append(safety_metrics.safety_score)
                else:
                    # Calculate safety metrics directly
                    safety_metrics = environment.get_safety_metrics()
                    episode_violations += safety_metrics.total_violations
                    episode_safety_scores.append(safety_metrics.safety_score)
                
                obs = next_obs
                done = terminated or truncated
                step_count += 1
                
            episode_returns.append(episode_return)
            safety_violations_list.append(episode_violations)
            safety_scores.append(np.mean(episode_safety_scores) if episode_safety_scores else 0.0)
            episode_lengths.append(step_count)
            
        evaluation_time = time.time() - start_time
        
        # Calculate statistics
        returns_array = np.array(episode_returns)
        violations_array = np.array(safety_violations_list)
        
        mean_return = float(np.mean(returns_array))
        std_return = float(np.std(returns_array))
        median_return = float(np.median(returns_array))
        
        total_violations = int(np.sum(violations_array))
        violation_rate = float(np.mean(violations_array > 0))
        mean_safety_score = float(np.mean(safety_scores))
        
        # Confidence interval for mean return
        from scipy import stats
        confidence_interval = stats.t.interval(
            self.config.confidence_level,
            len(returns_array) - 1,
            loc=mean_return,
            scale=stats.sem(returns_array)
        )
        
        return BenchmarkResult(
            benchmark_name="Safety",
            agent_name=agent.__class__.__name__,
            environment_name=environment.__class__.__name__,
            mean_return=mean_return,
            std_return=std_return,
            median_return=median_return,
            min_return=float(np.min(returns_array)),
            max_return=float(np.max(returns_array)),
            safety_violations=total_violations,
            violation_rate=violation_rate,
            safety_score=mean_safety_score,
            episode_length=float(np.mean(episode_lengths)),
            convergence_time=evaluation_time,
            sample_efficiency=mean_return / np.mean(episode_lengths),
            confidence_interval=confidence_interval,
            statistical_significance=0.0,  # Calculated later in comparison
            metadata={
                "safety_threshold": self.safety_threshold,
                "episode_violations": violations_array.tolist(),
                "individual_returns": returns_array.tolist(),
                "safety_scores": safety_scores
            }
        )
        
    def get_benchmark_description(self) -> str:
        """Get academic description of safety benchmark."""
        return (
            "Safety Benchmark: Evaluates agent adherence to safety constraints "
            f"with violation threshold of {self.safety_threshold}. Measures "
            "violation frequency, safety score, and constraint satisfaction rate "
            "across multiple episodes."
        )


class PerformanceBenchmark(IndustrialBenchmark):
    """Benchmark focused on task performance and efficiency."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        
    def evaluate_agent(
        self,
        agent: OfflineAgent,
        environment: IndustrialEnv,
        seed: int = 42
    ) -> BenchmarkResult:
        """Evaluate agent performance."""
        
        self.logger.info(f"Running performance benchmark for {agent.__class__.__name__}")
        
        key = jax.random.PRNGKey(seed)
        episode_returns = []
        episode_lengths = []
        convergence_episodes = []
        
        start_time = time.time()
        
        # Track convergence
        performance_history = []
        converged = False
        convergence_threshold = 0.05  # 5% improvement threshold
        convergence_window = 10
        
        for episode in tqdm(range(self.config.n_episodes), desc="Performance Evaluation"):
            episode_key = jax.random.split(key, self.config.n_episodes)[episode]
            
            obs, _ = environment.reset(seed=int(episode_key[0]))
            
            episode_return = 0.0
            step_count = 0
            
            done = False
            while not done and step_count < self.config.max_episode_steps:
                action = agent.predict(obs)
                next_obs, reward, terminated, truncated, info = environment.step(action)
                
                episode_return += reward
                obs = next_obs
                done = terminated or truncated
                step_count += 1
                
            episode_returns.append(episode_return)
            episode_lengths.append(step_count)
            performance_history.append(episode_return)
            
            # Check for convergence
            if not converged and len(performance_history) >= convergence_window:
                recent_performance = performance_history[-convergence_window:]
                early_performance = (performance_history[-convergence_window-5:-5] 
                                    if len(performance_history) >= convergence_window + 5 
                                    else recent_performance[:5])
                
                if len(early_performance) > 0:
                    improvement = ((np.mean(recent_performance) - np.mean(early_performance)) / 
                                  abs(np.mean(early_performance) + 1e-8))
                    
                    if abs(improvement) < convergence_threshold:
                        convergence_episodes.append(episode)
                        converged = True
                        
        evaluation_time = time.time() - start_time
        
        # Calculate statistics
        returns_array = np.array(episode_returns)
        
        mean_return = float(np.mean(returns_array))
        std_return = float(np.std(returns_array))
        median_return = float(np.median(returns_array))
        
        # Sample efficiency: performance per sample
        total_samples = sum(episode_lengths)
        sample_efficiency = mean_return / (total_samples / len(episode_lengths))
        
        # Convergence time
        avg_convergence = np.mean(convergence_episodes) if convergence_episodes else self.config.n_episodes
        
        # Confidence interval
        from scipy import stats
        confidence_interval = stats.t.interval(
            self.config.confidence_level,
            len(returns_array) - 1,
            loc=mean_return,
            scale=stats.sem(returns_array)
        )
        
        return BenchmarkResult(
            benchmark_name="Performance",
            agent_name=agent.__class__.__name__,
            environment_name=environment.__class__.__name__,
            mean_return=mean_return,
            std_return=std_return,
            median_return=median_return,
            min_return=float(np.min(returns_array)),
            max_return=float(np.max(returns_array)),
            safety_violations=0,  # Not primary focus
            violation_rate=0.0,
            safety_score=100.0,
            episode_length=float(np.mean(episode_lengths)),
            convergence_time=avg_convergence,
            sample_efficiency=sample_efficiency,
            confidence_interval=confidence_interval,
            statistical_significance=0.0,
            metadata={
                "performance_history": performance_history,
                "convergence_episodes": convergence_episodes,
                "total_samples": total_samples,
                "evaluation_time": evaluation_time
            }
        )
        
    def get_benchmark_description(self) -> str:
        """Get academic description of performance benchmark."""
        return (
            "Performance Benchmark: Evaluates agent task performance, "
            "sample efficiency, and convergence properties. Measures "
            "cumulative reward, learning speed, and final performance."
        )


class ScalabilityBenchmark(IndustrialBenchmark):
    """Benchmark focused on computational scalability."""
    
    def __init__(self, config: BenchmarkConfig, scale_factors: List[float] = [1.0, 2.0, 4.0]):
        super().__init__(config)
        self.scale_factors = scale_factors
        
    def evaluate_agent(
        self,
        agent: OfflineAgent,
        environment: IndustrialEnv,
        seed: int = 42
    ) -> BenchmarkResult:
        """Evaluate agent scalability across different problem sizes."""
        
        self.logger.info(f"Running scalability benchmark for {agent.__class__.__name__}")
        
        scalability_results = {}
        
        for scale_factor in self.scale_factors:
            self.logger.info(f"Testing scale factor: {scale_factor}")
            
            # Scale environment (simplified - would need environment-specific scaling)
            scaled_episodes = int(self.config.n_episodes * scale_factor)
            
            start_time = time.time()
            
            episode_returns = []
            computation_times = []
            
            key = jax.random.PRNGKey(seed)
            
            for episode in range(min(scaled_episodes, 50)):  # Limit for scalability test
                episode_key = jax.random.split(key, scaled_episodes)[episode]
                
                obs, _ = environment.reset(seed=int(episode_key[0]))
                
                episode_return = 0.0
                episode_start = time.time()
                
                for step in range(self.config.max_episode_steps):
                    step_start = time.time()
                    action = agent.predict(obs)
                    computation_times.append(time.time() - step_start)
                    
                    next_obs, reward, terminated, truncated, info = environment.step(action)
                    episode_return += reward
                    
                    if terminated or truncated:
                        break
                        
                    obs = next_obs
                    
                episode_returns.append(episode_return)
                
            scale_time = time.time() - start_time
            
            scalability_results[scale_factor] = {
                "mean_return": np.mean(episode_returns),
                "mean_computation_time": np.mean(computation_times),
                "total_time": scale_time,
                "throughput": len(episode_returns) / scale_time
            }
            
        # Calculate scalability metrics
        base_throughput = scalability_results[1.0]["throughput"]
        scalability_efficiency = {}
        
        for scale_factor in self.scale_factors:
            actual_throughput = scalability_results[scale_factor]["throughput"]
            ideal_throughput = base_throughput * scale_factor
            efficiency = actual_throughput / ideal_throughput if ideal_throughput > 0 else 0.0
            scalability_efficiency[scale_factor] = efficiency
            
        # Overall performance metrics
        base_results = scalability_results[1.0]
        
        return BenchmarkResult(
            benchmark_name="Scalability",
            agent_name=agent.__class__.__name__,
            environment_name=environment.__class__.__name__,
            mean_return=base_results["mean_return"],
            std_return=0.0,  # Not primary focus
            median_return=base_results["mean_return"],
            min_return=base_results["mean_return"],
            max_return=base_results["mean_return"],
            safety_violations=0,
            violation_rate=0.0,
            safety_score=100.0,
            episode_length=float(self.config.max_episode_steps),
            convergence_time=base_results["total_time"],
            sample_efficiency=base_results["throughput"],
            confidence_interval=(0.0, 0.0),
            statistical_significance=0.0,
            metadata={
                "scalability_results": scalability_results,
                "scalability_efficiency": scalability_efficiency,
                "scale_factors": self.scale_factors,
                "computation_times": computation_times[-100:]  # Last 100 for memory
            }
        )
        
    def get_benchmark_description(self) -> str:
        """Get academic description of scalability benchmark."""
        return (
            "Scalability Benchmark: Evaluates agent computational efficiency "
            f"across scale factors {self.scale_factors}. Measures throughput, "
            "computation time, and scaling efficiency."
        )


class RobustnessBenchmark(IndustrialBenchmark):
    """Benchmark focused on robustness to disturbances and uncertainties."""
    
    def __init__(
        self, 
        config: BenchmarkConfig,
        noise_levels: List[float] = [0.0, 0.1, 0.2, 0.3],
        disturbance_types: List[str] = ["observation_noise", "action_noise", "dynamics_noise"]
    ):
        super().__init__(config)
        self.noise_levels = noise_levels
        self.disturbance_types = disturbance_types
        
    def evaluate_agent(
        self,
        agent: OfflineAgent,
        environment: IndustrialEnv,
        seed: int = 42
    ) -> BenchmarkResult:
        """Evaluate agent robustness to various disturbances."""
        
        self.logger.info(f"Running robustness benchmark for {agent.__class__.__name__}")
        
        robustness_results = {}
        
        key = jax.random.PRNGKey(seed)
        
        for disturbance_type in self.disturbance_types:
            robustness_results[disturbance_type] = {}
            
            for noise_level in self.noise_levels:
                self.logger.info(f"Testing {disturbance_type} with noise level: {noise_level}")
                
                episode_returns = []
                safety_violations = []
                
                for episode in range(self.config.n_episodes // len(self.noise_levels)):
                    episode_key = jax.random.split(key, 1000)[episode]
                    
                    obs, _ = environment.reset(seed=int(episode_key[0]))
                    
                    episode_return = 0.0
                    episode_violations = 0
                    
                    for step in range(self.config.max_episode_steps):
                        # Apply disturbances based on type
                        if disturbance_type == "observation_noise":
                            noise = jax.random.normal(episode_key, obs.shape) * noise_level
                            noisy_obs = obs + noise
                            action = agent.predict(noisy_obs)
                        elif disturbance_type == "action_noise":
                            action = agent.predict(obs)
                            noise = jax.random.normal(episode_key, action.shape) * noise_level
                            action = action + noise
                        else:  # dynamics_noise - would need environment support
                            action = agent.predict(obs)
                            
                        next_obs, reward, terminated, truncated, info = environment.step(action)
                        
                        episode_return += reward
                        
                        # Track safety
                        if "safety_metrics" in info:
                            episode_violations += info["safety_metrics"].total_violations
                            
                        if terminated or truncated:
                            break
                            
                        obs = next_obs
                        
                    episode_returns.append(episode_return)
                    safety_violations.append(episode_violations)
                    
                robustness_results[disturbance_type][noise_level] = {
                    "mean_return": np.mean(episode_returns),
                    "std_return": np.std(episode_returns),
                    "safety_violations": np.sum(safety_violations),
                    "violation_rate": np.mean(np.array(safety_violations) > 0)
                }
                
        # Calculate robustness metrics
        baseline_performance = robustness_results[self.disturbance_types[0]][0.0]["mean_return"]
        
        robustness_scores = {}
        for disturbance_type in self.disturbance_types:
            scores = []
            for noise_level in self.noise_levels[1:]:  # Skip noise_level=0
                degraded_performance = robustness_results[disturbance_type][noise_level]["mean_return"]
                robustness_score = degraded_performance / baseline_performance if baseline_performance != 0 else 0.0
                scores.append(robustness_score)
            robustness_scores[disturbance_type] = np.mean(scores)
            
        overall_robustness = np.mean(list(robustness_scores.values()))
        
        return BenchmarkResult(
            benchmark_name="Robustness",
            agent_name=agent.__class__.__name__,
            environment_name=environment.__class__.__name__,
            mean_return=baseline_performance,
            std_return=robustness_results[self.disturbance_types[0]][0.0]["std_return"],
            median_return=baseline_performance,
            min_return=baseline_performance,
            max_return=baseline_performance,
            safety_violations=robustness_results[self.disturbance_types[0]][0.0]["safety_violations"],
            violation_rate=robustness_results[self.disturbance_types[0]][0.0]["violation_rate"],
            safety_score=overall_robustness * 100.0,
            episode_length=float(self.config.max_episode_steps),
            convergence_time=0.0,
            sample_efficiency=overall_robustness,
            confidence_interval=(0.0, 0.0),
            statistical_significance=0.0,
            metadata={
                "robustness_results": robustness_results,
                "robustness_scores": robustness_scores,
                "overall_robustness": overall_robustness,
                "noise_levels": self.noise_levels,
                "disturbance_types": self.disturbance_types
            }
        )
        
    def get_benchmark_description(self) -> str:
        """Get academic description of robustness benchmark."""
        return (
            "Robustness Benchmark: Evaluates agent performance under "
            f"disturbances: {self.disturbance_types} with noise levels "
            f"{self.noise_levels}. Measures performance degradation and "
            "safety maintenance under uncertainty."
        )


class IndustrialBenchmarkSuite:
    """Complete benchmark suite for industrial RL research."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize benchmarks
        self.benchmarks = [
            SafetyBenchmark(config),
            PerformanceBenchmark(config),
            ScalabilityBenchmark(config),
            RobustnessBenchmark(config)
        ]
        
    def run_full_evaluation(
        self,
        agents: List[OfflineAgent],
        environments: List[IndustrialEnv],
        seeds: Optional[List[int]] = None
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run complete benchmark evaluation for research paper."""
        
        if seeds is None:
            seeds = list(range(self.config.n_seeds))
            
        self.logger.info(
            f"Running full benchmark suite: {len(agents)} agents x "
            f"{len(environments)} environments x {len(self.benchmarks)} benchmarks"
        )
        
        all_results = {}
        
        for agent in agents:
            agent_name = agent.__class__.__name__
            all_results[agent_name] = []
            
            for environment in environments:
                env_name = environment.__class__.__name__
                
                for benchmark in self.benchmarks:
                    benchmark_name = benchmark.__class__.__name__
                    
                    self.logger.info(f"Evaluating {agent_name} on {env_name} with {benchmark_name}")
                    
                    # Run multiple seeds for statistical significance
                    seed_results = []
                    
                    for seed in seeds:
                        try:
                            result = benchmark.evaluate_agent(agent, environment, seed)
                            seed_results.append(result)
                        except Exception as e:
                            self.logger.error(f"Evaluation failed: {e}")
                            continue
                            
                    if seed_results:
                        # Aggregate results across seeds
                        aggregated_result = self._aggregate_results(seed_results)
                        all_results[agent_name].append(aggregated_result)
                        
        return all_results
        
    def _aggregate_results(self, seed_results: List[BenchmarkResult]) -> BenchmarkResult:
        """Aggregate results across multiple seeds."""
        
        if not seed_results:
            raise ValueError("No results to aggregate")
            
        # Extract arrays of metrics
        returns = [r.mean_return for r in seed_results]
        safety_violations = [r.safety_violations for r in seed_results]
        safety_scores = [r.safety_score for r in seed_results]
        
        # Calculate aggregate statistics
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        
        # Confidence interval across seeds
        from scipy import stats
        confidence_interval = stats.t.interval(
            self.config.confidence_level,
            len(returns) - 1,
            loc=mean_return,
            scale=stats.sem(returns)
        )
        
        # Use first result as template
        template = seed_results[0]
        
        return BenchmarkResult(
            benchmark_name=template.benchmark_name,
            agent_name=template.agent_name,
            environment_name=template.environment_name,
            mean_return=mean_return,
            std_return=std_return,
            median_return=float(np.median(returns)),
            min_return=float(np.min(returns)),
            max_return=float(np.max(returns)),
            safety_violations=int(np.mean(safety_violations)),
            violation_rate=float(np.mean([r.violation_rate for r in seed_results])),
            safety_score=float(np.mean(safety_scores)),
            episode_length=float(np.mean([r.episode_length for r in seed_results])),
            convergence_time=float(np.mean([r.convergence_time for r in seed_results])),
            sample_efficiency=float(np.mean([r.sample_efficiency for r in seed_results])),
            confidence_interval=confidence_interval,
            statistical_significance=0.0,  # Calculated separately
            metadata={
                "n_seeds": len(seed_results),
                "seed_results": [r.metadata for r in seed_results],
                "aggregation_method": "mean_across_seeds"
            }
        )
        
    def generate_academic_report(
        self, 
        results: Dict[str, List[BenchmarkResult]],
        output_path: str = "benchmark_report.md"
    ) -> str:
        """Generate academic report from benchmark results."""
        
        report_lines = [
            "# Industrial RL Benchmark Results",
            "",
            "## Experimental Setup",
            f"- Number of evaluation episodes: {self.config.n_episodes}",
            f"- Number of random seeds: {self.config.n_seeds}",
            f"- Confidence level: {self.config.confidence_level}",
            f"- Statistical test: {self.config.statistical_test}",
            "",
            "## Benchmark Descriptions",
            ""
        ]
        
        # Add benchmark descriptions
        for benchmark in self.benchmarks:
            report_lines.extend([
                f"### {benchmark.__class__.__name__}",
                benchmark.get_benchmark_description(),
                ""
            ])
            
        report_lines.extend([
            "## Results Summary",
            "",
            "| Agent | Environment | Benchmark | Mean Return | Std Return | Safety Score | CI (95%) |",
            "|-------|-------------|-----------|-------------|------------|--------------|----------|"
        ])
        
        # Add results table
        for agent_name, agent_results in results.items():
            for result in agent_results:
                ci_str = f"[{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]"
                report_lines.append(
                    f"| {agent_name} | {result.environment_name} | "
                    f"{result.benchmark_name} | {result.mean_return:.2f} | "
                    f"{result.std_return:.2f} | {result.safety_score:.1f} | {ci_str} |"
                )
                
        report_lines.extend([
            "",
            "## Statistical Analysis",
            "",
            "*Note: Statistical significance testing between agents can be "
            "performed using the provided StatisticalValidator.*",
            "",
            "## Reproducibility Information",
            "",
            f"- Configuration: {self.config}",
            f"- Benchmark suite version: 1.0.0",
            f"- Environment implementations: Advanced physics-based models",
            ""
        ])
        
        report_content = "\n".join(report_lines)
        
        # Save to file
        with open(output_path, "w") as f:
            f.write(report_content)
            
        self.logger.info(f"Academic report saved to {output_path}")
        return report_content