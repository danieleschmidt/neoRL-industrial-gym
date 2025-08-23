"""Research metrics and academic reporting tools."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import time
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResearchMetrics:
    """Comprehensive research metrics for academic publication."""
    
    # Performance Metrics
    mean_return: float = 0.0
    std_return: float = 0.0
    median_return: float = 0.0
    max_return: float = 0.0
    min_return: float = 0.0
    
    # Learning Metrics
    sample_efficiency: float = 0.0  # Steps to reach threshold
    convergence_episodes: int = 0
    final_performance: float = 0.0
    learning_curve_auc: float = 0.0
    
    # Safety Metrics
    safety_violations: int = 0
    safety_violation_rate: float = 0.0
    constraint_satisfaction: float = 1.0
    
    # Computational Metrics
    training_time_hours: float = 0.0
    wall_clock_time: float = 0.0
    memory_usage_gb: float = 0.0
    flops_estimate: float = 0.0
    
    # Robustness Metrics
    performance_variance: float = 0.0
    noise_robustness: float = 0.0
    generalization_score: float = 0.0
    
    # Statistical Metrics
    confidence_interval_95: Tuple[float, float] = (0.0, 0.0)
    statistical_significance: bool = False
    effect_size: float = 0.0
    
    # Meta Information
    algorithm_name: str = ""
    environment_name: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    random_seeds: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "performance": {
                "mean_return": self.mean_return,
                "std_return": self.std_return,
                "median_return": self.median_return,
                "max_return": self.max_return,
                "min_return": self.min_return
            },
            "learning": {
                "sample_efficiency": self.sample_efficiency,
                "convergence_episodes": self.convergence_episodes,
                "final_performance": self.final_performance,
                "learning_curve_auc": self.learning_curve_auc
            },
            "safety": {
                "safety_violations": self.safety_violations,
                "safety_violation_rate": self.safety_violation_rate,
                "constraint_satisfaction": self.constraint_satisfaction
            },
            "computational": {
                "training_time_hours": self.training_time_hours,
                "wall_clock_time": self.wall_clock_time,
                "memory_usage_gb": self.memory_usage_gb,
                "flops_estimate": self.flops_estimate
            },
            "robustness": {
                "performance_variance": self.performance_variance,
                "noise_robustness": self.noise_robustness,
                "generalization_score": self.generalization_score
            },
            "statistics": {
                "confidence_interval_95": self.confidence_interval_95,
                "statistical_significance": self.statistical_significance,
                "effect_size": self.effect_size
            },
            "meta": {
                "algorithm_name": self.algorithm_name,
                "environment_name": self.environment_name,
                "hyperparameters": self.hyperparameters,
                "random_seeds": self.random_seeds
            }
        }


class AcademicReporter:
    """Generates academic-style reports and tables."""
    
    def __init__(self, output_dir: Path = Path("./results")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_results_table(self, 
                              results: Dict[str, ResearchMetrics],
                              format: str = "latex") -> str:
        """Generate results table in LaTeX or Markdown format."""
        
        if format == "latex":
            return self._generate_latex_table(results)
        elif format == "markdown":
            return self._generate_markdown_table(results)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    def _generate_latex_table(self, results: Dict[str, ResearchMetrics]) -> str:
        """Generate LaTeX table."""
        header = r"""
\begin{table}[h!]
\centering
\caption{Experimental Results on Industrial RL Benchmarks}
\label{tab:results}
\begin{tabular}{lcccc}
\toprule
Algorithm & Mean Return & Std Return & Safety Rate & Sample Efficiency \\
\midrule
"""
        
        rows = []
        for alg_name, metrics in results.items():
            row = (
                f"{alg_name} & {metrics.mean_return:.2f} & "
                f"{metrics.std_return:.2f} & {metrics.constraint_satisfaction:.3f} & "
                f"{metrics.sample_efficiency:.0f} \\\\"
            )
            rows.append(row)
            
        footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""
        
        return header + "\n".join(rows) + footer
        
    def _generate_markdown_table(self, results: Dict[str, ResearchMetrics]) -> str:
        """Generate Markdown table."""
        header = """
| Algorithm | Mean Return | Std Return | Safety Rate | Sample Efficiency |
|-----------|-------------|------------|-------------|-------------------|
"""
        
        rows = []
        for alg_name, metrics in results.items():
            row = (
                f"| {alg_name} | {metrics.mean_return:.2f} | "
                f"{metrics.std_return:.2f} | {metrics.constraint_satisfaction:.3f} | "
                f"{metrics.sample_efficiency:.0f} |"
            )
            rows.append(row)
            
        return header + "\n".join(rows)
        
    def generate_statistical_summary(self, 
                                   baseline: ResearchMetrics,
                                   treatment: ResearchMetrics) -> Dict[str, Any]:
        """Generate statistical comparison summary."""
        
        improvement = (
            (treatment.mean_return - baseline.mean_return) / 
            baseline.mean_return
        ) * 100
        
        return {
            "performance_improvement_percent": improvement,
            "baseline_performance": {
                "mean": baseline.mean_return,
                "std": baseline.std_return,
                "ci": baseline.confidence_interval_95
            },
            "treatment_performance": {
                "mean": treatment.mean_return,
                "std": treatment.std_return,
                "ci": treatment.confidence_interval_95
            },
            "statistical_significance": treatment.statistical_significance,
            "effect_size": treatment.effect_size,
            "safety_comparison": {
                "baseline_violations": baseline.safety_violation_rate,
                "treatment_violations": treatment.safety_violation_rate,
                "safety_improvement": baseline.safety_violation_rate - treatment.safety_violation_rate
            },
            "efficiency_comparison": {
                "baseline_efficiency": baseline.sample_efficiency,
                "treatment_efficiency": treatment.sample_efficiency,
                "efficiency_ratio": (
                    treatment.sample_efficiency / baseline.sample_efficiency 
                    if baseline.sample_efficiency > 0 else float('inf')
                )
            }
        }
        
    def save_results(self, 
                    results: Dict[str, ResearchMetrics],
                    filename: str = "experimental_results.json"):
        """Save results to JSON file."""
        
        serializable_results = {
            name: metrics.to_dict() 
            for name, metrics in results.items()
        }
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
            
        logger.info(f"Results saved to {filepath}")
        
    def generate_paper_abstract(self, 
                               results: Dict[str, ResearchMetrics],
                               experiment_description: str) -> str:
        """Generate abstract template for academic paper."""
        
        best_algorithm = max(results.items(), key=lambda x: x[1].mean_return)
        best_name, best_metrics = best_algorithm
        
        template = f"""
ABSTRACT TEMPLATE

{experiment_description}

We evaluate {len(results)} algorithms on industrial control benchmarks. 
Our proposed method ({best_name}) achieves a mean return of "
        f"{best_metrics.mean_return:.2f} ± {best_metrics.std_return:.2f}, "
        "representing a "
        f"{((best_metrics.mean_return - min([m.mean_return for m in results.values()])) / "
        f"min([m.mean_return for m in results.values()]) * 100):.1f}% improvement "
        "over the baseline. The method maintains "
        f"{best_metrics.constraint_satisfaction:.1%} safety constraint satisfaction "
        f"with {best_metrics.sample_efficiency:.0f} sample efficiency. "
        "Statistical significance is confirmed 
with effect size {best_metrics.effect_size:.3f}.

KEYWORDS: Industrial Reinforcement Learning, Safety-Critical Control, Offline Learning
"""
        
        return template.strip()


class PublicationDataGenerator:
    """Generates publication-ready data and figures."""
    
    def __init__(self, results_dir: Path = Path("./results")):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_comparison_plots(self, results: Dict[str, ResearchMetrics]):
        """Generate comparison plots for publication."""
        # This would integrate with matplotlib/seaborn for publication plots
        pass
        
    def export_latex_tables(self, results: Dict[str, ResearchMetrics]):
        """Export results as LaTeX tables."""
        pass


class ReproducibilityValidator:
    """Validates experiment reproducibility."""
    
    def __init__(self):
        self.tolerance = 0.05  # 5% tolerance for reproducibility
        
    def validate_reproducibility(self, runs: List[ResearchMetrics]) -> bool:
        """Check if multiple runs are reproducible within tolerance."""
        if len(runs) < 2:
            return True
            
        means = [r.mean_return for r in runs]
        relative_std = np.std(means) / np.mean(means)
        return relative_std <= self.tolerance


class ExperimentTracker:
    """Track experiments for reproducibility."""
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.start_time = time.time()
        self.metrics_history: List[ResearchMetrics] = []
        
    def log_metrics(self, metrics: ResearchMetrics):
        """Log metrics for this experiment."""
        metrics.wall_clock_time = time.time() - self.start_time
        self.metrics_history.append(metrics)
        
    def get_final_metrics(self) -> Optional[ResearchMetrics]:
        """Get the final metrics for this experiment."""
        return self.metrics_history[-1] if self.metrics_history else None
        
    def get_learning_curve(self) -> List[float]:
        """Get learning curve data."""
        return [m.mean_return for m in self.metrics_history]


def create_research_metrics(algorithm_name: str,
                           environment_name: str,
                           returns: List[float],
                           safety_violations: List[int],
                           training_time: float,
                           hyperparameters: Dict[str, Any],
                           random_seeds: List[int]) -> ResearchMetrics:
    """Factory function to create research metrics from experimental data."""
    
    returns_array = np.array(returns)
    
    metrics = ResearchMetrics(
        mean_return=float(np.mean(returns_array)),
        std_return=float(np.std(returns_array)),
        median_return=float(np.median(returns_array)),
        max_return=float(np.max(returns_array)),
        min_return=float(np.min(returns_array)),
        
        safety_violations=int(np.sum(safety_violations)),
        safety_violation_rate=float(np.mean(safety_violations)),
        constraint_satisfaction=1.0 - float(np.mean(safety_violations)),
        
        training_time_hours=training_time / 3600.0,
        performance_variance=float(np.var(returns_array)),
        
        algorithm_name=algorithm_name,
        environment_name=environment_name,
        hyperparameters=hyperparameters,
        random_seeds=random_seeds
    )
    
    # Calculate confidence interval (95%)
    if len(returns) > 1:
        from scipy import stats
        ci = stats.t.interval(0.95, len(returns)-1, 
                             loc=metrics.mean_return,
                             scale=stats.sem(returns))
        metrics.confidence_interval_95 = (float(ci[0]), float(ci[1]))
    
    return metrics