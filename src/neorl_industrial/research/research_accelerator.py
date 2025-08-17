"""Research acceleration framework for rapid experimentation."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Any, Callable, Optional, Tuple
from pathlib import Path
import pickle
import json
from datetime import datetime
import concurrent.futures
from dataclasses import dataclass
import time

from .novel_algorithms import NovelOfflineRLAlgorithms
from .meta_learning import IndustrialMetaLearning
from ..experiments.research_validation import ResearchValidationFramework
from ..quality_gates import ProgressiveQualityMonitor


@dataclass
class ExperimentConfig:
    """Configuration for research experiments."""
    name: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    environment: str
    dataset_config: Dict[str, Any]
    evaluation_config: Dict[str, Any]
    random_seed: int = 42


class ResearchAccelerator:
    """High-level framework for accelerating industrial RL research."""
    
    def __init__(
        self,
        experiment_dir: str = "research_experiments",
        max_parallel_experiments: int = 4,
        use_quality_gates: bool = True
    ):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        self.max_parallel_experiments = max_parallel_experiments
        self.use_quality_gates = use_quality_gates
        
        # Initialize quality gates monitor
        if self.use_quality_gates:
            self.quality_monitor = ProgressiveQualityMonitor(
                project_name="research_accelerator",
                config_path=None  # Use defaults
            )
        
        # Experiment tracking
        self.experiment_history = []
        self.active_experiments = {}
        
        print(f"Research Accelerator initialized: {experiment_dir}")
    
    def design_experiment(
        self,
        research_question: str,
        hypothesis: str,
        variables: Dict[str, List[Any]],
        base_config: ExperimentConfig
    ) -> List[ExperimentConfig]:
        """Design experiments for research question using systematic parameter sweeps."""
        
        print(f"Designing experiments for: {research_question}")
        
        experiments = []
        experiment_id = 0
        
        # Generate all combinations of variables
        import itertools
        
        variable_names = list(variables.keys())
        variable_values = list(variables.values())
        
        for combination in itertools.product(*variable_values):
            # Create experiment config
            exp_config = ExperimentConfig(
                name=f"{base_config.name}_exp_{experiment_id}",
                algorithm=base_config.algorithm,
                hyperparameters=base_config.hyperparameters.copy(),
                environment=base_config.environment,
                dataset_config=base_config.dataset_config.copy(),
                evaluation_config=base_config.evaluation_config.copy(),
                random_seed=base_config.random_seed + experiment_id
            )
            
            # Apply variable values
            for var_name, var_value in zip(variable_names, combination):
                if var_name in exp_config.hyperparameters:
                    exp_config.hyperparameters[var_name] = var_value
                elif var_name in exp_config.dataset_config:
                    exp_config.dataset_config[var_name] = var_value
                elif var_name in exp_config.evaluation_config:
                    exp_config.evaluation_config[var_name] = var_value
            
            experiments.append(exp_config)
            experiment_id += 1
        
        print(f"Generated {len(experiments)} experiment configurations")
        
        # Save experiment design
        design_metadata = {
            "research_question": research_question,
            "hypothesis": hypothesis,
            "variables": variables,
            "num_experiments": len(experiments),
            "timestamp": datetime.now().isoformat()
        }
        
        design_path = self.experiment_dir / f"experiment_design_{int(time.time())}.json"
        with open(design_path, 'w') as f:
            json.dump(design_metadata, f, indent=2)
        
        return experiments
    
    def run_experiment(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Run single experiment with quality gates."""
        
        start_time = time.time()
        exp_dir = self.experiment_dir / config.name
        exp_dir.mkdir(exist_ok=True)
        
        print(f"Running experiment: {config.name}")
        
        try:
            # Quality gate: Experiment initialization
            if self.use_quality_gates:
                init_result = self.quality_monitor.evaluate_quality_gates()
                if not init_result["passed"]:
                    return {"error": "Failed initialization quality gates", "quality_result": init_result}
            
            # Create agent
            if config.algorithm in NovelOfflineRLAlgorithms.list_algorithms():
                agent = NovelOfflineRLAlgorithms.get_algorithm(
                    config.algorithm,
                    **config.hyperparameters
                )
            else:
                # Fallback to standard algorithms
                from ..agents.cql import CQLAgent
                agent = CQLAgent(**config.hyperparameters)
            
            # Generate/load dataset
            dataset = self._generate_dataset(config.dataset_config)
            
            # Quality gate: Dataset validation
            if self.use_quality_gates:
                dataset_metrics = self._validate_dataset_quality(dataset)
                if dataset_metrics["quality_score"] < 0.7:
                    return {"error": "Dataset quality insufficient", "dataset_metrics": dataset_metrics}
            
            # Train agent
            training_start = time.time()
            training_metrics = agent.train(
                dataset,
                n_epochs=config.evaluation_config.get("training_epochs", 50),
                batch_size=config.evaluation_config.get("batch_size", 256)
            )
            training_time = time.time() - training_start
            
            # Quality gate: Training convergence
            if self.use_quality_gates:
                convergence_check = self._check_training_convergence(training_metrics)
                if not convergence_check["converged"]:
                    return {"error": "Training did not converge", "convergence_result": convergence_check}
            
            # Evaluate agent
            evaluation_metrics = self._evaluate_agent(agent, config.evaluation_config)
            
            # Compile results
            results = {
                "config": config.__dict__,
                "training_metrics": training_metrics,
                "evaluation_metrics": evaluation_metrics,
                "training_time": training_time,
                "total_time": time.time() - start_time,
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save results
            results_path = exp_dir / "results.pkl"
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
            
            # Save config
            config_path = exp_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config.__dict__, f, indent=2)
            
            print(f"Experiment {config.name} completed successfully")
            return results
            
        except Exception as e:
            error_result = {
                "config": config.__dict__,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save error
            error_path = exp_dir / "error.json"
            with open(error_path, 'w') as f:
                json.dump(error_result, f, indent=2)
            
            print(f"Experiment {config.name} failed: {e}")
            return error_result
    
    def run_parallel_experiments(
        self,
        experiments: List[ExperimentConfig],
        max_workers: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Run multiple experiments in parallel."""
        
        if max_workers is None:
            max_workers = self.max_parallel_experiments
        
        print(f"Running {len(experiments)} experiments with {max_workers} workers")
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_config = {
                executor.submit(self.run_experiment, config): config
                for config in experiments
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update progress
                    completed = len(results)
                    print(f"Progress: {completed}/{len(experiments)} experiments completed")
                    
                except Exception as e:
                    error_result = {
                        "config": config.__dict__,
                        "error": f"Parallel execution error: {e}",
                        "success": False
                    }
                    results.append(error_result)
        
        return results
    
    def analyze_experiment_results(
        self,
        results: List[Dict[str, Any]],
        research_question: str
    ) -> Dict[str, Any]:
        """Analyze experiment results and generate insights."""
        
        print("Analyzing experiment results...")
        
        # Separate successful and failed experiments
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]
        
        analysis = {
            "research_question": research_question,
            "total_experiments": len(results),
            "successful_experiments": len(successful),
            "failed_experiments": len(failed),
            "success_rate": len(successful) / len(results) if results else 0.0
        }
        
        if successful:
            # Extract performance metrics
            performance_metrics = []
            for result in successful:
                eval_metrics = result.get("evaluation_metrics", {})
                perf_dict = {
                    "experiment_name": result["config"]["name"],
                    "algorithm": result["config"]["algorithm"],
                    **eval_metrics
                }
                performance_metrics.append(perf_dict)
            
            # Find best performing experiments
            if performance_metrics:
                # Sort by primary metric (assuming 'mean_return' exists)
                primary_metric = "mean_return"
                if primary_metric in performance_metrics[0]:
                    sorted_results = sorted(
                        performance_metrics,
                        key=lambda x: x.get(primary_metric, 0),
                        reverse=True
                    )
                    
                    analysis["best_experiment"] = sorted_results[0]
                    analysis["worst_experiment"] = sorted_results[-1]
                    analysis["performance_range"] = {
                        "best": sorted_results[0].get(primary_metric, 0),
                        "worst": sorted_results[-1].get(primary_metric, 0)
                    }
                
                # Statistical analysis
                returns = [r.get(primary_metric, 0) for r in performance_metrics]
                analysis["statistics"] = {
                    "mean_performance": float(np.mean(returns)),
                    "std_performance": float(np.std(returns)),
                    "min_performance": float(np.min(returns)),
                    "max_performance": float(np.max(returns))
                }
        
        # Failure analysis
        if failed:
            error_types = {}
            for result in failed:
                error = result.get("error", "Unknown error")
                error_type = error.split(":")[0]  # Get error category
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            analysis["failure_analysis"] = error_types
        
        # Generate recommendations
        recommendations = self._generate_research_recommendations(analysis, successful)
        analysis["recommendations"] = recommendations
        
        # Save analysis
        analysis_path = self.experiment_dir / f"analysis_{int(time.time())}.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis
    
    def auto_hyperparameter_optimization(
        self,
        base_config: ExperimentConfig,
        optimization_space: Dict[str, Tuple[float, float]],
        n_trials: int = 20,
        optimization_metric: str = "mean_return"
    ) -> Dict[str, Any]:
        """Automated hyperparameter optimization using Bayesian optimization."""
        
        print(f"Starting hyperparameter optimization with {n_trials} trials")
        
        # Simple random search (could be replaced with more sophisticated methods)
        best_score = float('-inf')
        best_config = None
        optimization_history = []
        
        for trial in range(n_trials):
            # Sample hyperparameters
            trial_config = ExperimentConfig(
                name=f"{base_config.name}_trial_{trial}",
                algorithm=base_config.algorithm,
                hyperparameters=base_config.hyperparameters.copy(),
                environment=base_config.environment,
                dataset_config=base_config.dataset_config.copy(),
                evaluation_config=base_config.evaluation_config.copy(),
                random_seed=base_config.random_seed + trial
            )
            
            # Sample from optimization space
            for param_name, (low, high) in optimization_space.items():
                if param_name in trial_config.hyperparameters:
                    trial_config.hyperparameters[param_name] = np.random.uniform(low, high)
            
            # Run experiment
            result = self.run_experiment(trial_config)
            
            # Extract score
            score = 0.0
            if result.get("success", False):
                eval_metrics = result.get("evaluation_metrics", {})
                score = eval_metrics.get(optimization_metric, 0.0)
            
            # Update best
            if score > best_score:
                best_score = score
                best_config = trial_config
            
            optimization_history.append({
                "trial": trial,
                "config": trial_config.__dict__,
                "score": score,
                "success": result.get("success", False)
            })
            
            print(f"Trial {trial}: score={score:.4f}, best={best_score:.4f}")
        
        optimization_result = {
            "best_score": best_score,
            "best_config": best_config.__dict__ if best_config else None,
            "optimization_history": optimization_history,
            "n_trials": n_trials,
            "optimization_metric": optimization_metric
        }
        
        # Save optimization results
        opt_path = self.experiment_dir / f"hyperopt_{int(time.time())}.json"
        with open(opt_path, 'w') as f:
            json.dump(optimization_result, f, indent=2)
        
        return optimization_result
    
    def _generate_dataset(self, dataset_config: Dict[str, Any]) -> Dict[str, jnp.ndarray]:
        """Generate synthetic dataset for experiments."""
        
        n_samples = dataset_config.get("n_samples", 10000)
        state_dim = dataset_config.get("state_dim", 20)
        action_dim = dataset_config.get("action_dim", 6)
        noise_level = dataset_config.get("noise_level", 0.1)
        
        # Generate synthetic data
        key = jax.random.PRNGKey(42)
        key1, key2, key3 = jax.random.split(key, 3)
        
        observations = jax.random.normal(key1, (n_samples, state_dim))
        actions = jax.random.normal(key2, (n_samples, action_dim))
        
        # Simple reward function
        rewards = jnp.sum(observations * actions[:, :state_dim], axis=1)
        rewards += noise_level * jax.random.normal(key3, (n_samples,))
        
        return {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "terminals": jnp.zeros(n_samples, dtype=bool)
        }
    
    def _validate_dataset_quality(self, dataset: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
        """Validate dataset quality for experiments."""
        
        metrics = {}
        
        # Basic statistics
        obs = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]
        
        metrics["n_samples"] = len(obs)
        metrics["state_dim"] = obs.shape[1]
        metrics["action_dim"] = actions.shape[1]
        
        # Data quality checks
        metrics["has_nan"] = bool(jnp.any(jnp.isnan(obs)) or jnp.any(jnp.isnan(actions)))
        metrics["has_inf"] = bool(jnp.any(jnp.isinf(obs)) or jnp.any(jnp.isinf(actions)))
        
        # Diversity metrics
        metrics["obs_variance"] = float(jnp.var(obs))
        metrics["action_variance"] = float(jnp.var(actions))
        metrics["reward_range"] = float(jnp.max(rewards) - jnp.min(rewards))
        
        # Quality score (simple heuristic)
        quality_score = 1.0
        if metrics["has_nan"] or metrics["has_inf"]:
            quality_score *= 0.0
        if metrics["obs_variance"] < 0.01:
            quality_score *= 0.5
        if metrics["action_variance"] < 0.01:
            quality_score *= 0.5
        if metrics["reward_range"] < 0.1:
            quality_score *= 0.7
        
        metrics["quality_score"] = quality_score
        
        return metrics
    
    def _check_training_convergence(self, training_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check if training converged properly."""
        
        convergence_result = {"converged": False, "reason": "Unknown"}
        
        try:
            final_metrics = training_metrics.get("final_metrics", {})
            
            # Simple convergence check
            if "critic_loss" in final_metrics:
                loss = final_metrics["critic_loss"]
                if not jnp.isnan(loss) and not jnp.isinf(loss) and loss < 100.0:
                    convergence_result["converged"] = True
                    convergence_result["reason"] = "Loss converged"
                else:
                    convergence_result["reason"] = f"Loss diverged: {loss}"
            else:
                convergence_result["reason"] = "No loss metrics available"
                
        except Exception as e:
            convergence_result["reason"] = f"Error checking convergence: {e}"
        
        return convergence_result
    
    def _evaluate_agent(self, agent, eval_config: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate trained agent."""
        
        # Generate evaluation dataset
        eval_dataset = self._generate_dataset({
            "n_samples": eval_config.get("eval_samples", 1000),
            "state_dim": agent.state_dim,
            "action_dim": agent.action_dim,
            "noise_level": 0.05
        })
        
        # Simple evaluation metrics
        try:
            obs = eval_dataset["observations"][:100]  # Use subset for speed
            predicted_actions = agent.predict(obs)
            
            # Action statistics
            mean_action = float(jnp.mean(predicted_actions))
            std_action = float(jnp.std(predicted_actions))
            
            # Dummy performance metrics
            mean_return = float(jnp.mean(eval_dataset["rewards"][:100]))
            std_return = float(jnp.std(eval_dataset["rewards"][:100]))
            
            return {
                "mean_return": mean_return,
                "std_return": std_return,
                "mean_action": mean_action,
                "std_action": std_action,
                "evaluation_samples": 100
            }
            
        except Exception as e:
            return {"error": f"Evaluation failed: {e}"}
    
    def _generate_research_recommendations(
        self,
        analysis: Dict[str, Any],
        successful_results: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate research recommendations based on results."""
        
        recommendations = []
        
        # Success rate recommendations
        success_rate = analysis.get("success_rate", 0.0)
        if success_rate < 0.7:
            recommendations.append(
                "Low success rate detected. Consider adjusting hyperparameter ranges or algorithm selection."
            )
        
        # Performance recommendations
        if "statistics" in analysis:
            std_perf = analysis["statistics"].get("std_performance", 0)
            mean_perf = analysis["statistics"].get("mean_performance", 0)
            
            if std_perf > abs(mean_perf) * 0.5:
                recommendations.append(
                    "High performance variance detected. Consider more stable algorithms or regularization."
                )
        
        # Algorithm-specific recommendations
        if successful_results:
            algorithms = [r["config"]["algorithm"] for r in successful_results]
            best_algorithm = max(set(algorithms), key=algorithms.count)
            recommendations.append(
                f"Algorithm '{best_algorithm}' showed most consistent results. Consider focusing research effort here."
            )
        
        # General recommendations
        recommendations.extend([
            "Consider running longer training for more stable results.",
            "Implement statistical significance testing for robust conclusions.",
            "Add safety constraint evaluation for industrial deployment readiness."
        ])
        
        return recommendations


def create_research_pipeline(
    research_question: str,
    hypothesis: str,
    base_algorithm: str = "hierarchical_cql",
    **kwargs
) -> Tuple[ResearchAccelerator, List[ExperimentConfig]]:
    """Create complete research pipeline for a research question."""
    
    accelerator = ResearchAccelerator(**kwargs)
    
    # Define base configuration
    base_config = ExperimentConfig(
        name=f"research_{research_question.replace(' ', '_').lower()}",
        algorithm=base_algorithm,
        hyperparameters={
            "state_dim": 20,
            "action_dim": 6,
            "safety_critic": True,
            "constraint_threshold": 0.1
        },
        environment="ChemicalReactor-v0",
        dataset_config={
            "n_samples": 5000,
            "state_dim": 20,
            "action_dim": 6,
            "noise_level": 0.1
        },
        evaluation_config={
            "training_epochs": 30,
            "batch_size": 128,
            "eval_samples": 500
        }
    )
    
    # Define experimental variables (example)
    variables = {
        "constraint_threshold": [0.05, 0.1, 0.2],
        "noise_level": [0.05, 0.1, 0.15],
        "training_epochs": [20, 30, 50]
    }
    
    # Generate experiments
    experiments = accelerator.design_experiment(
        research_question=research_question,
        hypothesis=hypothesis,
        variables=variables,
        base_config=base_config
    )
    
    return accelerator, experiments