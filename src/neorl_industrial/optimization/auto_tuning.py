"""Automated hyperparameter tuning and optimization."""

import time
import random
from typing import Any, Dict, List, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import json

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from ..monitoring.logger import get_logger
from ..core.types import Array
from ..validation import ValidationResult, validate_hyperparameters


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    n_trials: int = 100
    optimization_direction: str = "maximize"  # "maximize" or "minimize"
    timeout: Optional[float] = None  # seconds
    n_jobs: int = 1  # parallel jobs
    sampler: str = "tpe"  # "tpe", "random", "cmaes"
    pruner: str = "median"  # "median", "successive_halving", "hyperband"
    study_name: Optional[str] = None
    storage: Optional[str] = None  # database URL for persistence
    
    # Early stopping
    patience: int = 20
    min_improvement: float = 1e-4
    
    # Resource constraints
    max_memory_gb: Optional[float] = None
    max_training_time: Optional[float] = None


class HyperparameterSpace:
    """Defines the search space for hyperparameters."""
    
    def __init__(self):
        self.parameters = {}
        self.logger = get_logger("hyperparameter_space")
    
    def add_float(
        self,
        name: str,
        low: float,
        high: float,
        log: bool = False,
        step: Optional[float] = None
    ):
        """Add a float parameter to the search space."""
        self.parameters[name] = {
            "type": "float",
            "low": low,
            "high": high,
            "log": log,
            "step": step,
        }
        self.logger.debug(f"Added float parameter: {name} in [{low}, {high}]")
    
    def add_int(self, name: str, low: int, high: int, step: int = 1):
        """Add an integer parameter to the search space."""
        self.parameters[name] = {
            "type": "int",
            "low": low,
            "high": high,
            "step": step,
        }
        self.logger.debug(f"Added int parameter: {name} in [{low}, {high}]")
    
    def add_categorical(self, name: str, choices: List[Any]):
        """Add a categorical parameter to the search space."""
        self.parameters[name] = {
            "type": "categorical",
            "choices": choices,
        }
        self.logger.debug(f"Added categorical parameter: {name} with choices {choices}")
    
    def add_boolean(self, name: str):
        """Add a boolean parameter to the search space."""
        self.parameters[name] = {
            "type": "boolean",
        }
        self.logger.debug(f"Added boolean parameter: {name}")
    
    def sample_parameters(self, trial=None) -> Dict[str, Any]:
        """Sample parameters from the search space."""
        if trial is None:
            # Random sampling
            sampled = {}
            for name, param in self.parameters.items():
                if param["type"] == "float":
                    if param.get("log", False):
                        # Log-uniform sampling
                        import math
                        log_low = math.log(param["low"])
                        log_high = math.log(param["high"])
                        sampled[name] = math.exp(random.uniform(log_low, log_high))
                    else:
                        sampled[name] = random.uniform(param["low"], param["high"])
                        if param.get("step") is not None:
                            sampled[name] = round(sampled[name] / param["step"]) * param["step"]
                
                elif param["type"] == "int":
                    sampled[name] = random.randint(param["low"], param["high"])
                
                elif param["type"] == "categorical":
                    sampled[name] = random.choice(param["choices"])
                
                elif param["type"] == "boolean":
                    sampled[name] = random.choice([True, False])
            
            return sampled
        
        else:
            # Optuna trial sampling
            sampled = {}
            for name, param in self.parameters.items():
                if param["type"] == "float":
                    sampled[name] = trial.suggest_float(
                        name,
                        param["low"],
                        param["high"],
                        log=param.get("log", False),
                        step=param.get("step"),
                    )
                
                elif param["type"] == "int":
                    sampled[name] = trial.suggest_int(
                        name,
                        param["low"],
                        param["high"],
                        step=param.get("step", 1),
                    )
                
                elif param["type"] == "categorical":
                    sampled[name] = trial.suggest_categorical(name, param["choices"])
                
                elif param["type"] == "boolean":
                    sampled[name] = trial.suggest_categorical(name, [True, False])
            
            return sampled


class AutoTuner:
    """Automated hyperparameter tuning system."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = get_logger("auto_tuner")
        
        # Optimization state
        self.best_params = None
        self.best_score = None
        self.optimization_history = []
        self.trial_count = 0
        
        # Early stopping
        self.no_improvement_count = 0
        self.best_trial_number = 0
        
        # Resource monitoring
        self.start_time = None
        self.total_training_time = 0
        
        self.logger.info(f"AutoTuner initialized with {config.n_trials} trials")
    
    def optimize(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        parameter_space: HyperparameterSpace,
        validation_function: Optional[Callable[[Dict[str, Any]], ValidationResult]] = None,
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization.
        
        Args:
            objective_function: Function to optimize (takes params, returns score)
            parameter_space: Search space definition
            validation_function: Optional parameter validation
            
        Returns:
            Dictionary with optimization results
        """
        self.start_time = time.time()
        self.logger.info("Starting hyperparameter optimization")
        
        if OPTUNA_AVAILABLE and self.config.n_jobs > 1:
            return self._optimize_with_optuna(objective_function, parameter_space, validation_function)
        else:
            return self._optimize_sequential(objective_function, parameter_space, validation_function)
    
    def _optimize_with_optuna(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        parameter_space: HyperparameterSpace,
        validation_function: Optional[Callable[[Dict[str, Any]], ValidationResult]] = None,
    ) -> Dict[str, Any]:
        """Optimize using Optuna for parallel optimization."""
        
        def objective(trial):
            return self._evaluate_trial(
                trial, objective_function, parameter_space, validation_function
            )
        
        # Create study
        study = optuna.create_study(
            direction=self.config.optimization_direction,
            study_name=self.config.study_name,
            storage=self.config.storage,
            load_if_exists=True,
        )
        
        # Configure sampler
        if self.config.sampler == "tpe":
            sampler = optuna.samplers.TPESampler()
        elif self.config.sampler == "random":
            sampler = optuna.samplers.RandomSampler()
        elif self.config.sampler == "cmaes":
            sampler = optuna.samplers.CmaEsSampler()
        else:
            sampler = optuna.samplers.TPESampler()
        
        study.sampler = sampler
        
        # Configure pruner
        if self.config.pruner == "median":
            pruner = optuna.pruners.MedianPruner()
        elif self.config.pruner == "successive_halving":
            pruner = optuna.pruners.SuccessiveHalvingPruner()
        elif self.config.pruner == "hyperband":
            pruner = optuna.pruners.HyperbandPruner()
        else:
            pruner = optuna.pruners.MedianPruner()
        
        study.pruner = pruner
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
        )
        
        # Extract results
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        return {
            "best_parameters": self.best_params,
            "best_score": self.best_score,
            "n_trials": len(study.trials),
            "optimization_time": time.time() - self.start_time,
            "study": study,
        }
    
    def _optimize_sequential(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        parameter_space: HyperparameterSpace,
        validation_function: Optional[Callable[[Dict[str, Any]], ValidationResult]] = None,
    ) -> Dict[str, Any]:
        """Sequential optimization without Optuna."""
        
        best_score = float('-inf') if self.config.optimization_direction == 'maximize' else float('inf')
        best_params = None
        
        for trial in range(self.config.n_trials):
            # Check timeout
            if self.config.timeout and (time.time() - self.start_time) > self.config.timeout:
                self.logger.info(f"Optimization timeout after {trial} trials")
                break
            
            # Sample parameters
            params = parameter_space.sample_parameters()
            
            # Validate parameters
            if validation_function:
                validation = validation_function(params)
                if not validation.is_valid:
                    self.logger.warning(f"Trial {trial} parameters invalid: {validation.error_message}")
                    continue
                if validation.warnings:
                    self.logger.warning(f"Trial {trial} warnings: {validation.warnings}")
            
            # Evaluate
            try:
                score = objective_function(params)
                
                # Check if this is the best score
                is_better = (
                    (self.config.optimization_direction == 'maximize' and score > best_score) or
                    (self.config.optimization_direction == 'minimize' and score < best_score)
                )
                
                if is_better:
                    improvement = (
                        abs(score - best_score) 
                        if best_score != float('inf') and best_score != float('-inf') 
                        else float('inf')
                    )
                    
                    if improvement > self.config.min_improvement:
                        best_score = score
                        best_params = params.copy()
                        self.no_improvement_count = 0
                        self.best_trial_number = trial
                        
                        self.logger.info(f"Trial {trial}: New best score {score:.6f}")
                    else:
                        self.no_improvement_count += 1
                else:
                    self.no_improvement_count += 1
                
                # Store trial result
                self.optimization_history.append({
                    "trial": trial,
                    "parameters": params,
                    "score": score,
                    "is_best": is_better,
                })
                
                # Early stopping
                if self.no_improvement_count >= self.config.patience:
                    self.logger.info(f"Early stopping after {self.config.patience} trials without improvement")
                    break
                
            except Exception as e:
                self.logger.error(f"Trial {trial} failed: {e}")
                continue
        
        self.best_params = best_params
        self.best_score = best_score
        
        return {
            "best_parameters": self.best_params,
            "best_score": self.best_score,
            "n_trials": len(self.optimization_history),
            "optimization_time": time.time() - self.start_time,
            "optimization_history": self.optimization_history,
        }
    
    def _evaluate_trial(
        self,
        trial,
        objective_function: Callable[[Dict[str, Any]], float],
        parameter_space: HyperparameterSpace,
        validation_function: Optional[Callable[[Dict[str, Any]], ValidationResult]] = None,
    ) -> float:
        """Evaluate a single trial (for Optuna)."""
        params = parameter_space.sample_parameters(trial)
        
        # Validate parameters
        if validation_function:
            validation = validation_function(params)
            if not validation.is_valid:
                raise optuna.TrialPruned(f"Invalid parameters: {validation.error_message}")
        
        # Resource checks
        if self.config.max_training_time and time.time() - self.start_time > self.config.max_training_time:
            raise optuna.TrialPruned("Training time limit exceeded")
        
        try:
            score = objective_function(params)
            return score
        except Exception as e:
            self.logger.error(f"Trial evaluation failed: {e}")
            raise optuna.TrialPruned(f"Evaluation failed: {e}")
    
    def export_results(self, filename: str):
        """Export optimization results to JSON file."""
        results = {
            "config": {
                "n_trials": self.config.n_trials,
                "optimization_direction": self.config.optimization_direction,
                "sampler": self.config.sampler,
                "pruner": self.config.pruner,
            },
            "best_parameters": self.best_params,
            "best_score": self.best_score,
            "optimization_history": self.optimization_history,
            "optimization_time": time.time() - self.start_time if self.start_time else 0,
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results exported to {filename}")


def create_default_search_space(agent_type: str = "CQL") -> HyperparameterSpace:
    """Create a default hyperparameter search space for an agent type.
    
    Args:
        agent_type: Type of agent ("CQL", "IQL", "TD3BC")
        
    Returns:
        HyperparameterSpace instance
    """
    space = HyperparameterSpace()
    
    # Common parameters
    space.add_float("learning_rate", 1e-5, 1e-2, log=True)
    space.add_int("batch_size", 64, 512, step=64)
    space.add_float("gamma", 0.9, 0.999)
    space.add_float("tau", 0.001, 0.01, log=True)
    
    # Hidden layer configurations
    space.add_categorical("hidden_dims", [
        (128, 128),
        (256, 256),
        (512, 256),
        (256, 256, 256),
        (512, 512),
    ])
    
    # Agent-specific parameters
    if agent_type == "CQL":
        space.add_float("cql_alpha", 0.1, 10.0, log=True)
        space.add_float("alpha", 0.1, 1.0)
    
    elif agent_type == "IQL":
        space.add_float("beta", 1.0, 10.0)
        space.add_float("expectile", 0.7, 0.9)
    
    elif agent_type == "TD3BC":
        space.add_float("policy_noise", 0.1, 0.5)
        space.add_float("noise_clip", 0.3, 0.8)
        space.add_int("policy_freq", 1, 4)
        space.add_float("alpha", 1.0, 5.0)  # BC regularization
    
    # Safety parameters
    if agent_type in ["CQL", "IQL", "TD3BC"]:
        space.add_float("safety_penalty", 10.0, 500.0)
        space.add_float("constraint_threshold", 0.05, 0.2)
    
    return space


def auto_tune_agent(
    agent_class,
    base_config: Dict[str, Any],
    dataset: Dict[str, Array],
    eval_env,
    n_trials: int = 50,
    n_jobs: int = 1,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Convenience function for automated agent tuning.
    
    Args:
        agent_class: Agent class to tune
        base_config: Base configuration (non-tunable parameters)
        dataset: Training dataset
        eval_env: Evaluation environment
        n_trials: Number of optimization trials
        n_jobs: Number of parallel jobs
        timeout: Optimization timeout in seconds
        
    Returns:
        Dictionary with tuning results
    """
    # Create optimization config
    config = OptimizationConfig(
        n_trials=n_trials,
        optimization_direction="maximize",
        n_jobs=n_jobs,
        timeout=timeout,
    )
    
    # Create tuner
    tuner = AutoTuner(config)
    
    # Create search space
    agent_type = agent_class.__name__.replace("Agent", "")
    search_space = create_default_search_space(agent_type)
    
    # Define objective function
    def objective(params):
        # Merge with base config
        full_config = {**base_config, **params}
        
        try:
            # Create and train agent
            agent = agent_class(**full_config)
            training_results = agent.train(
                dataset=dataset,
                n_epochs=50,  # Reduced for tuning speed
                eval_env=eval_env,
                eval_freq=10,
            )
            
            # Return evaluation performance
            if "final_metrics" in training_results and "eval_return_mean" in training_results["final_metrics"]:
                return training_results["final_metrics"]["eval_return_mean"]
            else:
                return -1000.0  # Penalty for failed training
                
        except Exception as e:
            tuner.logger.error(f"Training failed: {e}")
            return -1000.0
    
    # Run optimization
    results = tuner.optimize(objective, search_space, validate_hyperparameters)
    
    return results
