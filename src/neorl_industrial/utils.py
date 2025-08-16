"""Utility functions for neoRL-industrial-gym."""

import numpy as np
from typing import Dict, Any, Optional

from .environments import ChemicalReactorEnv, PowerGridEnv, RobotAssemblyEnv
from .environments.advanced_chemical_reactor import AdvancedChemicalReactorEnv
from .environments.advanced_power_grid import AdvancedPowerGridEnv
from .core.types import Array


def make(env_id: str, **kwargs) -> Any:
    """Create industrial environment by ID.
    
    Args:
        env_id: Environment identifier
        **kwargs: Additional arguments for environment
        
    Returns:
        Environment instance
        
    Examples:
        >>> import neorl_industrial as ni
        >>> env = ni.make('ChemicalReactor-v0')
    """
    env_registry = {
        'ChemicalReactor-v0': ChemicalReactorEnv,
        'PowerGrid-v0': PowerGridEnv,
        'RobotAssembly-v0': RobotAssemblyEnv,
        'AdvancedChemicalReactor-v0': AdvancedChemicalReactorEnv,
        'AdvancedPowerGrid-v0': AdvancedPowerGridEnv,
    }
    
    if env_id not in env_registry:
        available = ', '.join(env_registry.keys())
        raise ValueError(f"Unknown environment '{env_id}'. Available: {available}")
    
    env_class = env_registry[env_id]
    return env_class(**kwargs)


def evaluate_with_safety(
    agent: Any,
    env: Any, 
    n_episodes: int = 100,
    record_video: bool = False,
    render: bool = False,
) -> Dict[str, Any]:
    """Evaluate agent with comprehensive safety metrics.
    
    Args:
        agent: Trained agent to evaluate
        env: Environment to evaluate on
        n_episodes: Number of episodes to run
        record_video: Whether to record video (placeholder)
        render: Whether to render during evaluation
        
    Returns:
        Dictionary with evaluation metrics including safety
        
    Examples:
        >>> import neorl_industrial as ni
        >>> env = ni.make('ChemicalReactor-v0')
        >>> agent = ni.CQLAgent(state_dim=12, action_dim=3)
        >>> # ... train agent ...
        >>> metrics = ni.evaluate_with_safety(agent, env, n_episodes=50)
        >>> print(f"Safety violations: {metrics['safety_violations']}")
    """
    if not hasattr(agent, 'is_trained') or not agent.is_trained:
        raise RuntimeError("Agent must be trained before evaluation")
    
    # Standard metrics
    episode_returns = []
    episode_lengths = []
    
    # Safety metrics
    total_violations = 0
    critical_violations = 0
    emergency_shutdowns = 0
    constraint_satisfaction_rates = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_return = 0.0
        episode_length = 0
        episode_violations = 0
        episode_critical = 0
        episode_shutdowns = 0
        
        done = False
        while not done:
            # Get action from agent
            action = agent.predict(obs[None], deterministic=True)[0]
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
            episode_length += 1
            
            # Track safety metrics
            if 'safety_metrics' in info:
                safety_metrics = info['safety_metrics']
                episode_violations += safety_metrics.violation_count
                episode_critical += safety_metrics.critical_violations
                
                constraint_satisfaction_rates.append(safety_metrics.satisfaction_rate)
            
            if info.get('critical_shutdown', False):
                episode_shutdowns += 1
            
            obs = next_obs
            
            if render:
                try:
                    env.render()
                except:
                    pass
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        total_violations += episode_violations
        critical_violations += episode_critical
        emergency_shutdowns += episode_shutdowns
    
    # Compile results
    results = {
        # Performance metrics
        'return_mean': np.mean(episode_returns),
        'return_std': np.std(episode_returns),
        'return_min': np.min(episode_returns),
        'return_max': np.max(episode_returns),
        
        # Episode metrics
        'length_mean': np.mean(episode_lengths),
        'length_std': np.std(episode_lengths),
        
        # Safety metrics
        'safety_violations': total_violations,
        'safety_violations_per_episode': total_violations / n_episodes,
        'critical_violations': critical_violations,
        'emergency_shutdowns': emergency_shutdowns,
        'constraint_satisfaction_rate': (
            np.mean(constraint_satisfaction_rates) 
            if constraint_satisfaction_rates else 1.0
        ),
        
        # Success metrics
        'successful_episodes': sum(1 for r in episode_returns if r > 0),
        'success_rate': sum(1 for r in episode_returns if r > 0) / n_episodes,
    }
    
    return results