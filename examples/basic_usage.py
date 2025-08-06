"""Basic usage example for neoRL-industrial-gym."""

import numpy as np
import neorl_industrial as ni


def main():
    """Demonstrate basic usage of the industrial RL library."""
    
    print("🏭 neoRL-industrial-gym Basic Example")
    print("=" * 50)
    
    # Create environment
    print("\n1. Creating ChemicalReactor-v0 environment...")
    env = ni.make('ChemicalReactor-v0')
    print(f"   State dimension: {env.observation_space.shape[0]}")
    print(f"   Action dimension: {env.action_space.shape[0]}")
    print(f"   Safety constraints: {len(env.safety_constraints)}")
    
    # Reset environment and test basic functionality
    print("\n2. Testing environment reset and step...")
    obs, info = env.reset()
    print(f"   Initial temperature: {obs[0]:.2f} K")
    print(f"   Initial pressure: {obs[1]:.0f} Pa")
    print(f"   Safety status: {info['safety_metrics']}")
    
    # Take a few random actions
    print("\n3. Testing random actions...")
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Step {step+1}: reward={reward:.2f}, temp={obs[0]:.1f}K, violations={info.get('violations', 0)}")
        
        if terminated or truncated:
            print("   Episode ended")
            break
    
    # Load dataset
    print("\n4. Loading offline dataset...")
    dataset = env.get_dataset(quality='mixed')
    print(f"   Dataset size: {len(dataset['observations']):,} transitions")
    print(f"   Average reward: {np.mean(dataset['rewards']):.2f}")
    print(f"   Episodes: {np.sum(dataset['terminals']):,}")
    
    # Initialize agent
    print("\n5. Initializing CQL agent...")
    agent = ni.CQLAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        safety_critic=True,
        constraint_threshold=0.1
    )
    print("   ✓ CQL agent created with safety critic")
    
    # Quick training demo (just 5 epochs for demo)
    print("\n6. Training agent (demo - 5 epochs)...")
    training_metrics = agent.train(
        dataset,
        n_epochs=5,
        batch_size=64,
        eval_env=env,
        eval_freq=5
    )
    print("   ✓ Training completed")
    print(f"   Final critic loss: {training_metrics['training_metrics'][-1].get('td_loss', 0):.4f}")
    
    # Evaluate trained agent
    print("\n7. Evaluating trained agent...")
    eval_metrics = ni.evaluate_with_safety(
        agent, 
        env, 
        n_episodes=10
    )
    
    print(f"   Average return: {eval_metrics['return_mean']:.2f} ± {eval_metrics['return_std']:.2f}")
    print(f"   Safety violations: {eval_metrics['safety_violations']}")
    print(f"   Success rate: {eval_metrics['success_rate']:.1%}")
    print(f"   Constraint satisfaction: {eval_metrics['constraint_satisfaction_rate']:.1%}")
    
    print("\n✅ Basic usage example completed successfully!")
    print("    Check out advanced examples for more features like:")
    print("    - Custom safety constraints")
    print("    - MLflow experiment tracking") 
    print("    - Multi-environment benchmarking")


if __name__ == "__main__":
    main()