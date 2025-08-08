#!/usr/bin/env python3
"""Basic functionality test for neoRL-industrial-gym."""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_import():
    """Test basic imports work."""
    print("Testing basic imports...")
    try:
        import neorl_industrial as ni
        print("✓ Main package imported successfully")
        
        # Test environment creation
        env = ni.make('ChemicalReactor-v0')
        print("✓ ChemicalReactor environment created")
        
        # Test agent creation
        agent = ni.CQLAgent(state_dim=12, action_dim=3)
        print("✓ CQL agent created")
        
        return True
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        return False

def test_environment_functionality():
    """Test basic environment functionality."""
    print("\nTesting environment functionality...")
    try:
        import neorl_industrial as ni
        
        env = ni.make('ChemicalReactor-v0')
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ Environment reset successful, obs shape: {obs.shape}")
        
        # Test step
        action = np.random.uniform(-1, 1, 3)
        next_obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step successful, reward: {reward:.3f}")
        
        # Test dataset generation
        dataset = env.get_dataset(quality="mixed")
        print(f"✓ Dataset generated with {len(dataset['observations'])} samples")
        
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_agent_functionality():
    """Test basic agent functionality."""
    print("\nTesting agent functionality...")
    try:
        import neorl_industrial as ni
        
        # Create small synthetic dataset
        n_samples = 1000
        state_dim, action_dim = 12, 3
        
        dataset = {
            'observations': np.random.randn(n_samples, state_dim).astype(np.float32),
            'actions': np.random.uniform(-1, 1, (n_samples, action_dim)).astype(np.float32),
            'rewards': np.random.randn(n_samples).astype(np.float32),
            'terminals': np.random.choice([True, False], n_samples, p=[0.1, 0.9]),
            'next_observations': np.random.randn(n_samples, state_dim).astype(np.float32),
        }
        
        # Create agent
        agent = ni.CQLAgent(state_dim=state_dim, action_dim=action_dim)
        print("✓ Agent created")
        
        # Test training (small scale)
        metrics = agent.train(dataset, n_epochs=2, batch_size=64)
        print("✓ Agent training completed")
        
        # Test prediction
        test_obs = np.random.randn(5, state_dim).astype(np.float32)
        actions = agent.predict(test_obs)
        print(f"✓ Agent prediction successful, actions shape: {actions.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        return False

def test_safety_functionality():
    """Test safety constraint functionality."""
    print("\nTesting safety functionality...")
    try:
        import neorl_industrial as ni
        
        env = ni.make('ChemicalReactor-v0')
        agent = ni.CQLAgent(state_dim=12, action_dim=3, safety_critic=True)
        print("✓ Safety-aware agent created")
        
        # Test safety evaluation  
        obs, _ = env.reset()
        
        # Generate minimal training data for safety critic
        dataset = env.get_dataset(quality="expert")
        agent.train(dataset, n_epochs=1, batch_size=128)
        
        # Test safety prediction
        actions, safety_probs = agent.predict_with_safety(obs[None])
        print(f"✓ Safety prediction successful, safety prob: {safety_probs[0]:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Safety test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== neoRL-industrial-gym Basic Functionality Tests ===\n")
    
    tests = [
        test_basic_import,
        test_environment_functionality,
        test_agent_functionality,
        test_safety_functionality,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=== Test Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! Basic functionality is working.")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed.")
        return 1

if __name__ == "__main__":
    exit(main())