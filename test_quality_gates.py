#!/usr/bin/env python3
"""Quality gates testing for neoRL-industrial-gym."""

import os
import sys
import time
import subprocess

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def run_linting():
    """Run code linting checks."""
    print("🔍 Running code linting checks...")
    
    # Check if we have any linting tools available
    linting_passed = True
    
    # We'll use basic Python syntax checking instead
    try:
        import ast
        
        # Check all Python files for syntax errors
        python_files = []
        for root, dirs, files in os.walk('./src'):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        syntax_errors = 0
        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                ast.parse(content)
            except SyntaxError as e:
                print(f"  ✗ Syntax error in {file_path}: {e}")
                syntax_errors += 1
                linting_passed = False
        
        if syntax_errors == 0:
            print(f"  ✓ Syntax check passed for {len(python_files)} files")
        else:
            print(f"  ✗ Found {syntax_errors} syntax errors")
            
    except Exception as e:
        print(f"  ⚠️ Linting check failed: {e}")
        linting_passed = False
    
    return linting_passed


def run_security_checks():
    """Run security validation checks."""
    print("\n🛡️ Running security checks...")
    
    try:
        import neorl_industrial
        from neorl_industrial.security import get_security_manager, SecurityError
        
        security_manager = get_security_manager()
        
        # Test various security scenarios
        security_tests_passed = 0
        total_security_tests = 0
        
        # Test 1: Valid input validation
        total_security_tests += 1
        try:
            import numpy as np
            valid_data = np.random.randn(10, 5).astype(np.float32)
            security_manager.validate_input_array(valid_data, expected_dtype=np.float32)
            security_tests_passed += 1
            print("  ✓ Input validation test passed")
        except Exception as e:
            print(f"  ✗ Input validation test failed: {e}")
        
        # Test 2: Malicious input detection
        total_security_tests += 1
        try:
            malicious_data = np.full((5, 5), np.inf)
            security_manager.validate_input_array(malicious_data, allow_inf=False)
            print("  ✗ Malicious input detection failed")
        except SecurityError:
            security_tests_passed += 1
            print("  ✓ Malicious input detection passed")
        
        # Test 3: String sanitization
        total_security_tests += 1
        try:
            clean_string = security_manager.sanitize_string("clean_input")
            if clean_string == "clean_input":
                security_tests_passed += 1
                print("  ✓ String sanitization test passed")
            else:
                print("  ✗ String sanitization test failed")
        except Exception as e:
            print(f"  ✗ String sanitization test failed: {e}")
        
        # Test 4: XSS detection
        total_security_tests += 1
        try:
            security_manager.sanitize_string("<script>alert('xss')</script>")
            print("  ✗ XSS detection failed")
        except SecurityError:
            security_tests_passed += 1
            print("  ✓ XSS detection passed")
        
        print(f"  Security tests: {security_tests_passed}/{total_security_tests} passed")
        return security_tests_passed == total_security_tests
        
    except Exception as e:
        print(f"  ✗ Security checks failed: {e}")
        return False


def run_performance_tests():
    """Run performance benchmarks."""
    print("\n⚡ Running performance tests...")
    
    try:
        import neorl_industrial as ni
        import numpy as np
        
        # Create environment and agent
        env = ni.make('ChemicalReactor-v0')
        agent = ni.CQLAgent(state_dim=12, action_dim=3)
        
        # Performance test 1: Environment step speed
        obs, _ = env.reset()
        start_time = time.time()
        
        for _ in range(100):
            action = np.random.uniform(-1, 1, 3)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        
        env_step_time = (time.time() - start_time) / 100
        print(f"  Environment step time: {env_step_time:.4f}s")
        
        # Performance test 2: Small training speed
        dataset = env.get_dataset(quality="expert")
        small_dataset = {
            key: val[:1000] for key, val in dataset.items()
        }
        
        start_time = time.time()
        agent.train(small_dataset, n_epochs=1, batch_size=64)
        training_time = time.time() - start_time
        print(f"  Training time (1k samples): {training_time:.2f}s")
        
        # Performance test 3: Inference speed
        test_obs = np.random.randn(100, 12).astype(np.float32)
        start_time = time.time()
        actions = agent.predict(test_obs)
        inference_time = (time.time() - start_time) / 100
        print(f"  Inference time per sample: {inference_time:.4f}s")
        
        # Validate performance thresholds
        performance_passed = True
        if env_step_time > 0.01:  # 10ms per step max
            print(f"  ⚠️ Environment step time too slow: {env_step_time:.4f}s")
            performance_passed = False
            
        if inference_time > 0.001:  # 1ms per inference max
            print(f"  ⚠️ Inference time too slow: {inference_time:.4f}s")  
            performance_passed = False
            
        if performance_passed:
            print("  ✓ All performance tests passed")
        
        return performance_passed
        
    except Exception as e:
        print(f"  ✗ Performance tests failed: {e}")
        return False


def run_integration_tests():
    """Run integration tests."""
    print("\n🔄 Running integration tests...")
    
    try:
        import neorl_industrial as ni
        import numpy as np
        
        # Test 1: Full training pipeline
        env = ni.make('ChemicalReactor-v0')
        agent = ni.CQLAgent(state_dim=12, action_dim=3, safety_critic=True)
        
        # Generate small dataset
        dataset = env.get_dataset(quality="expert")
        small_dataset = {key: val[:500] for key, val in dataset.items()}
        
        # Train agent
        training_result = agent.train(
            small_dataset,
            n_epochs=1,
            batch_size=32,
            use_mlflow=False
        )
        
        # Test predictions
        test_obs = np.random.randn(5, 12).astype(np.float32)
        actions = agent.predict(test_obs)
        
        # Test safety predictions
        actions, safety_probs = agent.predict_with_safety(test_obs)
        
        # Test evaluation
        eval_results = ni.evaluate_with_safety(agent, env, n_episodes=5)
        
        print("  ✓ Full pipeline integration test passed")
        print(f"    - Training metrics available: {'training_metrics' in training_result}")
        print(f"    - Actions shape: {actions.shape}")
        print(f"    - Safety probabilities shape: {safety_probs.shape}")
        print(f"    - Evaluation return: {eval_results['return_mean']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Integration tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_memory_tests():
    """Run memory usage tests."""
    print("\n💾 Running memory tests...")
    
    try:
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple agents and environments
        import neorl_industrial as ni
        
        agents = []
        envs = []
        
        for i in range(3):
            env = ni.make('ChemicalReactor-v0')
            agent = ni.CQLAgent(state_dim=12, action_dim=3)
            envs.append(env)
            agents.append(agent)
        
        # Check memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Cleanup
        del agents
        del envs
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Peak memory: {peak_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Memory increase: {peak_memory - initial_memory:.1f}MB")
        
        # Check for memory leaks (final should be close to initial)
        memory_leak = final_memory - initial_memory
        if memory_leak > 50:  # 50MB threshold
            print(f"  ⚠️ Possible memory leak: {memory_leak:.1f}MB")
            return False
        else:
            print("  ✓ Memory usage test passed")
            return True
            
    except ImportError:
        print("  ⚠️ psutil not available, skipping memory tests")
        return True
    except Exception as e:
        print(f"  ✗ Memory tests failed: {e}")
        return False


def main():
    """Run all quality gate tests."""
    print("🎯 neoRL-industrial-gym Quality Gates Testing\n")
    
    quality_gates = [
        ("Code Linting", run_linting),
        ("Security Checks", run_security_checks),
        ("Performance Tests", run_performance_tests),
        ("Integration Tests", run_integration_tests),
        ("Memory Tests", run_memory_tests),
    ]
    
    passed_gates = 0
    total_gates = len(quality_gates)
    
    for gate_name, gate_function in quality_gates:
        print(f"Running {gate_name}...")
        try:
            if gate_function():
                passed_gates += 1
                print(f"✅ {gate_name} PASSED\n")
            else:
                print(f"❌ {gate_name} FAILED\n")
        except Exception as e:
            print(f"💥 {gate_name} CRASHED: {e}\n")
    
    print("=" * 60)
    print(f"Quality Gates Summary: {passed_gates}/{total_gates} PASSED")
    
    if passed_gates == total_gates:
        print("🎉 ALL QUALITY GATES PASSED - READY FOR PRODUCTION!")
        return 0
    else:
        print(f"⚠️ {total_gates - passed_gates} quality gate(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())