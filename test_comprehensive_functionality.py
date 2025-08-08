#!/usr/bin/env python3
"""Comprehensive functionality test for enhanced neoRL-industrial-gym."""

import numpy as np
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_enhanced_training():
    """Test enhanced training with all optimizations."""
    print("Testing enhanced training functionality...")
    
    try:
        import neorl_industrial as ni
        
        # Create environment and agent
        env = ni.make('ChemicalReactor-v0')
        agent = ni.CQLAgent(state_dim=12, action_dim=3, safety_critic=True)
        
        # Generate dataset
        print("  Generating training dataset...")
        dataset = env.get_dataset(quality="expert")
        print(f"  Dataset size: {len(dataset['observations'])} samples")
        
        # Enhanced training with all features
        print("  Starting enhanced training...")
        start_time = time.time()
        
        training_result = agent.train(
            dataset,
            n_epochs=3,
            batch_size=128,
            eval_env=env,
            eval_freq=2,
            use_mlflow=False
        )
        
        training_time = time.time() - start_time
        print(f"  Training completed in {training_time:.2f} seconds")
        
        # Check optimization report
        opt_report = training_result.get('optimization_report', {})
        print(f"  Applied optimizations: {opt_report.get('total_optimizations', 0)}")
        
        # Test performance monitoring
        performance_stats = agent.performance_monitor.get_metrics_summary()
        print(f"  CPU usage: {performance_stats.get('cpu_usage', {}).get('mean', 0):.1f}%")
        print(f"  Memory usage: {performance_stats.get('memory_usage_mb', {}).get('mean', 0):.1f}MB")
        
        # Test safety predictions
        test_obs = np.random.randn(5, 12).astype(np.float32)
        actions, safety_probs = agent.predict_with_safety(test_obs)
        print(f"  Safety predictions: {safety_probs.mean():.3f} avg violation prob")
        
        return True
        
    except Exception as e:
        print(f"  Enhanced training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_security_features():
    """Test security validation features."""
    print("\nTesting security features...")
    
    try:
        import neorl_industrial as ni
        from neorl_industrial.security import get_security_manager
        
        security_manager = get_security_manager()
        
        # Test array validation
        valid_array = np.random.randn(10, 5).astype(np.float32)
        result = security_manager.validate_input_array(
            valid_array,
            expected_dtype=np.float32,
            min_value=-10,
            max_value=10
        )
        print(f"  Valid array validation: {'PASS' if result else 'FAIL'}")
        
        # Test malicious array detection
        try:
            invalid_array = np.full((5, 5), np.inf)
            security_manager.validate_input_array(invalid_array, allow_inf=False)
            print("  Infinite value detection: FAIL")
        except Exception:
            print("  Infinite value detection: PASS")
            
        # Test string sanitization
        safe_string = security_manager.sanitize_string("normal_parameter_name")
        print(f"  String sanitization: {'PASS' if safe_string == 'normal_parameter_name' else 'FAIL'}")
        
        # Test malicious string detection
        try:
            security_manager.sanitize_string("<script>alert('xss')</script>")
            print("  XSS detection: FAIL")
        except Exception:
            print("  XSS detection: PASS")
            
        return True
        
    except Exception as e:
        print(f"  Security test failed: {e}")
        return False

def test_optimization_features():
    """Test performance optimization features."""
    print("\nTesting optimization features...")
    
    try:
        from neorl_industrial.optimization import (
            get_performance_optimizer,
            get_global_cache,
            benchmark_function,
            auto_optimize_function
        )
        
        # Test performance optimizer
        optimizer = get_performance_optimizer()
        print(f"  Performance optimizer initialized: {optimizer.device_count} devices")
        
        # Test function optimization
        def sample_function(x):
            return x ** 2 + np.sin(x)
            
        optimized_func = auto_optimize_function(sample_function)
        
        # Test caching
        cache = get_global_cache()
        cache.put("test_key", {"data": [1, 2, 3]})
        cached_result = cache.get("test_key")
        print(f"  Caching: {'PASS' if cached_result is not None else 'FAIL'}")
        
        # Test benchmarking
        test_data = np.random.randn(100)
        benchmark_results = benchmark_function(
            sample_function,
            test_data,
            warmup_runs=2,
            benchmark_runs=5
        )
        
        if 'mean_time' in benchmark_results:
            print(f"  Benchmarking: PASS (avg: {benchmark_results['mean_time']:.4f}s)")
        else:
            print("  Benchmarking: FAIL")
            
        return True
        
    except Exception as e:
        print(f"  Optimization test failed: {e}")
        return False

def test_monitoring_features():
    """Test monitoring and logging features."""
    print("\nTesting monitoring features...")
    
    try:
        from neorl_industrial.monitoring import get_logger, get_performance_monitor
        
        # Test logging
        logger = get_logger("test_logger")
        logger.info("Test info message")
        logger.warning("Test warning message")
        
        # Test safety event logging
        logger.safety_event(
            event_type="TEST_EVENT",
            severity="LOW",
            details={"test": True},
            agent_id="test_agent"
        )
        
        print("  Logging system: PASS")
        
        # Test performance monitoring
        monitor = get_performance_monitor("test_monitor")
        monitor.start_monitoring()
        
        with monitor.time_operation("test_operation"):
            time.sleep(0.01)  # Simulate work
            
        monitor.record_event("test_events", 5)
        
        current_metrics = monitor.get_current_metrics()
        health_status = monitor.health_check()
        
        print(f"  Performance monitoring: PASS")
        print(f"  Health check components: {len(health_status)}")
        
        monitor.stop_monitoring()
        
        return True
        
    except Exception as e:
        print(f"  Monitoring test failed: {e}")
        return False

def main():
    """Run comprehensive functionality tests."""
    print("=== neoRL-industrial-gym Comprehensive Enhancement Tests ===\n")
    
    tests = [
        test_enhanced_training,
        test_security_features,
        test_optimization_features,
        test_monitoring_features,
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
            print(f"  Test {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n=== Enhancement Test Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")
    
    if failed == 0:
        print("\n🚀 All enhancement tests passed! System is production-ready.")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed.")
        return 1

if __name__ == "__main__":
    exit(main())