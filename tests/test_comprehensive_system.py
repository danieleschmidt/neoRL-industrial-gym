"""Comprehensive system tests for neoRL-industrial-gym."""

import pytest
import numpy as np
import jax.numpy as jnp
from unittest.mock import Mock, patch
import tempfile
import time
from pathlib import Path

# Import core components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import neorl_industrial as ni
from neorl_industrial.core.types import SafetyConstraint, SafetyMetrics
from neorl_industrial.environments import ChemicalReactorEnv, PowerGridEnv, RobotAssemblyEnv
from neorl_industrial.agents import CQLAgent, IQLAgent, TD3BCAgent
from neorl_industrial.monitoring.dashboard import MonitoringDashboard, SystemMetric
from neorl_industrial.resilience.circuit_breaker_enhanced import AdaptiveCircuitBreaker
from neorl_industrial.validation.comprehensive_validator import ComprehensiveValidator
from neorl_industrial.optimization.intelligent_caching import IntelligentCache

class TestEnvironmentIntegration:
    """Test environment integration and functionality."""
    
    def test_environment_creation(self):
        """Test all environments can be created."""
        env_ids = ['ChemicalReactor-v0', 'PowerGrid-v0', 'RobotAssembly-v0']
        
        for env_id in env_ids:
            env = ni.make(env_id)
            assert env is not None
            assert hasattr(env, 'state_dim')
            assert hasattr(env, 'action_dim')
            assert env.state_dim > 0
            assert env.action_dim > 0
    
    def test_environment_episode_lifecycle(self):
        """Test complete environment episode lifecycle."""
        env = ni.make('ChemicalReactor-v0')
        
        # Reset
        obs, info = env.reset()
        assert obs.shape == (env.state_dim,)
        assert isinstance(info, dict)
        
        # Step through episode
        episode_length = 0
        done = False
        total_reward = 0
        
        while not done and episode_length < 100:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Validate outputs
            assert next_obs.shape == (env.state_dim,)
            assert isinstance(reward, (int, float))
            assert isinstance(terminated, bool)
            assert isinstance(truncated, bool)
            assert isinstance(info, dict)
            
            total_reward += reward
            episode_length += 1
            done = terminated or truncated
            obs = next_obs
        
        assert episode_length > 0
        assert abs(total_reward) < 1e6  # Reasonable reward bounds
    
    def test_safety_constraints(self):
        """Test safety constraint evaluation."""
        env = ni.make('ChemicalReactor-v0')
        obs, _ = env.reset()
        action = env.action_space.sample()
        
        # Test safety constraint evaluation
        for constraint in env.safety_constraints:
            result = constraint.check_fn(obs, action)
            assert isinstance(result, (bool, np.bool_))
    
    def test_dataset_generation(self):
        """Test dataset generation for all environments."""
        env_classes = [ChemicalReactorEnv, PowerGridEnv, RobotAssemblyEnv]
        qualities = ['expert', 'medium', 'mixed', 'random']
        
        for env_class in env_classes:
            env = env_class()
            
            for quality in qualities:
                dataset = env.get_dataset(quality=quality)
                
                # Validate dataset structure
                required_keys = ['observations', 'actions', 'rewards', 'terminals']
                for key in required_keys:
                    assert key in dataset
                    assert len(dataset[key]) > 0
                
                # Validate shapes
                assert dataset['observations'].shape[1] == env.state_dim
                assert dataset['actions'].shape[1] == env.action_dim
                
                # Validate data consistency
                assert len(dataset['observations']) == len(dataset['actions'])
                assert len(dataset['actions']) == len(dataset['rewards'])

class TestAgentIntegration:
    """Test agent training and inference."""
    
    @pytest.fixture
    def sample_env(self):
        """Create sample environment for testing."""
        return ni.make('ChemicalReactor-v0')
    
    @pytest.fixture
    def sample_dataset(self, sample_env):
        """Create sample dataset for testing."""
        return sample_env.get_dataset(quality='random')
    
    def test_agent_creation(self):
        """Test all agent types can be created."""
        state_dim, action_dim = 12, 3
        
        agents = [
            CQLAgent(state_dim, action_dim),
            IQLAgent(state_dim, action_dim),
            TD3BCAgent(state_dim, action_dim),
        ]
        
        for agent in agents:
            assert agent.state_dim == state_dim
            assert agent.action_dim == action_dim
            assert hasattr(agent, 'train')
            assert hasattr(agent, 'predict')
    
    def test_agent_training_workflow(self, sample_env, sample_dataset):
        """Test complete agent training workflow."""
        agent = CQLAgent(
            state_dim=sample_env.state_dim,
            action_dim=sample_env.action_dim,
            safety_critic=True
        )
        
        # Ensure dataset is properly sized for testing
        dataset_size = min(1000, len(sample_dataset['observations']))
        test_dataset = {
            key: values[:dataset_size] for key, values in sample_dataset.items()
        }
        
        # Train agent
        training_metrics = agent.train(
            test_dataset,
            n_epochs=2,  # Short training for tests
            batch_size=32,
            eval_env=sample_env,
            eval_freq=1
        )
        
        # Validate training results
        assert isinstance(training_metrics, dict)
        assert 'training_metrics' in training_metrics
        assert len(training_metrics['training_metrics']) == 2  # 2 epochs
        assert agent.is_trained
    
    def test_agent_prediction(self, sample_env):
        """Test agent prediction after training."""
        agent = CQLAgent(
            state_dim=sample_env.state_dim,
            action_dim=sample_env.action_dim
        )
        
        # Create minimal dataset for training
        obs = np.random.randn(100, sample_env.state_dim).astype(np.float32)
        actions = np.random.uniform(-1, 1, (100, sample_env.action_dim)).astype(np.float32)
        rewards = np.random.randn(100).astype(np.float32)
        terminals = np.random.choice([True, False], 100)
        
        dataset = {
            'observations': obs,
            'actions': actions,
            'rewards': rewards,
            'terminals': terminals,
        }
        
        # Train briefly
        agent.train(dataset, n_epochs=1, batch_size=16)
        
        # Test prediction
        test_obs = np.random.randn(sample_env.state_dim).astype(np.float32)
        predicted_action = agent.predict(test_obs)
        
        assert predicted_action.shape == (sample_env.action_dim,)
        assert np.all(predicted_action >= -1.0) and np.all(predicted_action <= 1.0)
    
    def test_safety_critic_functionality(self, sample_env):
        """Test safety critic functionality."""
        agent = CQLAgent(
            state_dim=sample_env.state_dim,
            action_dim=sample_env.action_dim,
            safety_critic=True
        )
        
        # Create dataset with safety labels
        obs = np.random.randn(100, sample_env.state_dim).astype(np.float32)
        actions = np.random.uniform(-1, 1, (100, sample_env.action_dim)).astype(np.float32)
        rewards = np.random.randn(100).astype(np.float32)
        terminals = np.random.choice([True, False], 100)
        
        dataset = {
            'observations': obs,
            'actions': actions,
            'rewards': rewards,
            'terminals': terminals,
        }
        
        # Train
        agent.train(dataset, n_epochs=1, batch_size=16)
        
        # Test safety prediction
        test_obs = np.random.randn(1, sample_env.state_dim).astype(np.float32)
        actions, safety_probs = agent.predict_with_safety(test_obs)
        
        assert actions.shape == (1, sample_env.action_dim)
        assert safety_probs.shape == (1,)
        assert 0 <= safety_probs[0] <= 1

class TestMonitoringSystem:
    """Test monitoring and alerting system."""
    
    def test_dashboard_metrics_recording(self):
        """Test metrics recording and retrieval."""
        dashboard = MonitoringDashboard()
        
        # Record test metrics
        metrics = [
            SystemMetric("test_metric_1", 0.5, time.time(), {"component": "test"}),
            SystemMetric("test_metric_2", 0.8, time.time(), {"component": "test"}),
        ]
        
        for metric in metrics:
            dashboard.record_metric(metric)
        
        # Test retrieval
        current_metrics = dashboard.get_current_metrics()
        assert "test_metric_1" in current_metrics
        assert "test_metric_2" in current_metrics
        
        # Test history
        history = dashboard.get_metric_history("test_metric_1")
        assert len(history) == 1
    
    def test_alert_system(self):
        """Test alert triggering and management."""
        dashboard = MonitoringDashboard()
        
        # Record metric that should trigger alert
        metric = SystemMetric(
            "high_latency", 
            1.5, 
            time.time(), 
            {"component": "api"},
            alert_threshold=1.0,
            critical_threshold=2.0
        )
        
        dashboard.record_metric(metric)
        
        # Check alerts
        alerts = dashboard.get_active_alerts()
        assert len(alerts) > 0
        assert any(alert.level == "WARNING" for alert in alerts)
    
    def test_health_reporting(self):
        """Test system health reporting."""
        dashboard = MonitoringDashboard()
        
        # Record some metrics
        metrics = [
            SystemMetric("cpu_usage", 0.3, time.time(), {"component": "system"}),
            SystemMetric("memory_usage", 0.7, time.time(), {"component": "system"}),
        ]
        
        for metric in metrics:
            dashboard.record_metric(metric)
        
        # Generate health report
        health_report = dashboard.generate_health_report()
        
        assert "system_health" in health_report
        assert "recent_metrics" in health_report
        assert health_report["system_health"]["overall_status"] in ["HEALTHY", "WARNING", "CRITICAL"]

class TestResilienceSystem:
    """Test resilience components."""
    
    def test_circuit_breaker_basic_functionality(self):
        """Test basic circuit breaker functionality."""
        breaker = AdaptiveCircuitBreaker(
            name="test_breaker",
            failure_threshold=3,
            timeout_duration=1.0
        )
        
        # Test successful calls
        def success_func():
            return "success"
        
        result = breaker.call(success_func)
        assert result == "success"
        
        # Test failure handling
        def failing_func():
            raise Exception("test failure")
        
        # Trigger failures to open circuit
        for _ in range(4):
            try:
                breaker.call(failing_func)
            except:
                pass
        
        # Circuit should be open now
        metrics = breaker.get_metrics()
        assert metrics["state"] == "open"
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery mechanism."""
        breaker = AdaptiveCircuitBreaker(
            name="recovery_test",
            failure_threshold=2,
            timeout_duration=0.1,  # Short timeout for testing
            success_threshold=2
        )
        
        # Force circuit to open
        def failing_func():
            raise Exception("failure")
        
        for _ in range(3):
            try:
                breaker.call(failing_func)
            except:
                pass
        
        assert breaker.get_metrics()["state"] == "open"
        
        # Wait for timeout
        time.sleep(0.2)
        
        # Test recovery with successful calls
        def success_func():
            return "ok"
        
        # Should transition to half-open, then closed
        result1 = breaker.call(success_func)
        result2 = breaker.call(success_func)
        
        assert result1 == "ok"
        assert result2 == "ok"
        assert breaker.get_metrics()["state"] == "closed"

class TestValidationSystem:
    """Test validation framework."""
    
    def test_data_shape_validation(self):
        """Test data shape validation."""
        from neorl_industrial.validation.comprehensive_validator import DataShapeValidator
        
        validator = DataShapeValidator({
            "observations": (12,),
            "actions": (3,),
        })
        
        # Valid data
        valid_data = {
            "observations": np.random.randn(10, 12),
            "actions": np.random.randn(10, 3),
        }
        
        result = validator.validate(valid_data)
        assert result.is_valid
        assert result.confidence == 1.0
        
        # Invalid data
        invalid_data = {
            "observations": np.random.randn(10, 15),  # Wrong shape
            "actions": np.random.randn(10, 3),
        }
        
        result = validator.validate(invalid_data)
        assert not result.is_valid
    
    def test_safety_constraint_validation(self):
        """Test safety constraint validation."""
        from neorl_industrial.validation.comprehensive_validator import SafetyConstraintValidator
        
        # Create test constraint
        def test_constraint(state, action):
            return np.all(state >= -1) and np.all(state <= 1)
        
        constraint = SafetyConstraint(
            name="test_constraint",
            check_fn=test_constraint,
            penalty=-10.0
        )
        
        validator = SafetyConstraintValidator([constraint])
        
        # Valid data
        valid_data = {
            "observations": np.random.uniform(-0.5, 0.5, (10, 5)),
            "actions": np.random.uniform(-0.5, 0.5, (10, 3)),
        }
        
        result = validator.validate(valid_data)
        assert result.is_valid
        
        # Invalid data
        invalid_data = {
            "observations": np.random.uniform(-2, 2, (10, 5)),  # Outside bounds
            "actions": np.random.uniform(-0.5, 0.5, (10, 3)),
        }
        
        result = validator.validate(invalid_data)
        assert not result.is_valid
    
    def test_comprehensive_validator(self):
        """Test comprehensive validation system."""
        validator = ComprehensiveValidator()
        
        # Add multiple validators
        from neorl_industrial.validation.comprehensive_validator import (
            DataShapeValidator, DataRangeValidator
        )
        
        validator.add_validator(DataShapeValidator({
            "observations": (5,),
            "actions": (2,),
        }))
        
        validator.add_validator(DataRangeValidator({
            "observations": (-1.0, 1.0),
            "actions": (-1.0, 1.0),
        }))
        
        # Test with valid data
        valid_data = {
            "observations": np.random.uniform(-0.5, 0.5, (10, 5)),
            "actions": np.random.uniform(-0.5, 0.5, (10, 2)),
        }
        
        results = validator.validate_all(valid_data)
        
        assert len(results) == 2
        assert all(result.is_valid for result in results.values())
        
        is_valid, confidence = validator.get_overall_validity(results)
        assert is_valid
        assert confidence > 0.5

class TestOptimizationSystem:
    """Test optimization components."""
    
    def test_intelligent_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = IntelligentCache(max_size_mb=1, max_entries=100)
        
        # Test put and get
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None
        
        # Test with default
        assert cache.get("nonexistent", "default") == "default"
        
        # Test removal
        assert cache.remove("key1")
        assert cache.get("key1") is None
        assert not cache.remove("nonexistent")
    
    def test_cache_eviction(self):
        """Test cache eviction policies."""
        cache = IntelligentCache(max_size_mb=0.001, max_entries=5)  # Very small cache
        
        # Fill cache beyond capacity
        for i in range(10):
            cache.put(f"key{i}", f"value{i}" * 1000)  # Large values
        
        stats = cache.get_stats()
        assert stats["entries"] <= 5  # Should not exceed max_entries
        assert stats["evictions"] > 0  # Should have evicted some entries
    
    def test_cache_ttl(self):
        """Test cache TTL functionality."""
        cache = IntelligentCache()
        
        # Put with short TTL
        cache.put("temp_key", "temp_value", ttl=0.1)
        assert cache.get("temp_key") == "temp_value"
        
        # Wait for expiration
        time.sleep(0.2)
        assert cache.get("temp_key") is None
    
    def test_cache_decorator(self):
        """Test cache decorator functionality."""
        from neorl_industrial.optimization.intelligent_caching import cached
        
        call_count = 0
        
        @cached(ttl=1.0)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should compute
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Should not increment
        
        # Different arguments should compute
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2

class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_complete_training_pipeline(self):
        """Test complete training pipeline from environment to trained agent."""
        # Create environment
        env = ni.make('ChemicalReactor-v0')
        
        # Generate dataset
        dataset = env.get_dataset(quality='mixed')
        
        # Subsample for testing
        sample_size = min(500, len(dataset['observations']))
        test_dataset = {
            key: values[:sample_size] for key, values in dataset.items()
        }
        
        # Create and train agent
        agent = CQLAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            safety_critic=True
        )
        
        # Train
        metrics = agent.train(
            test_dataset,
            n_epochs=2,
            batch_size=32,
            eval_env=env,
            eval_freq=1
        )
        
        # Evaluate trained agent
        eval_metrics = ni.evaluate_with_safety(
            agent,
            env,
            n_episodes=5,
            render=False
        )
        
        # Validate results
        assert agent.is_trained
        assert 'return_mean' in eval_metrics
        assert 'safety_violations' in eval_metrics
        assert eval_metrics['return_mean'] is not None
        assert eval_metrics['safety_violations'] >= 0
    
    def test_multi_environment_compatibility(self):
        """Test that agents work across different environments."""
        environments = [
            ('ChemicalReactor-v0', 12, 3),
            ('PowerGrid-v0', 32, 8),
            ('RobotAssembly-v0', 24, 7),
        ]
        
        for env_id, state_dim, action_dim in environments:
            env = ni.make(env_id)
            assert env.state_dim == state_dim
            assert env.action_dim == action_dim
            
            # Test that agent can be created for this environment
            agent = CQLAgent(
                state_dim=state_dim,
                action_dim=action_dim
            )
            
            # Test basic interaction
            obs, _ = env.reset()
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Validate basic properties
            assert obs.shape == (state_dim,)
            assert next_obs.shape == (state_dim,)
            assert len(action) == action_dim
    
    def test_safety_constraint_integration(self):
        """Test safety constraint integration across the system."""
        env = ni.make('ChemicalReactor-v0')
        
        # Test environment safety constraints
        assert len(env.safety_constraints) > 0
        
        # Test that constraints are evaluated during episodes
        obs, _ = env.reset()
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Should have safety metrics in info
        if 'safety_metrics' in info:
            safety_metrics = info['safety_metrics']
            assert hasattr(safety_metrics, 'violation_count')
            assert hasattr(safety_metrics, 'safety_score')

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Clean up any global state
    try:
        from neorl_industrial.optimization.intelligent_caching import _global_cache
        if _global_cache is not None:
            _global_cache.shutdown()
            _global_cache = None
    except:
        pass

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])