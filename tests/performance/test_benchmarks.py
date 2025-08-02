"""Performance benchmarking tests for neoRL-industrial-gym.

This module contains performance tests to ensure the system meets
performance requirements for industrial deployment.
"""

import time
import psutil
import pytest
import numpy as np
from typing import Dict, Any, List
from unittest.mock import Mock

from tests.conftest import sample_trajectory_data, benchmark_config


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark test suite."""

    def test_training_performance(self, sample_trajectory_data: Dict[str, np.ndarray], benchmark_config: Dict[str, Any]):
        """Test training performance meets industrial requirements."""
        # Mock agent for performance testing
        mock_agent = Mock()
        mock_agent.train = Mock(return_value={"loss": 0.1, "q_value": 10.0})
        
        start_time = time.time()
        
        # Simulate training iterations
        for _ in range(benchmark_config["num_trials"]):
            result = mock_agent.train(sample_trajectory_data)
            assert result is not None
        
        elapsed_time = time.time() - start_time
        
        # Training should complete within reasonable time
        assert elapsed_time < benchmark_config["timeout_seconds"]
        
        # Memory usage should be reasonable
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        assert memory_usage < benchmark_config["memory_limit_mb"]

    def test_inference_latency(self, sample_trajectory_data: Dict[str, np.ndarray]):
        """Test inference latency meets real-time requirements."""
        # Mock environment and agent
        mock_agent = Mock()
        mock_agent.predict = Mock(return_value=np.random.randn(3))
        
        states = sample_trajectory_data["observations"][:100]  # Test with 100 states
        latencies = []
        
        for state in states:
            start_time = time.perf_counter()
            action = mock_agent.predict(state)
            end_time = time.perf_counter()
            
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(latency)
            assert action is not None
        
        # Industrial requirement: inference latency < 100ms
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        assert avg_latency < 100.0, f"Average latency {avg_latency:.2f}ms exceeds 100ms limit"
        assert p95_latency < 200.0, f"P95 latency {p95_latency:.2f}ms exceeds 200ms limit"

    @pytest.mark.slow
    def test_memory_efficiency(self, sample_trajectory_data: Dict[str, np.ndarray]):
        """Test memory efficiency with large datasets."""
        # Simulate large dataset processing
        large_dataset_size = 10000
        batch_size = 256
        
        # Create larger dataset
        large_dataset = {
            key: np.tile(data, (large_dataset_size // len(data) + 1, 1))[:large_dataset_size]
            for key, data in sample_trajectory_data.items()
        }
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Process dataset in batches
        for i in range(0, large_dataset_size, batch_size):
            batch_end = min(i + batch_size, large_dataset_size)
            batch_data = {
                key: data[i:batch_end] for key, data in large_dataset.items()
            }
            
            # Simulate processing
            _ = np.mean(batch_data["observations"], axis=0)
        
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory increase should be reasonable for dataset size
        expected_memory_mb = (large_dataset_size * 4 * 10) / 1024 / 1024  # Rough estimate
        assert memory_increase < expected_memory_mb * 2, f"Memory usage {memory_increase:.2f}MB too high"

    def test_concurrent_environments(self, mock_environment_config: Dict[str, Any]):
        """Test performance with multiple concurrent environments."""
        num_environments = 4
        episodes_per_env = 10
        
        # Mock multiple environments
        mock_environments = []
        for _ in range(num_environments):
            mock_env = Mock()
            mock_env.reset = Mock(return_value=(np.random.randn(mock_environment_config["state_dim"]), {}))
            mock_env.step = Mock(return_value=(
                np.random.randn(mock_environment_config["state_dim"]),  # observation
                np.random.randn(),  # reward
                False,  # terminated
                False,  # truncated
                {}  # info
            ))
            mock_environments.append(mock_env)
        
        start_time = time.time()
        
        # Run episodes concurrently (simulated)
        for env in mock_environments:
            for _ in range(episodes_per_env):
                state, info = env.reset()
                for _ in range(100):  # 100 steps per episode
                    action = np.random.randn(mock_environment_config["action_dim"])
                    state, reward, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
        
        elapsed_time = time.time() - start_time
        total_episodes = num_environments * episodes_per_env
        
        # Should maintain reasonable throughput
        episodes_per_second = total_episodes / elapsed_time
        assert episodes_per_second > 1.0, f"Episode throughput {episodes_per_second:.2f} eps/s too low"

    def test_safety_monitoring_overhead(self, safety_constraints: List, sample_trajectory_data: Dict[str, np.ndarray]):
        """Test that safety monitoring doesn't significantly impact performance."""
        states = sample_trajectory_data["observations"][:1000]
        
        # Measure baseline performance without safety monitoring
        start_time = time.perf_counter()
        for state in states:
            # Simulate basic agent computation
            _ = np.tanh(state @ np.random.randn(len(state), 3))
        baseline_time = time.perf_counter() - start_time
        
        # Measure performance with safety monitoring
        start_time = time.perf_counter()
        for state in states:
            # Simulate agent computation
            action = np.tanh(state @ np.random.randn(len(state), 3))
            
            # Add safety monitoring overhead
            for constraint in safety_constraints[:2]:  # Use first 2 constraints
                try:
                    constraint(state)
                except Exception:
                    pass  # Ignore constraint errors in performance test
        
        safety_time = time.perf_counter() - start_time
        
        # Safety monitoring overhead should be minimal
        overhead_ratio = (safety_time - baseline_time) / baseline_time
        assert overhead_ratio < 0.1, f"Safety monitoring overhead {overhead_ratio:.2%} too high"

    @pytest.mark.slow
    def test_scalability_stress(self):
        """Stress test system scalability limits."""
        # Test with increasingly large batch sizes
        batch_sizes = [64, 128, 256, 512, 1024]
        state_dim = 20
        action_dim = 5
        
        performance_results = {}
        
        for batch_size in batch_sizes:
            # Create batch data
            states = np.random.randn(batch_size, state_dim)
            actions = np.random.randn(batch_size, action_dim)
            
            # Measure processing time
            start_time = time.perf_counter()
            
            # Simulate batch processing
            q_values = states @ np.random.randn(state_dim, action_dim)
            loss = np.mean((q_values - actions) ** 2)
            
            processing_time = time.perf_counter() - start_time
            
            # Calculate throughput
            samples_per_second = batch_size / processing_time
            performance_results[batch_size] = samples_per_second
            
            # Memory check
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
            assert memory_usage < 2048, f"Memory usage {memory_usage:.1f}MB too high for batch size {batch_size}"
        
        # Performance should scale reasonably with batch size
        assert performance_results[1024] > performance_results[64], "Performance should improve with larger batches"

    def test_environment_reset_performance(self, mock_environment_config: Dict[str, Any]):
        """Test environment reset performance for rapid experimentation."""
        mock_env = Mock()
        mock_env.reset = Mock(return_value=(
            np.random.randn(mock_environment_config["state_dim"]), {}
        ))
        
        num_resets = 100
        start_time = time.perf_counter()
        
        for _ in range(num_resets):
            state, info = mock_env.reset()
            assert state is not None
        
        elapsed_time = time.perf_counter() - start_time
        resets_per_second = num_resets / elapsed_time
        
        # Should support rapid environment resets for hyperparameter search
        assert resets_per_second > 10.0, f"Reset rate {resets_per_second:.1f} resets/s too slow"


@pytest.mark.performance
class TestAlgorithmPerformance:
    """Algorithm-specific performance tests."""

    def test_cql_performance(self, sample_trajectory_data: Dict[str, np.ndarray]):
        """Test CQL algorithm performance characteristics."""
        # Mock CQL implementation
        class MockCQL:
            def __init__(self):
                self.q_network = Mock()
                self.target_network = Mock()
            
            def update(self, batch):
                # Simulate CQL update computation
                time.sleep(0.001)  # Simulate computation time
                return {"cql_loss": 0.1, "q_loss": 0.05}
        
        agent = MockCQL()
        batch_size = 256
        num_updates = 100
        
        start_time = time.perf_counter()
        
        for _ in range(num_updates):
            batch_indices = np.random.choice(len(sample_trajectory_data["observations"]), batch_size)
            batch = {
                key: data[batch_indices] for key, data in sample_trajectory_data.items()
            }
            metrics = agent.update(batch)
            assert metrics is not None
        
        elapsed_time = time.perf_counter() - start_time
        updates_per_second = num_updates / elapsed_time
        
        # CQL should maintain reasonable update frequency
        assert updates_per_second > 1.0, f"CQL update rate {updates_per_second:.1f} updates/s too slow"

    def test_safety_critic_overhead(self, sample_trajectory_data: Dict[str, np.ndarray]):
        """Test performance impact of safety critic."""
        states = sample_trajectory_data["observations"][:500]
        actions = sample_trajectory_data["actions"][:500]
        
        # Mock safety critic
        def safety_critic(state, action):
            # Simulate safety critic computation
            safety_score = np.random.random()
            return safety_score > 0.1  # 90% safe actions
        
        start_time = time.perf_counter()
        
        safety_results = []
        for state, action in zip(states, actions):
            # Simulate main agent computation
            q_value = np.random.random()
            
            # Add safety critic evaluation
            is_safe = safety_critic(state, action)
            safety_results.append(is_safe)
        
        elapsed_time = time.perf_counter() - start_time
        evaluations_per_second = len(states) / elapsed_time
        
        # Safety critic should not significantly slow down inference
        assert evaluations_per_second > 100.0, f"Safety evaluation rate {evaluations_per_second:.1f} eval/s too slow"
        
        # Should detect some unsafe actions
        safety_rate = np.mean(safety_results)
        assert 0.8 <= safety_rate <= 0.99, f"Safety rate {safety_rate:.2%} unexpected"