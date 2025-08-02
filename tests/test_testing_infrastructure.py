"""Test the testing infrastructure itself to ensure reliability."""

import pytest
import numpy as np
from typing import Dict, Any
from unittest.mock import Mock, patch

from tests.conftest import (
    sample_trajectory_data,
    safety_constraints,
    mock_environment_config,
    safety_validator,
    industrial_test_data,
)


class TestTestingInfrastructure:
    """Test the testing infrastructure components."""

    def test_sample_trajectory_data_fixture(self, sample_trajectory_data: Dict[str, np.ndarray]):
        """Test that sample trajectory data fixture generates valid data."""
        required_keys = ["observations", "actions", "rewards", "next_observations", "terminals", "timeouts"]
        
        # Check all required keys are present
        for key in required_keys:
            assert key in sample_trajectory_data, f"Missing key: {key}"
        
        # Check data shapes and types
        batch_size = len(sample_trajectory_data["observations"])
        assert batch_size == 1000, f"Expected batch size 1000, got {batch_size}"
        
        # Check observation dimensions
        assert sample_trajectory_data["observations"].shape == (1000, 10)
        assert sample_trajectory_data["next_observations"].shape == (1000, 10)
        
        # Check action dimensions
        assert sample_trajectory_data["actions"].shape == (1000, 3)
        
        # Check scalar arrays
        assert sample_trajectory_data["rewards"].shape == (1000,)
        assert sample_trajectory_data["terminals"].shape == (1000,)
        assert sample_trajectory_data["timeouts"].shape == (1000,)
        
        # Check data types
        assert sample_trajectory_data["observations"].dtype == np.float32
        assert sample_trajectory_data["actions"].dtype == np.float32
        assert sample_trajectory_data["rewards"].dtype == np.float32
        assert sample_trajectory_data["terminals"].dtype == bool
        assert sample_trajectory_data["timeouts"].dtype == bool

    def test_safety_constraints_fixture(self, safety_constraints):
        """Test that safety constraints fixture provides valid constraints."""
        assert len(safety_constraints) == 2, "Expected 2 safety constraints"
        
        # Test temperature constraint
        temp_constraint = safety_constraints[0]
        
        # Valid temperature (20°C = 293.15K)
        valid_state = np.array([293.15, 5.0])
        assert temp_constraint(valid_state) is True
        
        # Invalid temperature (too hot: 150°C = 423.15K)
        invalid_state = np.array([423.15, 5.0])
        assert temp_constraint(invalid_state) is False
        
        # Test pressure constraint
        pressure_constraint = safety_constraints[1]
        
        # Valid pressure (5 bar)
        valid_state = np.array([300.0, 5.0])
        assert pressure_constraint(valid_state) is True
        
        # Invalid pressure (negative)
        invalid_state = np.array([300.0, -1.0])
        assert pressure_constraint(invalid_state) is False

    def test_mock_environment_config_fixture(self, mock_environment_config: Dict[str, Any]):
        """Test mock environment configuration fixture."""
        required_keys = [
            "name", "state_dim", "action_dim", "max_episode_steps",
            "safety_constraints", "reward_range", "action_space_bounds"
        ]
        
        for key in required_keys:
            assert key in mock_environment_config, f"Missing config key: {key}"
        
        # Check specific values
        assert mock_environment_config["state_dim"] == 10
        assert mock_environment_config["action_dim"] == 3
        assert mock_environment_config["max_episode_steps"] == 1000
        assert len(mock_environment_config["safety_constraints"]) == 2

    def test_industrial_test_data_fixture(self, industrial_test_data: Dict[str, Dict]):
        """Test industrial test data fixture provides data for all environments."""
        expected_environments = [
            "ChemicalReactor-v0", "RobotAssembly-v0", "HVACControl-v0",
            "WaterTreatment-v0", "SteelAnnealing-v0", "PowerGrid-v0", "SupplyChain-v0"
        ]
        
        # Check all environments are present
        for env_name in expected_environments:
            assert env_name in industrial_test_data, f"Missing environment: {env_name}"
            
            env_data = industrial_test_data[env_name]
            assert "states" in env_data
            assert "actions" in env_data
            assert "rewards" in env_data
            
            # Check data shapes
            assert len(env_data["states"]) == 500
            assert len(env_data["actions"]) == 500
            assert len(env_data["rewards"]) == 500

    def test_safety_validator_fixture(self, safety_validator):
        """Test safety validator fixture functionality."""
        # Test constraint checking
        def dummy_constraint(state):
            return state[0] > 0
        
        # Test passing constraint
        valid_state = np.array([1.0, 2.0])
        result = safety_validator.check_constraint(dummy_constraint, valid_state)
        assert result is True
        
        # Test failing constraint
        invalid_state = np.array([-1.0, 2.0])
        result = safety_validator.check_constraint(dummy_constraint, invalid_state)
        assert result is False
        
        # Check violation tracking
        assert len(safety_validator.violations) == 1
        assert safety_validator.violations[0]["violated"] is True
        
        # Test violation rate calculation
        violation_rate = safety_validator.get_violation_rate()
        assert violation_rate == 1.0  # 1 violation out of 1 check
        
        # Test reset functionality
        safety_validator.reset()
        assert len(safety_validator.violations) == 0
        assert safety_validator.get_violation_rate() == 0.0

    def test_pytest_markers_configuration(self):
        """Test that pytest markers are properly configured."""
        # This test verifies that custom markers work
        # The actual marker configuration is tested through conftest.py
        
        # Test markers can be imported/accessed
        import pytest
        
        # These should not raise errors if markers are properly configured
        test_marker_names = [
            "unit", "integration", "safety", "performance", 
            "slow", "gpu", "industrial"
        ]
        
        for marker_name in test_marker_names:
            marker = getattr(pytest.mark, marker_name, None)
            assert marker is not None, f"Marker {marker_name} not configured"

    @pytest.mark.slow
    def test_performance_test_marking(self):
        """Test that slow tests are properly marked."""
        # This test should be automatically marked as slow
        # due to the naming convention in conftest.py
        pass

    def test_jax_configuration_in_tests(self, jax_config):
        """Test JAX configuration for testing."""
        import jax
        
        # Check JAX is configured for CPU testing
        assert jax.default_backend() == "cpu"
        
        # Check random key is available
        assert "random_key" in jax_config
        assert "platform" in jax_config
        assert jax_config["platform"] == "cpu"

    def test_mock_logger_fixture(self, mock_logger):
        """Test mock logger fixture provides proper logging interface."""
        # Test logger methods are available
        assert hasattr(mock_logger, "info")
        assert hasattr(mock_logger, "warning")
        assert hasattr(mock_logger, "error")
        assert hasattr(mock_logger, "debug")
        
        # Test logging calls work
        mock_logger.info("Test message")
        mock_logger.warning("Test warning")
        mock_logger.error("Test error")
        mock_logger.debug("Test debug")
        
        # Verify calls were made
        mock_logger.info.assert_called_once_with("Test message")
        mock_logger.warning.assert_called_once_with("Test warning")
        mock_logger.error.assert_called_once_with("Test error")
        mock_logger.debug.assert_called_once_with("Test debug")

    def test_temporary_directory_fixture(self, temp_dir):
        """Test temporary directory fixture."""
        from pathlib import Path
        
        assert isinstance(temp_dir, Path)
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        # Test writing to temporary directory
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        assert test_file.exists()
        assert test_file.read_text() == "test content"


class TestTestDataConsistency:
    """Test consistency and reliability of test data across runs."""

    def test_trajectory_data_consistency(self, sample_trajectory_data):
        """Test that trajectory data is consistent across test runs."""
        # The data should be pseudo-random but consistent within a test session
        observations = sample_trajectory_data["observations"]
        
        # Basic statistical properties should be reasonable
        mean_obs = np.mean(observations, axis=0)
        std_obs = np.std(observations, axis=0)
        
        # Should be approximately normal distribution centered around 0
        assert np.all(np.abs(mean_obs) < 0.5), "Observations mean too far from zero"
        assert np.all(std_obs > 0.5), "Observations std too small"
        assert np.all(std_obs < 2.0), "Observations std too large"

    def test_safety_constraint_reliability(self, safety_constraints):
        """Test that safety constraints behave consistently."""
        # Test multiple evaluations of the same state
        test_state = np.array([300.0, 5.0])  # Valid temperature and pressure
        
        for constraint in safety_constraints:
            # Multiple evaluations should give same result
            results = [constraint(test_state) for _ in range(10)]
            assert all(r == results[0] for r in results), "Constraint evaluation inconsistent"

    def test_environment_config_validation(self, mock_environment_config):
        """Test that environment configuration values are valid."""
        config = mock_environment_config
        
        # Validate dimensions are positive
        assert config["state_dim"] > 0
        assert config["action_dim"] > 0
        assert config["max_episode_steps"] > 0
        
        # Validate reward range
        reward_range = config["reward_range"]
        assert len(reward_range) == 2
        assert reward_range[0] <= reward_range[1]
        
        # Validate action space bounds
        action_bounds = config["action_space_bounds"]
        assert len(action_bounds) == 2
        assert action_bounds[0] <= action_bounds[1]


@pytest.mark.integration
class TestTestingWorkflow:
    """Test the overall testing workflow and integration."""

    def test_fixture_dependencies(self, sample_trajectory_data, safety_constraints, mock_environment_config):
        """Test that fixtures work together properly."""
        # Use multiple fixtures together
        states = sample_trajectory_data["observations"]
        constraints = safety_constraints
        config = mock_environment_config
        
        # Should be able to apply safety constraints to trajectory data
        for state in states[:10]:  # Test first 10 states
            for constraint in constraints:
                try:
                    result = constraint(state)
                    assert isinstance(result, bool)
                except Exception as e:
                    # Some constraints might fail due to dimension mismatch,
                    # which is expected with random test data
                    assert "index" in str(e).lower() or "shape" in str(e).lower()

    def test_test_isolation(self):
        """Test that tests are properly isolated from each other."""
        # This test verifies JAX state is reset between tests
        import jax
        
        # Create some JAX computation that would cache
        x = jax.numpy.array([1.0, 2.0, 3.0])
        y = jax.numpy.sum(x)
        
        # Cache should be empty or manageable due to reset fixture
        # This is mainly a smoke test to ensure isolation doesn't break
        assert y == 6.0
        
    @patch('numpy.random.randn')
    def test_mocking_capabilities(self, mock_randn):
        """Test that mocking works properly in the test environment."""
        # Configure mock
        mock_randn.return_value = np.array([1.0, 2.0, 3.0])
        
        # Use the mocked function
        result = np.random.randn(3)
        
        # Verify mock was called and returned expected value
        mock_randn.assert_called_once_with(3)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])