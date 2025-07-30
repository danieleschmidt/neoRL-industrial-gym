"""Pytest configuration and shared fixtures for neoRL-industrial-gym tests."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
import warnings

import pytest
import numpy as np
import jax
import jax.numpy as jnp

# Suppress JAX warnings during testing
warnings.filterwarnings("ignore", category=UserWarning, module="jax")
warnings.filterwarnings("ignore", category=FutureWarning, module="jax")

# Configure JAX for testing
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "false"  # Use float32 for faster tests


@pytest.fixture(scope="session")
def jax_config():
    """Configure JAX for testing session."""
    # Set deterministic random key for reproducible tests
    key = jax.random.PRNGKey(42)
    return {"random_key": key, "platform": "cpu"}


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_trajectory_data():
    """Generate sample trajectory data for testing."""
    batch_size = 1000
    state_dim = 10
    action_dim = 3
    
    return {
        "observations": np.random.randn(batch_size, state_dim).astype(np.float32),
        "actions": np.random.randn(batch_size, action_dim).astype(np.float32),
        "rewards": np.random.randn(batch_size).astype(np.float32),
        "next_observations": np.random.randn(batch_size, state_dim).astype(np.float32),
        "terminals": np.random.choice([0, 1], size=batch_size).astype(bool),
        "timeouts": np.random.choice([0, 1], size=batch_size).astype(bool),
    }


@pytest.fixture
def sample_data():
    """Provide sample industrial data for testing (legacy fixture)."""
    return {
        "states": np.random.randn(100, 12),
        "actions": np.random.randn(100, 3),
        "rewards": np.random.randn(100),
        "dones": np.random.choice([True, False], 100),
        "safety_constraints": np.random.choice([True, False], 100),
    }


@pytest.fixture
def safety_constraints():
    """Define sample safety constraints for testing."""
    def temperature_constraint(state: jnp.ndarray) -> bool:
        """Temperature must be within safe operating range."""
        temp = state[0]  # Assume first state element is temperature
        return 273.15 <= temp <= 373.15  # 0°C to 100°C
    
    def pressure_constraint(state: jnp.ndarray) -> bool:
        """Pressure must be within safe operating range."""
        pressure = state[1]  # Assume second state element is pressure
        return 0.0 <= pressure <= 10.0  # 0 to 10 bar
    
    return [temperature_constraint, pressure_constraint]


@pytest.fixture
def mock_environment_config():
    """Mock environment configuration for testing."""
    return {
        "name": "TestEnvironment-v0",
        "state_dim": 10,
        "action_dim": 3,
        "max_episode_steps": 1000,
        "safety_constraints": ["temperature", "pressure"],
        "reward_range": (-100.0, 100.0),
        "action_space_bounds": (-1.0, 1.0),
    }


@pytest.fixture
def benchmark_config():
    """Configuration for performance benchmarking."""
    return {
        "num_episodes": 10,
        "max_steps_per_episode": 100,
        "num_trials": 3,
        "timeout_seconds": 30,
        "memory_limit_mb": 1024,
    }


@pytest.fixture(scope="session")
def industrial_test_data():
    """Load or generate industrial test datasets."""
    # In a real implementation, this would load standardized test data
    # For now, generate synthetic industrial-like data
    
    environments = [
        "ChemicalReactor-v0",
        "RobotAssembly-v0", 
        "HVACControl-v0",
        "WaterTreatment-v0",
        "SteelAnnealing-v0",
        "PowerGrid-v0",
        "SupplyChain-v0"
    ]
    
    test_data = {}
    for env_name in environments:
        # Generate environment-specific test data
        state_dims = {
            "ChemicalReactor-v0": 12,
            "RobotAssembly-v0": 24,
            "HVACControl-v0": 18,
            "WaterTreatment-v0": 15,
            "SteelAnnealing-v0": 20,
            "PowerGrid-v0": 32,
            "SupplyChain-v0": 28,
        }
        
        action_dims = {
            "ChemicalReactor-v0": 3,
            "RobotAssembly-v0": 7,
            "HVACControl-v0": 5,
            "WaterTreatment-v0": 4,
            "SteelAnnealing-v0": 6,
            "PowerGrid-v0": 8,
            "SupplyChain-v0": 10,
        }
        
        batch_size = 500
        test_data[env_name] = {
            "states": np.random.randn(batch_size, state_dims[env_name]).astype(np.float32),
            "actions": np.random.randn(batch_size, action_dims[env_name]).astype(np.float32),
            "rewards": np.random.randn(batch_size).astype(np.float32),
        }
    
    return test_data


@pytest.fixture
def safety_monitoring_config():
    """Configuration for safety monitoring during tests."""
    return {
        "enable_monitoring": True,
        "violation_threshold": 0.01,  # 1% violation rate threshold
        "emergency_shutdown_enabled": True,
        "log_safety_events": True,
        "safety_buffer_size": 1000,
    }


# Pytest hooks for enhanced test management

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "safety: mark test as a safety validation test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "industrial: mark test as industrial environment test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file/function names."""
    for item in items:
        # Auto-mark tests based on naming conventions
        if "test_safety" in item.name or "safety" in str(item.fspath):
            item.add_marker(pytest.mark.safety)
        
        if "test_performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)
            
        if "test_integration" in item.name or "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
            
        if "gpu" in item.name or "cuda" in item.name:
            item.add_marker(pytest.mark.gpu)
            
        # Mark slow tests
        if any(keyword in item.name.lower() for keyword in ["slow", "long", "stress"]):
            item.add_marker(pytest.mark.slow)


def pytest_runtest_setup(item):
    """Setup hook called before each test."""
    # Skip GPU tests if no GPU available
    if "gpu" in item.keywords:
        try:
            import jax
            if jax.default_backend() == "cpu":
                pytest.skip("GPU not available")
        except Exception:
            pytest.skip("JAX GPU check failed")
    
    # Skip slow tests in CI unless explicitly requested
    if "slow" in item.keywords and os.environ.get("CI") and not os.environ.get("RUN_SLOW_TESTS"):
        pytest.skip("Slow test skipped in CI")


@pytest.fixture(autouse=True)
def reset_jax_state():
    """Reset JAX state between tests to ensure isolation."""
    yield
    # Clear JAX caches
    jax.clear_caches()


@pytest.fixture
def mock_logger():
    """Mock logger for testing logging functionality."""
    import logging
    from unittest.mock import Mock
    
    logger = Mock(spec=logging.Logger)
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.debug = Mock()
    
    return logger


@pytest.fixture
def safety_validator():
    """Safety validation utilities for testing."""
    class SafetyValidator:
        def __init__(self):
            self.violations = []
            self.monitoring_enabled = True
        
        def check_constraint(self, constraint_func, state):
            """Check if a safety constraint is satisfied."""
            try:
                result = constraint_func(state)
                if not result:
                    self.violations.append({
                        "constraint": constraint_func.__name__,
                        "state": state,
                        "violated": True
                    })
                return result
            except Exception as e:
                self.violations.append({
                    "constraint": constraint_func.__name__,
                    "state": state,
                    "error": str(e)
                })
                return False
        
        def get_violation_rate(self):
            """Calculate the violation rate."""
            total_checks = len(self.violations)
            if total_checks == 0:
                return 0.0
            violations = sum(1 for v in self.violations if v.get("violated", False))
            return violations / total_checks
        
        def reset(self):
            """Reset violation tracking."""
            self.violations.clear()
    
    return SafetyValidator()