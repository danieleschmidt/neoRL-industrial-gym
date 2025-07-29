"""Pytest configuration and fixtures for neoRL-industrial-gym tests."""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_data():
    """Provide sample industrial data for testing."""
    import numpy as np
    
    return {
        "states": np.random.randn(100, 12),
        "actions": np.random.randn(100, 3),
        "rewards": np.random.randn(100),
        "dones": np.random.choice([True, False], 100),
        "safety_constraints": np.random.choice([True, False], 100),
    }