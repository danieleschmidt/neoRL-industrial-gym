"""Basic tests for neoRL-industrial-gym package."""

import pytest
import neorl_industrial


def test_import():
    """Test that the package can be imported."""
    assert hasattr(neorl_industrial, "__version__")
    assert hasattr(neorl_industrial, "__author__")


def test_version_format():
    """Test that version follows semantic versioning."""
    version = neorl_industrial.__version__
    if version != "unknown":
        parts = version.split(".")
        assert len(parts) >= 2, "Version should have at least major.minor"