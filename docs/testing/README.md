# Testing Guide

This document provides comprehensive guidance on testing practices for neoRL-industrial-gym.

## Testing Philosophy

Our testing strategy follows industrial software development best practices:

1. **Safety First**: All safety-critical components have comprehensive test coverage
2. **Performance Validated**: Performance tests ensure industrial deployment readiness
3. **Deterministic**: Tests are reproducible and deterministic across environments
4. **Comprehensive**: Unit, integration, end-to-end, and performance testing
5. **Automated**: All tests run automatically in CI/CD pipelines

## Test Structure

```
tests/
├── unit/                   # Unit tests for individual components
├── integration/           # Integration tests for component interactions
├── e2e/                   # End-to-end tests for complete workflows
├── performance/           # Performance and benchmarking tests
├── fixtures/              # Test data and shared fixtures
├── conftest.py           # Pytest configuration and shared fixtures
└── test_*.py             # General test modules
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit              # Unit tests only
pytest -m integration      # Integration tests only
pytest -m safety           # Safety validation tests
pytest -m performance      # Performance tests only

# Run with coverage
pytest --cov=neorl_industrial --cov-report=html

# Run in parallel
pytest -n auto             # Auto-detect CPU cores
pytest -n 4               # Use 4 workers
```

### Test Selection

```bash
# Run specific test file
pytest tests/unit/test_safety_constraints.py

# Run specific test function
pytest tests/unit/test_safety_constraints.py::test_temperature_constraint

# Run tests matching pattern
pytest -k "safety"         # All tests with "safety" in name
pytest -k "not slow"       # Exclude slow tests
```

### Environment-Specific Testing

```bash
# Skip slow tests (good for development)
pytest -m "not slow"

# Run GPU tests (if GPU available)
pytest -m gpu

# Run industrial environment tests
pytest -m industrial

# CI mode (skip slow tests, enable coverage)
CI=true pytest -m "not slow" --cov=neorl_industrial
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)

Test individual components in isolation:

```python
@pytest.mark.unit
def test_safety_constraint_validation():
    constraint = TemperatureConstraint(min_temp=273.15, max_temp=373.15)
    
    # Test valid temperature
    assert constraint.validate(300.0) is True
    
    # Test invalid temperatures
    assert constraint.validate(200.0) is False
    assert constraint.validate(400.0) is False
```

### Integration Tests (`@pytest.mark.integration`)

Test component interactions:

```python
@pytest.mark.integration
def test_agent_environment_interaction():
    env = make_environment("ChemicalReactor-v0")
    agent = CQLAgent(state_dim=12, action_dim=3)
    
    state, info = env.reset()
    action = agent.predict(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    
    assert next_state.shape == state.shape
    assert isinstance(reward, float)
```

### Safety Tests (`@pytest.mark.safety`)

Validate safety-critical functionality:

```python
@pytest.mark.safety
def test_emergency_shutdown_system():
    safety_monitor = SafetyMonitor(constraints=[pressure_limit, temperature_limit])
    
    # Simulate safety violation
    dangerous_state = create_dangerous_state()
    
    with pytest.raises(EmergencyShutdownTriggered):
        safety_monitor.evaluate(dangerous_state)
```

### Performance Tests (`@pytest.mark.performance`)

Validate performance requirements:

```python
@pytest.mark.performance
def test_inference_latency():
    agent = load_trained_agent()
    states = generate_test_states(1000)
    
    start_time = time.perf_counter()
    for state in states:
        action = agent.predict(state)
    elapsed = time.perf_counter() - start_time
    
    avg_latency = elapsed / len(states) * 1000  # ms
    assert avg_latency < 100.0  # Industrial requirement: < 100ms
```

## Writing Good Tests

### Test Naming Convention

- Test files: `test_<component>.py`
- Test functions: `test_<functionality>`
- Test classes: `Test<Component>`

### Test Structure (Arrange-Act-Assert)

```python
def test_cql_loss_computation():
    # Arrange
    batch_size = 256
    state_dim = 10
    action_dim = 3
    agent = CQLAgent(state_dim=state_dim, action_dim=action_dim)
    batch_data = create_test_batch(batch_size, state_dim, action_dim)
    
    # Act
    loss_dict = agent.compute_loss(batch_data)
    
    # Assert
    assert "cql_loss" in loss_dict
    assert "q_loss" in loss_dict
    assert loss_dict["cql_loss"] >= 0.0
    assert isinstance(loss_dict["q_loss"], float)
```

### Using Fixtures

```python
def test_with_fixtures(sample_trajectory_data, safety_constraints):
    # Use shared test data and configurations
    states = sample_trajectory_data["observations"]
    
    for constraint in safety_constraints:
        results = [constraint(state) for state in states[:10]]
        assert all(isinstance(r, bool) for r in results)
```

### Parameterized Tests

```python
@pytest.mark.parametrize("env_name,expected_state_dim", [
    ("ChemicalReactor-v0", 12),
    ("RobotAssembly-v0", 24),
    ("HVACControl-v0", 18),
])
def test_environment_dimensions(env_name, expected_state_dim):
    env = make_environment(env_name)
    state, info = env.reset()
    assert state.shape[0] == expected_state_dim
```

## Test Data Management

### Using Test Fixtures

Our test suite provides comprehensive fixtures in `conftest.py`:

- `sample_trajectory_data`: Standard D4RL-format trajectory data
- `safety_constraints`: Industrial safety constraint functions
- `mock_environment_config`: Environment configuration for testing
- `industrial_test_data`: Environment-specific test datasets
- `safety_validator`: Utilities for safety constraint testing

### Creating Custom Test Data

```python
def create_chemical_reactor_data(batch_size=1000):
    """Create realistic chemical reactor test data."""
    return {
        "temperature": np.random.normal(300, 10, batch_size),  # Kelvin
        "pressure": np.random.normal(2.0, 0.5, batch_size),   # Bar
        "flow_rate": np.random.normal(100, 20, batch_size),   # L/min
        "concentration": np.random.beta(2, 5, batch_size),    # Fraction
    }
```

## Mocking and Test Doubles

### Environment Mocking

```python
@pytest.fixture
def mock_environment():
    env = Mock()
    env.reset.return_value = (np.zeros(10), {})
    env.step.return_value = (np.zeros(10), 0.0, False, False, {})
    env.action_space.shape = (3,)
    env.observation_space.shape = (10,)
    return env
```

### Agent Mocking

```python
@pytest.fixture
def mock_agent():
    agent = Mock()
    agent.predict.return_value = np.zeros(3)
    agent.train.return_value = {"loss": 0.1}
    return agent
```

## Coverage Requirements

### Minimum Coverage Targets

- **Overall Coverage**: 85%
- **Safety Components**: 95%
- **Core Algorithms**: 90%
- **Environment Interfaces**: 90%
- **Utility Functions**: 80%

### Coverage Exclusions

- Third-party library integrations
- Development/debugging utilities
- Performance monitoring (unless safety-critical)

### Generating Coverage Reports

```bash
# HTML report (detailed)
pytest --cov=neorl_industrial --cov-report=html
open htmlcov/index.html

# Terminal report
pytest --cov=neorl_industrial --cov-report=term-missing

# XML report (for CI)
pytest --cov=neorl_industrial --cov-report=xml
```

## Continuous Integration

### GitHub Actions Integration

Tests run automatically on:
- Pull requests
- Pushes to main branch
- Nightly builds
- Release tags

### Test Matrix

We test across multiple configurations:
- Python versions: 3.8, 3.9, 3.10, 3.11
- Operating systems: Ubuntu, macOS, Windows
- JAX backends: CPU, GPU (when available)

### Performance Monitoring

Performance tests track:
- Inference latency trends
- Memory usage patterns
- Training throughput
- Safety monitoring overhead

## Debugging Tests

### Running Tests in Debug Mode

```bash
# Verbose output
pytest -v

# Show print statements
pytest -s

# Drop into debugger on failure
pytest --pdb

# Stop after first failure
pytest -x
```

### Common Test Issues

1. **Flaky Tests**: Use fixed random seeds and deterministic operations
2. **Memory Leaks**: Use `@pytest.fixture(autouse=True)` for cleanup
3. **Slow Tests**: Mark with `@pytest.mark.slow` and optimize
4. **Platform Dependencies**: Use `pytest.skip()` for unavailable features

## Best Practices

### Test Independence

```python
# Good: Each test is independent
def test_agent_training():
    agent = create_fresh_agent()
    # Test training logic

def test_agent_evaluation():
    agent = create_fresh_agent()
    # Test evaluation logic

# Bad: Tests depend on each other
agent = None  # Global state

def test_agent_training():
    global agent
    agent = create_agent()
    # Train agent

def test_agent_evaluation():  # Depends on previous test
    # Evaluate global agent
```

### Assertion Messages

```python
# Good: Descriptive assertion messages
assert latency < 100.0, f"Latency {latency:.2f}ms exceeds 100ms requirement"

# Bad: No context on failure
assert latency < 100.0
```

### Test Documentation

```python
def test_safety_constraint_temperature_bounds():
    """Test that temperature constraints properly validate operating bounds.
    
    The chemical reactor must maintain temperature between 280K and 320K
    for safe operation. This test validates the constraint implementation
    correctly identifies safe and unsafe temperature values.
    """
```

## Safety Testing Guidelines

Safety is paramount in industrial applications. Our safety testing includes:

### Constraint Validation
- Test all safety constraints under normal conditions
- Test constraint behavior at boundary values
- Test constraint response to violation scenarios
- Validate emergency shutdown procedures

### Failure Mode Testing
- Simulate sensor failures
- Test communication timeouts
- Validate graceful degradation
- Test recovery procedures

### Compliance Testing
- Validate against industrial standards
- Test audit trail generation
- Verify regulatory requirement compliance
- Test documentation completeness

For more information, see the [Safety Testing Specification](safety-testing.md).