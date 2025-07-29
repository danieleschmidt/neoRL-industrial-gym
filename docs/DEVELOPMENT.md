# Development Guide

This guide covers the development setup and workflow for neoRL-industrial-gym.

## Prerequisites

- Python 3.8 or higher
- Git
- Docker (optional, for containerized development)

## Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/terragon-labs/neoRL-industrial-gym.git
   cd neoRL-industrial-gym
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   make install-dev
   # or manually: pip install -r requirements-dev.txt && pre-commit install
   ```

4. **Verify installation**:
   ```bash
   make test
   ```

## Development Workflow

### Code Quality Tools

We use several tools to maintain code quality:

- **Black**: Code formatting
- **Isort**: Import sorting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pre-commit**: Git hooks for automated checks

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run specific test file
pytest tests/test_specific.py -v
```

### Code Formatting

```bash
# Format code
make format

# Check formatting without making changes
black --check src/ tests/
```

### Type Checking

```bash
make type-check
```

### Safety Validation

Before any industrial deployment consideration:

```bash
make validate-safety
```

## Docker Development

For containerized development:

```bash
# Build and run development container
docker-compose up neorl-dev

# Run tests in container
docker-compose run neorl-dev pytest tests/

# Start Jupyter Lab
docker-compose up jupyter
# Access at http://localhost:8888

# Start MLflow server
docker-compose up mlflow
# Access at http://localhost:5000
```

## Project Structure

```
neoRL-industrial-gym/
├── src/neorl_industrial/     # Main package
├── tests/                    # Test suite
├── docs/                     # Documentation
├── scripts/                  # Utility scripts
├── examples/                 # Example usage
├── requirements.txt          # Core dependencies
├── requirements-dev.txt      # Development dependencies
├── pyproject.toml           # Project configuration
└── Makefile                 # Development tasks
```

## Adding New Features

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Implement your changes with tests
3. Run quality checks: `make check-all`
4. Update documentation if needed
5. Submit a pull request

## Industrial Safety Guidelines

When developing features for industrial environments:

1. **Always include safety constraints**
2. **Implement emergency shutdown mechanisms**
3. **Provide human override capabilities**
4. **Include comprehensive logging**
5. **Validate against real-world safety standards**

## Performance Considerations

- Use JAX for numerical computations
- Profile code for bottlenecks
- Consider memory usage for large datasets
- Test with realistic industrial data sizes

## Debugging Tips

1. **Use JAX debugging tools**: `jax.debug.print()`
2. **Enable verbose logging**: Set `JAX_LOG_LEVEL=DEBUG`
3. **Profile with**: `python -m cProfile your_script.py`
4. **Memory profiling**: Use `memory_profiler` package

## Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`
5. GitHub Actions will handle the rest

## Getting Help

- Check existing [issues](https://github.com/terragon-labs/neoRL-industrial-gym/issues)
- Read the [documentation](https://neorl-industrial.readthedocs.io)
- Contact maintainers: dev@terragon.ai