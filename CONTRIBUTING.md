# Contributing to neoRL-industrial-gym

We welcome contributions to neoRL-industrial-gym! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/neoRL-industrial-gym.git`
3. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
4. Install development dependencies: `pip install -r requirements-dev.txt`
5. Install pre-commit hooks: `pre-commit install`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests: `pytest tests/`
4. Run code formatting: `black src/ tests/` and `isort src/ tests/`
5. Run type checking: `mypy src/`
6. Commit your changes with a descriptive message
7. Push to your fork and create a pull request

## Code Standards

- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write comprehensive docstrings in Google style
- Maintain test coverage above 90%
- Include safety validations for industrial environments

## Testing

- Write unit tests for all new functionality
- Include integration tests for complex features
- Test safety constraints thoroughly
- Validate against real industrial data when possible

## Pull Request Process

1. Ensure all tests pass and code meets quality standards
2. Update documentation for any new features
3. Include performance benchmarks for new algorithms
4. Add safety analysis for industrial applications
5. Request review from maintainers

## Questions?

Open an issue or contact the maintainers at dev@terragon.ai.

Thank you for contributing!