.PHONY: install install-dev test lint format type-check clean docs

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=neorl_industrial --cov-report=html --cov-report=term-missing

# Code quality
lint:
	flake8 src/ tests/

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/

# Development
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Documentation
docs:
	cd docs && make html

# All quality checks
check-all: lint type-check test

# Build package
build:
	python -m build

# Safety validation
validate-safety:
	python scripts/validate_safety.py --env all