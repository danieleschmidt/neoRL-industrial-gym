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
	ruff check src/ tests/
	flake8 src/ tests/  # Keeping flake8 for compatibility

lint-fix:
	ruff check --fix src/ tests/

format:
	ruff format src/ tests/
	black src/ tests/  # Keeping black for compatibility
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

# Modern quality checks with ruff
check-modern: lint-fix format type-check test

# Build package
build:
	python -m build

# Safety validation
validate-safety:
	python scripts/validate_safety.py --env all

# Security scanning
security-scan:
	python scripts/security_scan.py --output security-report.json

security-scan-strict:
	python scripts/security_scan.py --output security-report.json --fail-on-vuln

# Container security
container-security:
	python scripts/container_security.py --output container-security-report.json

container-security-quick:
	python scripts/container_security.py --skip-build --output container-security-report.json

# Configuration validation
validate-config:
	python scripts/validate_config.py

# Comprehensive validation
validate-all: validate-config validate-safety security-scan