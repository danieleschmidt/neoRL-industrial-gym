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

# ================================
# BUILD AND CONTAINERIZATION
# ================================

# Container builds
docker-dev:
	./scripts/build.sh dev

docker-prod:
	./scripts/build.sh prod

docker-gpu:
	./scripts/build.sh gpu

docker-all:
	./scripts/build.sh all

docker-multi:
	./scripts/build.sh multi

# Container builds with SBOM and security scanning
docker-secure:
	./scripts/build.sh all --sbom --security-scan

# Development environment with docker-compose
docker-up:
	docker-compose up -d neorl-dev mlflow postgres

docker-up-full:
	docker-compose --profile monitoring --profile jupyter up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f neorl-dev

# Container management
docker-clean:
	docker system prune -f
	docker volume prune -f

docker-clean-all:
	docker system prune -af
	docker volume prune -f
	docker builder prune -af

# SBOM generation
sbom:
	./scripts/build.sh dev --sbom

# ================================
# RELEASE AUTOMATION
# ================================

# Version management
version-patch:
	./scripts/release.sh prepare --type patch

version-minor:
	./scripts/release.sh prepare --type minor

version-major:
	./scripts/release.sh prepare --type major

# Release builds
release-build:
	./scripts/release.sh build --type patch

release-build-minor:
	./scripts/release.sh build --type minor

release-build-major:
	./scripts/release.sh build --type major

# Full release process
release-patch:
	./scripts/release.sh full --type patch

release-minor:
	./scripts/release.sh full --type minor

release-major:
	./scripts/release.sh full --type major

# Pre-release
pre-release:
	./scripts/release.sh full --type patch --pre-release

# Dry run releases (safe testing)
release-dry-run:
	./scripts/release.sh full --type patch --dry-run

# ================================
# DEVELOPMENT WORKFLOWS
# ================================

# Complete development setup
dev-setup: install-dev docker-up
	@echo "Development environment ready!"
	@echo "MLflow UI: http://localhost:5000"
	@echo "Development container: docker-compose exec neorl-dev bash"

# Complete testing workflow
test-full: check-all test-cov validate-all
	@echo "All tests and validations completed!"

# Complete CI workflow (what CI should run)
ci: install-dev check-all test-cov validate-all docker-secure
	@echo "CI pipeline completed successfully!"

# Performance benchmarks
benchmark:
	python scripts/benchmark_suite.py --output benchmark-results.json

benchmark-full:
	python scripts/benchmark_suite.py --full --output benchmark-results-full.json

# ================================
# MAINTENANCE
# ================================

# Update dependencies
update-deps:
	pip-compile requirements.in
	pip-compile requirements-dev.in

# Security updates
security-update:
	pip-audit --fix

# Clean everything
clean-all: clean docker-clean
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf .pytest_cache/
	rm -rf logs/
	rm -rf *.log

# ================================
# HELP
# ================================

help:
	@echo "neoRL-industrial-gym Makefile"
	@echo ""
	@echo "Installation:"
	@echo "  install          Install package"
	@echo "  install-dev      Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test             Run tests"
	@echo "  test-cov         Run tests with coverage"
	@echo "  test-full        Run complete testing workflow"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             Run linting"
	@echo "  lint-fix         Run linting with auto-fix"
	@echo "  format           Format code"
	@echo "  type-check       Run type checking"
	@echo "  check-all        Run all quality checks"
	@echo ""
	@echo "Security:"
	@echo "  validate-safety  Run safety validation"
	@echo "  security-scan    Run security scan"
	@echo "  validate-all     Run all validations"
	@echo ""
	@echo "Containers:"
	@echo "  docker-dev       Build development image"
	@echo "  docker-prod      Build production image"
	@echo "  docker-gpu       Build GPU image"
	@echo "  docker-all       Build all images"
	@echo "  docker-secure    Build with security scanning"
	@echo "  docker-up        Start development environment"
	@echo "  docker-down      Stop environment"
	@echo ""
	@echo "Release:"
	@echo "  version-patch    Bump patch version"
	@echo "  release-patch    Release patch version"
	@echo "  release-dry-run  Test release process"
	@echo ""
	@echo "Development:"
	@echo "  dev-setup        Complete development setup"
	@echo "  ci               Run CI pipeline locally"
	@echo "  benchmark        Run performance benchmarks"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean            Clean build artifacts"
	@echo "  clean-all        Clean everything"