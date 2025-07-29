# CI/CD Setup Guide

This document provides templates and guidelines for setting up Continuous Integration and Continuous Deployment for neoRL-industrial-gym.

## Required GitHub Actions Workflows

### 1. Main CI Workflow (.github/workflows/ci.yml)

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
    
    - name: Lint with flake8
      run: flake8 src/ tests/
    
    - name: Type check with mypy
      run: mypy src/
    
    - name: Test with pytest
      run: pytest tests/ --cov=neorl_industrial --cov-report=xml
    
    - name: Safety validation
      run: python scripts/validate_safety.py --env all
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### 2. Security Scanning (.github/workflows/security.yml)

```yaml
name: Security

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * 1'  # Weekly

jobs:
  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Snyk to check for vulnerabilities
      uses: snyk/actions/python@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      with:
        args: --severity-threshold=high
```

### 3. Release Workflow (.github/workflows/release.yml)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.8'
    
    - name: Install build dependencies
      run: |
        python -m pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
      run: twine upload dist/*
```

## Required Secrets

Configure these secrets in your GitHub repository settings:

- `SNYK_TOKEN`: For security vulnerability scanning
- `PYPI_TOKEN`: For automated package publishing
- `CODECOV_TOKEN`: For code coverage reporting

## Branch Protection Rules

Configure the following branch protection rules for `main`:

- Require pull request reviews before merging
- Require status checks to pass before merging:
  - `test (3.8)`
  - `test (3.9)`
  - `test (3.10)`
  - `security`
- Require branches to be up to date before merging
- Restrict pushes to matching branches

## Integration Requirements

- **Code Coverage**: Maintain minimum 90% test coverage
- **Security**: All high-severity vulnerabilities must be resolved
- **Safety**: Industrial safety validation must pass
- **Documentation**: All public APIs must be documented

## Manual Setup Steps

1. Enable GitHub Actions in repository settings
2. Add required secrets to repository
3. Configure branch protection rules
4. Set up external integrations (Codecov, Snyk)
5. Configure deployment environments if needed