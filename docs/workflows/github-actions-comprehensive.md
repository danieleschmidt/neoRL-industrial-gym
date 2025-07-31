# Comprehensive GitHub Actions Setup

This document provides templates and guidance for setting up production-ready GitHub Actions workflows for the neoRL-industrial-gym project.

## Workflow Architecture

```
.github/workflows/
├── ci.yml                 # Main CI pipeline
├── security.yml           # Security scanning and analysis
├── release.yml            # Automated releases
├── container-security.yml # Container security scanning
├── dependency-update.yml  # Automated dependency updates
├── docs.yml              # Documentation building and deployment
└── performance.yml       # Performance benchmarking
```

## 1. Main CI Pipeline (`ci.yml`)

```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

concurrency:
  group: ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  test-matrix:
    name: Test (Python ${{ matrix.python-version }})
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
        
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}
        cache: 'pip'
    
    - name: Install JAX dependencies
      run: |
        pip install --upgrade pip
        pip install "jax[cpu]" jaxlib
    
    - name: Install package with test dependencies
      run: |
        pip install -e ".[test,dev]"
    
    - name: Run ruff linting
      run: |
        ruff check src/ tests/
        ruff format --check src/ tests/
    
    - name: Run traditional linting (compatibility)
      run: |
        flake8 src/ tests/
        black --check src/ tests/
        isort --check-only src/ tests/
    
    - name: Type checking with mypy
      run: mypy src/
    
    - name: Run tests with coverage
      run: |
        pytest tests/ --cov=neorl_industrial --cov-report=xml --cov-report=term-missing
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v4
      with:
        file: ./coverage.xml
        fail_ci_if_error: true
        token: ${{ secrets.CODECOV_TOKEN }}
    
    - name: Run safety validation
      run: python scripts/validate_safety.py --env all

  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: test-matrix
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python 3.9
      uses: actions/setup-python@v5
      with:
        python-version: '3.9'
        cache: 'pip'
    
    - name: Install package
      run: |
        pip install --upgrade pip
        pip install -e ".[test]"
    
    - name: Run integration tests
      run: pytest tests/integration/ --slow -v
    
    - name: Run benchmark suite
      run: python scripts/benchmark_suite.py --quick

  docker-build:
    name: Docker Build Test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Build Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: false
        tags: neorl-industrial:test
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

## 2. Security Scanning (`security.yml`)

```yaml
name: Security Scanning

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM

jobs:
  dependency-scan:
    name: Dependency Security Scan
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install --upgrade pip
        pip install safety bandit semgrep
        pip install -e .
    
    - name: Run Safety check
      run: safety check --json --output safety-report.json
      continue-on-error: true
    
    - name: Run Bandit security scan
      run: bandit -r src/ -f json -o bandit-report.json
      continue-on-error: true
    
    - name: Run Semgrep scan
      uses: semgrep/semgrep-action@v1
      with:
        config: auto
        publishToken: ${{ secrets.SEMGREP_APP_TOKEN }}
    
    - name: Upload security reports
      uses: actions/upload-artifact@v4
      with:
        name: security-reports
        path: |
          safety-report.json
          bandit-report.json

  sbom-generation:
    name: Generate SBOM
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.9'
    
    - name: Install package
      run: |
        pip install --upgrade pip
        pip install -e .
        pip install cyclonedx-bom
    
    - name: Generate SBOM
      run: |
        cyclonedx-py -o sbom.json
        cyclonedx-py -o sbom.xml --format xml
    
    - name: Upload SBOM artifacts
      uses: actions/upload-artifact@v4
      with:
        name: sbom
        path: |
          sbom.json
          sbom.xml

  license-compliance:
    name: License Compliance Check
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.9'
    
    - name: Install pip-licenses
      run: pip install pip-licenses
    
    - name: Install package
      run: pip install -e .
    
    - name: Check licenses
      run: |
        pip-licenses --format=json --output-file=licenses.json
        pip-licenses --format=csv --output-file=licenses.csv
        # Fail if GPL or other copyleft licenses are found
        pip-licenses --fail-on="GPL.*,AGPL.*,LGPL.*"
    
    - name: Upload license report
      uses: actions/upload-artifact@v4
      with:
        name: license-report
        path: |
          licenses.json
          licenses.csv
```

## 3. Container Security (`container-security.yml`)

```yaml
name: Container Security

on:
  push:
    branches: [ main ]
    paths: 
      - 'Dockerfile'
      - 'docker-compose.yml'
      - 'requirements*.txt'
      - 'pyproject.toml'
  pull_request:
    paths:
      - 'Dockerfile'
      - 'docker-compose.yml'

jobs:
  container-scan:
    name: Container Security Scan
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Build image for scanning
      uses: docker/build-push-action@v5
      with:
        context: .
        load: true
        tags: neorl-industrial:scan
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        image-ref: 'neorl-industrial:scan'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: 'trivy-results.sarif'
    
    - name: Run Grype vulnerability scanner
      uses: anchore/scan-action@v3
      with:
        image: "neorl-industrial:scan"
        fail-build: false
        acs-report-enable: true
    
    - name: Docker Bench Security
      run: |
        docker run --rm --net host --pid host --userns host --cap-add audit_control \
          -e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
          -v /etc:/etc:ro \
          -v /usr/bin/containerd:/usr/bin/containerd:ro \
          -v /usr/bin/runc:/usr/bin/runc:ro \
          -v /usr/lib/systemd:/usr/lib/systemd:ro \
          -v /var/lib:/var/lib:ro \
          -v /var/run/docker.sock:/var/run/docker.sock:ro \
          --label docker_bench_security \
          docker/docker-bench-security

  dockerfile-lint:
    name: Dockerfile Linting
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Lint Dockerfile with Hadolint
      uses: hadolint/hadolint-action@v3.1.0
      with:
        dockerfile: Dockerfile
        format: sarif
        output-file: hadolint-results.sarif
        no-fail: true
    
    - name: Upload Hadolint results
      uses: github/codeql-action/upload-sarif@v3
      with:
        sarif_file: hadolint-results.sarif
```

## 4. Automated Release (`release.yml`)

```yaml
name: Automated Release

on:
  push:
    tags:
      - 'v*.*.*'

jobs:
  build-and-publish:
    name: Build and Publish
    runs-on: ubuntu-latest
    environment: release
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.9'
    
    - name: Install build dependencies
      run: |
        pip install --upgrade pip
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Check package
      run: twine check dist/*
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
    
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: dist/*
        generate_release_notes: true
        draft: false
        prerelease: ${{ contains(github.ref_name, 'alpha') || contains(github.ref_name, 'beta') || contains(github.ref_name, 'rc') }}

  docker-release:
    name: Docker Release
    runs-on: ubuntu-latest
    needs: build-and-publish
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: terragonlabs/neorl-industrial-gym
        tags: |
          type=ref,event=tag
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
    
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

## 5. Performance Monitoring (`performance.yml`)

```yaml
name: Performance Monitoring

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 4 * * *'  # Daily at 4 AM

jobs:
  benchmark:
    name: Performance Benchmarks
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.9'
    
    - name: Install package with dependencies
      run: |
        pip install --upgrade pip
        pip install -e ".[test]"
        pip install pytest-benchmark
    
    - name: Run benchmarks
      run: |
        python scripts/benchmark_suite.py --output=benchmark-results.json
        pytest tests/benchmarks/ --benchmark-json=pytest-benchmark.json
    
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'customSmallerIsBetter'
        output-file-path: benchmark-results.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
        comment-on-alert: true
        alert-threshold: '120%'
        fail-on-alert: true

  memory-profiling:
    name: Memory Profiling
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.9'
    
    - name: Install profiling tools
      run: |
        pip install --upgrade pip
        pip install -e .
        pip install memory-profiler psutil
    
    - name: Run memory profiling
      run: |
        python -m memory_profiler scripts/memory_profile.py > memory-report.txt
    
    - name: Upload memory report
      uses: actions/upload-artifact@v4
      with:
        name: memory-profile
        path: memory-report.txt
```

## Setup Instructions

### 1. Required Secrets

Add these secrets to your GitHub repository settings:

```
CODECOV_TOKEN        # From codecov.io
PYPI_API_TOKEN       # From PyPI account settings
DOCKER_USERNAME      # Docker Hub username
DOCKER_PASSWORD      # Docker Hub password or access token
SEMGREP_APP_TOKEN    # From semgrep.dev (optional)
```

### 2. Required Files

Create these additional files in your repository:

- `scripts/memory_profile.py` - Memory profiling script
- `tests/benchmarks/` - Directory for benchmark tests
- `tests/integration/` - Directory for integration tests

### 3. Branch Protection Rules

Configure these branch protection rules for `main`:

- Require status checks to pass before merging
- Require branches to be up to date before merging
- Required status checks:
  - `Test (Python 3.8)`
  - `Test (Python 3.9)`
  - `Test (Python 3.10)`
  - `Test (Python 3.11)`
  - `Integration Tests`
  - `Docker Build Test`
  - `Dependency Security Scan`
  - `Container Security Scan`

### 4. Environment Configuration

Create a `release` environment in GitHub with:
- Required reviewers for production releases
- Deployment protection rules
- Environment secrets for PyPI and Docker Hub

## Customization Notes

1. **JAX GPU Support**: Uncomment GPU installation lines if using GPU runners
2. **MLflow Integration**: Add MLflow tracking to performance benchmarks
3. **Industrial Safety**: Customize safety validation for your specific use case
4. **Compliance**: Add additional compliance checks as needed
5. **Notifications**: Configure Slack/Teams notifications for critical failures

## Migration from Current Setup

1. Create `.github/workflows/` directory
2. Add workflow files one by one, starting with `ci.yml`
3. Test each workflow with a feature branch
4. Update existing scripts to work with CI environment
5. Configure required secrets and environment variables
6. Enable branch protection rules
7. Monitor and tune performance/resource usage

This comprehensive setup provides production-ready CI/CD with security, performance monitoring, and automated releases while maintaining compatibility with your existing tooling.