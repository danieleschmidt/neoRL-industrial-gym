# GitHub Actions CI/CD Setup Guide

This document provides comprehensive CI/CD workflow configurations for neoRL-industrial-gym. Since this is an industrial RL library with safety-critical applications, our workflows emphasize security, safety validation, and comprehensive testing.

## Required GitHub Actions Workflows

### 1. Continuous Integration (`ci.yml`)

**Location**: `.github/workflows/ci.yml`

**Triggers**: 
- Push to main/develop branches
- Pull requests to main/develop branches
- Schedule: Daily at 2 AM UTC

**Key Components**:
```yaml
# Multi-platform testing matrix
strategy:
  matrix:
    os: [ubuntu-latest, macos-latest, windows-latest]
    python-version: ['3.8', '3.9', '3.10', '3.11']
    jax-version: ['0.4.1', 'latest']

# Essential steps:
- Checkout code with full history
- Setup Python with caching
- Install JAX (CPU/GPU variants)
- Install dependencies with pip-tools
- Run pre-commit hooks
- Execute test suite with coverage
- Run safety validation scripts
- Upload coverage to Codecov
- Cache dependencies and models
```

**Safety Validations**:
- Industrial constraint validation
- Emergency shutdown mechanism tests  
- Safety metric verification
- Compliance check execution

### 2. Security Scanning (`security.yml`)

**Location**: `.github/workflows/security.yml`

**Triggers**:
- Push to main branch
- Pull requests
- Schedule: Weekly on Sundays

**Security Checks**:
```yaml
jobs:
  dependency-scan:
    # Dependabot integration
    # Safety package scanning
    # pip-audit vulnerability checks
    
  code-scanning:
    # CodeQL analysis for Python
    # Semgrep security rules
    # Bandit security linting
    
  container-scanning:
    # Trivy container vulnerability scan
    # Docker image security analysis
    
  secrets-scan:
    # GitLeaks secret detection
    # TruffleHog credential scanning
```

### 3. Performance Benchmarking (`benchmarks.yml`)

**Location**: `.github/workflows/benchmarks.yml`

**Triggers**:
- Push to main branch
- Schedule: Weekly
- Manual dispatch

**Benchmark Suite**:
```yaml
jobs:
  performance-benchmarks:
    # JAX compilation benchmarks
    # Training speed benchmarks
    # Memory usage profiling
    # Environment simulation performance
    
  regression-detection:
    # Compare against baseline metrics
    # Alert on significant performance degradation
    # Upload results to GitHub Pages
```

### 4. Release Automation (`release.yml`)

**Location**: `.github/workflows/release.yml`

**Triggers**:
- Tag creation (v*.*.*)
- Manual dispatch

**Release Process**:
```yaml
jobs:
  release:
    # Validate version consistency
    # Build distribution packages
    # Run comprehensive test suite
    # Safety validation on all environments
    # Create GitHub release
    # Publish to PyPI
    # Update documentation
    # Generate changelog
```

### 5. Documentation Building (`docs.yml`)

**Location**: `.github/workflows/docs.yml`

**Triggers**:
- Push to main branch
- Pull requests affecting docs/
- Manual dispatch

**Documentation Pipeline**:
```yaml
jobs:
  build-docs:
    # Sphinx documentation build
    # API documentation generation
    # Tutorial validation
    # Deploy to GitHub Pages or ReadTheDocs
```

## Advanced Workflow Features

### Safety-Critical Validations

All workflows must include safety validation steps:

```yaml
- name: Safety Constraint Validation
  run: |
    python scripts/validate_safety.py --env all --strict
    python scripts/check_emergency_shutdown.py
    python scripts/validate_industrial_compliance.py

- name: Industrial Standards Compliance
  run: |
    # IEC 61508 functional safety checks
    # ISO 13849 machinery safety validation
    # ANSI/ISA-84.00.01 SIS compliance
```

### Performance Monitoring

```yaml
- name: Performance Regression Detection  
  run: |
    python scripts/benchmark_suite.py --compare-baseline
    python scripts/memory_profiling.py --environments all
    python scripts/jax_compilation_benchmark.py
```

### Multi-Environment Testing

```yaml
strategy:
  matrix:
    environment: 
      - ChemicalReactor-v0
      - RobotAssembly-v0  
      - HVACControl-v0
      - WaterTreatment-v0
      - SteelAnnealing-v0
      - PowerGrid-v0
      - SupplyChain-v0
```

## Environment Variables and Secrets

### Required Secrets

```yaml
# PyPI publishing
PYPI_API_TOKEN: ${{ secrets.PYPI_API_TOKEN }}

# Security scanning
CODECOV_TOKEN: ${{ secrets.CODECOV_TOKEN }}
SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}

# Performance monitoring  
WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}

# Documentation
READTHEDOCS_TOKEN: ${{ secrets.READTHEDOCS_TOKEN }}
```

### Environment Variables

```yaml
# JAX configuration
JAX_PLATFORM_NAME: cpu  # or gpu for GPU runners
JAX_ENABLE_X64: true

# Testing configuration
PYTEST_ADDOPTS: "--cov=neorl_industrial --cov-report=xml"
PYTHONPATH: "${GITHUB_WORKSPACE}/src:${PYTHONPATH}"

# Industrial safety
INDUSTRIAL_SAFETY_STRICT: true
EMERGENCY_SHUTDOWN_TEST: true
```

## Workflow Dependencies and Caching

### Dependency Caching Strategy

```yaml
- name: Cache Python Dependencies
  uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt') }}
    restore-keys: |
      ${{ runner.os }}-pip-

- name: Cache JAX Compilation
  uses: actions/cache@v3  
  with:
    path: ~/.cache/jax
    key: ${{ runner.os }}-jax-${{ hashFiles('src/**/*.py') }}

- name: Cache Pre-trained Models
  uses: actions/cache@v3
  with:
    path: models/pretrained/
    key: models-${{ hashFiles('models/pretrained/checksums.txt') }}
```

## Integration Requirements

### Required GitHub Apps/Integrations

1. **Dependabot**: Automated dependency updates
2. **CodeQL**: Security vulnerability scanning  
3. **Codecov**: Code coverage reporting
4. **ReadTheDocs**: Documentation hosting
5. **Mergify**: PR automation and safety checks

### Branch Protection Rules

```yaml
# main branch protection
required_status_checks:
  - ci/ubuntu-latest (3.8)
  - ci/ubuntu-latest (3.9) 
  - ci/ubuntu-latest (3.10)
  - security/dependency-scan
  - security/code-scanning
  - safety/constraint-validation
  - safety/emergency-shutdown-test
  
required_reviews: 2
dismiss_stale_reviews: true
require_code_owner_reviews: true
```

### Safety Gates

All workflows must pass these safety gates before deployment:

1. **Constraint Validation**: All safety constraints properly enforced
2. **Emergency Shutdown**: Emergency mechanisms function correctly  
3. **Industrial Compliance**: Standards compliance verified
4. **Security Scan**: No high/critical vulnerabilities
5. **Performance Regression**: No significant performance degradation

## Monitoring and Alerts

### Slack/Teams Integration

```yaml
- name: Notify on Failure
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: failure
    channel: '#neorl-ci'
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Metrics Collection

```yaml
- name: Collect Workflow Metrics
  run: |
    # Workflow execution time
    # Test execution time by category
    # Security scan duration
    # Safety validation time
    # Resource usage metrics
```

## Rollback and Recovery

### Automatic Rollback Triggers

- Critical security vulnerability detected
- Safety validation failures
- Performance regression > 20%
- Test coverage drops below 90%

### Manual Override Process

Emergency deployment bypass requires:
1. Two maintainer approvals
2. Security team approval
3. Safety engineer sign-off
4. Incident report filing

## Implementation Checklist

- [ ] Create `.github/workflows/` directory
- [ ] Implement `ci.yml` with full test matrix
- [ ] Setup `security.yml` with comprehensive scanning
- [ ] Configure `benchmarks.yml` for performance monitoring
- [ ] Create `release.yml` for automated releases
- [ ] Setup `docs.yml` for documentation building
- [ ] Configure required secrets in repository settings
- [ ] Setup branch protection rules
- [ ] Configure required GitHub Apps
- [ ] Test workflows with sample PRs
- [ ] Validate safety gates functionality
- [ ] Setup monitoring and alerting
- [ ] Document emergency procedures

## Safety Notice

⚠️ **Critical**: All workflow changes must be reviewed by safety engineers before deployment to ensure industrial safety standards are maintained.

## Support

For questions about CI/CD setup:
- Technical: dev@terragon.ai  
- Safety: safety@terragon.ai
- Security: security@terragon.ai