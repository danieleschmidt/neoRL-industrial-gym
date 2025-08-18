# GitHub Workflows Troubleshooting Guide

This guide helps diagnose and resolve common issues with GitHub Actions workflows in neoRL-industrial-gym.

## Quick Diagnosis Checklist

When workflows fail, check these items in order:

- [ ] **Workflow file syntax**: YAML syntax errors
- [ ] **Secrets availability**: Required secrets are configured
- [ ] **Permissions**: Repository and workflow permissions
- [ ] **Dependencies**: Python packages and system requirements
- [ ] **Environment**: Industrial-specific configuration issues
- [ ] **Resource limits**: GitHub Actions resource constraints

## Common Issues and Solutions

### 1. Workflow Not Triggering

#### Symptoms
- Workflow doesn't run when expected
- No workflow runs appear in Actions tab
- Status checks not appearing on PRs

#### Diagnosis Steps
```bash
# Check workflow file location
ls -la .github/workflows/

# Validate YAML syntax
yamllint .github/workflows/ci.yml

# Check branch protection settings
gh api repos/{owner}/{repo}/branches/main/protection
```

#### Common Causes and Solutions

| Cause | Solution |
|-------|----------|
| Workflow file not in `.github/workflows/` | Move files to correct directory |
| YAML syntax errors | Use `yamllint` or online YAML validator |
| Branch name mismatch in triggers | Update `on.push.branches` in workflow |
| File permissions too restrictive | Set files to 644: `chmod 644 .github/workflows/*.yml` |

### 2. Authentication and Permissions Errors

#### Symptoms
- "Permission denied" errors
- Cannot push to registry
- Cannot create releases
- Secrets not accessible

#### Industrial-Specific Permissions
```yaml
# Required permissions for industrial workflows
permissions:
  contents: write          # For creating releases
  packages: write          # For container registry
  security-events: write   # For SARIF uploads
  id-token: write         # For OIDC attestations
  actions: read           # For downloading artifacts
```

#### Solutions
```bash
# Check repository permissions
gh api repos/{owner}/{repo} --jq '.permissions'

# Verify secrets exist
gh secret list

# Check GITHUB_TOKEN permissions
echo "Token permissions: ${{ github.token }}" # In workflow
```

### 3. Safety and Compliance Validation Failures

#### Symptoms
- IEC 62443 validation fails
- Safety constraint tests fail
- Industrial compliance checks error

#### Debugging Safety Issues
```yaml
# Add debug output to safety validation
- name: Debug Safety Validation
  if: failure()
  run: |
    echo "🔍 Debugging safety validation failure..."
    python -c "
    import neorl_industrial.safety as safety
    print('Safety modules loaded:', dir(safety))
    print('Available constraints:', safety.list_constraints())
    "
    
    # Run with verbose output
    python -m pytest tests/safety/ -v -s --tb=long
```

#### Common Safety Validation Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `SafetyConstraintError` | Missing constraint definition | Implement all required constraints in test environments |
| `EmergencyShutdownTimeout` | Shutdown system not responding | Check emergency shutdown mock implementation |
| `ComplianceStandardNotFound` | Missing compliance validation | Ensure all standards (IEC 62443, NIST) are implemented |

### 4. Container Build and Security Scan Failures

#### Symptoms
- Docker build fails
- Security scans timeout or error
- SBOM generation fails

#### Container Debugging
```yaml
# Debug container build
- name: Debug Docker Build
  run: |
    echo "🐳 Docker build context:"
    ls -la
    
    echo "🔍 Dockerfile validation:"
    docker run --rm -i hadolint/hadolint < Dockerfile
    
    echo "📦 Build args:"
    echo "VERSION=$VERSION"
    echo "BUILD_DATE=$BUILD_DATE"
    echo "VCS_REF=$VCS_REF"
```

#### Security Scan Debugging
```yaml
# Debug security scans
- name: Debug Security Scans
  if: failure()
  run: |
    echo "🔍 Available security tools:"
    which syft || echo "syft not found"
    which grype || echo "grype not found"
    which cosign || echo "cosign not found"
    
    echo "📋 Image information:"
    docker images
    
    echo "🔬 Manual SBOM generation:"
    syft --help || echo "SBOM generation unavailable"
```

### 5. Industrial Environment Testing Issues

#### Symptoms
- Environment-specific tests fail
- Performance benchmarks timeout
- Industrial data validation errors

#### Environment Testing Debug
```yaml
# Debug industrial environments
- name: Debug Industrial Environments
  if: failure()
  run: |
    echo "🏭 Available environments:"
    python -c "
    import neorl_industrial
    envs = neorl_industrial.list_environments()
    print('Environments:', envs)
    for env in envs:
        try:
            e = neorl_industrial.make(env)
            print(f'{env}: OK')
        except Exception as ex:
            print(f'{env}: ERROR - {ex}')
    "
    
    echo "📊 JAX/Hardware configuration:"
    python -c "
    import jax
    print('JAX version:', jax.__version__)
    print('JAX devices:', jax.devices())
    print('JAX backend:', jax.default_backend())
    "
```

### 6. Resource and Performance Issues

#### Symptoms
- Workflows timeout
- Out of memory errors
- Disk space issues
- Rate limiting

#### Resource Monitoring
```yaml
# Monitor resource usage
- name: Monitor Resources
  run: |
    echo "💾 Memory usage:"
    free -h
    
    echo "💽 Disk usage:"
    df -h
    
    echo "⏱️ Process information:"
    ps aux --sort=-%cpu | head -10
    
    echo "🔄 GitHub API rate limits:"
    gh api rate_limit
```

#### Solutions for Resource Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Workflow timeout | Long-running tests | Split into parallel jobs, optimize test selection |
| Out of memory | Large datasets/models | Use smaller test datasets, increase VM size |
| Disk space full | Large artifacts | Clean up artifacts, use external storage |
| API rate limiting | Too many API calls | Implement backoff, cache results |

### 7. Dependency and Package Issues

#### Symptoms
- Package installation fails
- Version conflicts
- Missing industrial dependencies

#### Dependency Debugging
```yaml
# Debug Python dependencies
- name: Debug Dependencies
  if: failure()
  run: |
    echo "🐍 Python environment:"
    python --version
    pip --version
    
    echo "📦 Installed packages:"
    pip list
    
    echo "🔍 Dependency conflicts:"
    pip check
    
    echo "🏭 Industrial-specific packages:"
    python -c "
    packages = ['jax', 'jaxlib', 'mlflow', 'optree']
    for pkg in packages:
        try:
            __import__(pkg)
            print(f'{pkg}: OK')
        except ImportError as e:
            print(f'{pkg}: ERROR - {e}')
    "
```

### 8. Artifact and Release Issues

#### Symptoms
- Artifact upload fails
- Release creation fails
- Asset upload errors

#### Release Debugging
```yaml
# Debug release process
- name: Debug Release Process
  if: failure()
  run: |
    echo "📦 Artifacts created:"
    find . -name "*.whl" -o -name "*.tar.gz" -o -name "*.json" | head -20
    
    echo "🏷️ Tag information:"
    git describe --tags --always
    git tag -l | tail -10
    
    echo "📋 Release notes:"
    ls -la *release* || echo "No release files found"
    
    echo "🔐 GitHub token permissions:"
    gh auth status
```

## Advanced Troubleshooting

### Enable Debug Logging

Add to any job for detailed debugging:

```yaml
- name: Enable Debug Logging
  run: |
    echo "ACTIONS_STEP_DEBUG=true" >> $GITHUB_ENV
    echo "ACTIONS_RUNNER_DEBUG=true" >> $GITHUB_ENV
```

### Industrial-Specific Debug Information

```yaml
- name: Industrial System Debug
  run: |
    echo "🏭 Industrial configuration:"
    python -c "
    import os
    print('INDUSTRIAL_MODE:', os.getenv('INDUSTRIAL_MODE', 'Not set'))
    print('SAFETY_VALIDATION_LEVEL:', os.getenv('SAFETY_VALIDATION_LEVEL', 'Not set'))
    print('COMPLIANCE_STANDARDS:', os.getenv('COMPLIANCE_STANDARDS', 'Not set'))
    
    # Check industrial modules
    try:
        from neorl_industrial.safety import SafetyMonitor
        print('Safety monitoring: Available')
    except ImportError as e:
        print('Safety monitoring: ERROR -', e)
    
    try:
        from neorl_industrial.compliance import ComplianceValidator
        print('Compliance validation: Available')
    except ImportError as e:
        print('Compliance validation: ERROR -', e)
    "
```

### Network and Connectivity Issues

```yaml
- name: Debug Network Connectivity
  run: |
    echo "🌐 Network connectivity:"
    ping -c 3 github.com || echo "GitHub unreachable"
    ping -c 3 pypi.org || echo "PyPI unreachable"
    
    echo "🐳 Container registry connectivity:"
    docker pull hello-world || echo "Docker Hub unreachable"
    
    echo "📡 DNS resolution:"
    nslookup github.com
    nslookup ghcr.io
```

## Workflow Optimization Tips

### 1. Parallel Job Execution
```yaml
# Split testing across multiple jobs
strategy:
  matrix:
    python-version: ['3.8', '3.9', '3.10', '3.11']
    test-group: ['unit', 'integration', 'safety', 'compliance']
```

### 2. Conditional Job Execution
```yaml
# Skip expensive jobs for documentation-only changes
- name: Skip tests for docs
  run: |
    if git diff --name-only HEAD~1 | grep -q "^docs/"; then
      echo "SKIP_TESTS=true" >> $GITHUB_ENV
    fi
```

### 3. Cache Optimization
```yaml
# Cache industrial datasets and models
- name: Cache Industrial Data
  uses: actions/cache@v4
  with:
    path: |
      ~/.cache/neorl-industrial
      data/industrial-datasets
    key: industrial-data-${{ hashFiles('data/datasets.json') }}
```

## Performance Monitoring

### Workflow Performance Metrics
```yaml
- name: Measure Workflow Performance
  run: |
    echo "⏱️ Workflow timing information:"
    echo "Start time: $(date)"
    echo "Job duration: ${{ steps.performance.outputs.duration }}"
    
    echo "💾 Resource usage:"
    echo "Peak memory: $(cat /proc/meminfo | grep MemTotal)"
    echo "CPU cores: $(nproc)"
```

## Getting Help

### 1. Internal Support
- **Engineering Team**: `#engineering-ci` Slack channel
- **Security Team**: `#security-team` channel for security-related issues
- **DevOps Team**: `#devops` channel for infrastructure issues

### 2. External Resources
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Build Troubleshooting](https://docs.docker.com/build/troubleshooting/)
- [Industrial Security Standards](https://www.isa.org/standards-and-publications/isa-standards/isa-standards-committees/isa99)

### 3. Creating Support Tickets

When creating a support ticket, include:

```bash
# Workflow run information
echo "Workflow: $GITHUB_WORKFLOW"
echo "Run ID: $GITHUB_RUN_ID"  
echo "Run Number: $GITHUB_RUN_NUMBER"
echo "Repository: $GITHUB_REPOSITORY"
echo "Event: $GITHUB_EVENT_NAME"
echo "Ref: $GITHUB_REF"
echo "SHA: $GITHUB_SHA"

# Error context
echo "Error occurred in step: [STEP_NAME]"
echo "Error message: [ERROR_MESSAGE]"
echo "Expected behavior: [EXPECTED_RESULT]"
echo "Actual behavior: [ACTUAL_RESULT]"

# Environment information
python --version
pip --version
docker --version
```

---

**Remember**: Industrial environments require extra attention to safety and compliance. Always validate thoroughly in staging before production deployment.