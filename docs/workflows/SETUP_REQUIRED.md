# Required GitHub Workflows Setup

This document outlines the manual setup required for GitHub Actions workflows since automated workflow creation requires elevated permissions.

## ⚠️ Important Notice

Due to GitHub App permission limitations, the workflows in `docs/workflows/examples/` must be manually created by repository maintainers. This document provides step-by-step instructions for setting up the complete CI/CD pipeline.

## Quick Setup Checklist

- [ ] Create `.github/workflows/` directory
- [ ] Copy workflow files from `docs/workflows/examples/`
- [ ] Configure repository secrets
- [ ] Set up branch protection rules
- [ ] Configure environments
- [ ] Test workflows with feature branch
- [ ] Enable security features

## Workflow Files to Create

Repository maintainers must create these workflow files in `.github/workflows/`:

### 1. Core CI/CD Workflows

| File | Source | Priority | Description |
|------|---------|----------|-------------|
| `ci.yml` | `docs/workflows/examples/ci.yml` | **CRITICAL** | Main CI pipeline with tests, linting, and validation |
| `security-scan.yml` | `docs/workflows/examples/security-scan.yml` | **HIGH** | Comprehensive security scanning |
| `release.yml` | `docs/workflows/examples/release.yml` | **HIGH** | Automated releases and publishing |
| `container-security.yml` | `docs/workflows/examples/container-security.yml` | **MEDIUM** | Container-specific security scans |
| `dependency-update.yml` | `docs/workflows/examples/dependency-update.yml` | **MEDIUM** | Automated dependency updates |
| `docs.yml` | `docs/workflows/examples/docs.yml` | **LOW** | Documentation building and deployment |
| `performance.yml` | `docs/workflows/examples/performance.yml` | **LOW** | Performance benchmarking |

### 2. Setup Commands

```bash
# 1. Create workflows directory
mkdir -p .github/workflows

# 2. Copy workflow files (run from repository root)
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/security-scan.yml .github/workflows/
cp docs/workflows/examples/release.yml .github/workflows/

# 3. Commit and push
git add .github/workflows/
git commit -m "feat: add GitHub Actions workflows for CI/CD and security"
git push origin main
```

## Required Repository Secrets

Configure these secrets in **Settings > Secrets and variables > Actions**:

### Essential Secrets (Required for basic CI/CD)

| Secret Name | Description | How to Obtain |
|-------------|-------------|---------------|
| `CODECOV_TOKEN` | Code coverage reporting | Sign up at [codecov.io](https://codecov.io), add repository |
| `PYPI_API_TOKEN` | PyPI package publishing | Create token at [PyPI Account Settings](https://pypi.org/manage/account/) |
| `DOCKER_USERNAME` | Docker Hub username | Your Docker Hub username |
| `DOCKER_PASSWORD` | Docker Hub token | Create token at [Docker Hub Settings](https://hub.docker.com/settings/security) |

### Security Scanning Secrets (Recommended)

| Secret Name | Description | How to Obtain |
|-------------|-------------|---------------|
| `SEMGREP_APP_TOKEN` | Advanced security scanning | Sign up at [semgrep.dev](https://semgrep.dev) |
| `SNYK_TOKEN` | Vulnerability scanning | Create account at [snyk.io](https://snyk.io) |

### Notification Secrets (Optional)

| Secret Name | Description | How to Obtain |
|-------------|-------------|---------------|
| `SLACK_WEBHOOK` | CI/CD notifications | Create webhook in Slack workspace |
| `SECURITY_SLACK_WEBHOOK` | Security alerts | Separate webhook for security team |

### Example Secret Configuration

```bash
# Using GitHub CLI
gh secret set CODECOV_TOKEN --body "your-codecov-token"
gh secret set PYPI_API_TOKEN --body "pypi-your-token"
gh secret set DOCKER_USERNAME --body "your-docker-username"
gh secret set DOCKER_PASSWORD --body "your-docker-token"
```

## Branch Protection Rules

Configure branch protection for `main` branch at **Settings > Branches**:

### Required Status Checks

Enable these status checks (they will appear after first workflow run):

- `Pre-commit Checks`
- `Test (Python 3.8, ubuntu-latest)`
- `Test (Python 3.9, ubuntu-latest)`
- `Test (Python 3.10, ubuntu-latest)`
- `Test (Python 3.11, ubuntu-latest)`
- `Integration Tests`
- `Docker Build and Test`
- `Safety Validation`
- `Dependency Security Scan`
- `Code Security Analysis`

### Protection Settings

```yaml
# Recommended branch protection settings
Require a pull request before merging: ✅
  Require approvals: ✅ (1 approval minimum)
  Dismiss stale reviews: ✅
  Require review from code owners: ✅

Require status checks to pass: ✅
  Require branches to be up to date: ✅
  Status checks: [all checks listed above]

Require conversation resolution: ✅
Require signed commits: ✅ (if using commit signing)
Require linear history: ✅
Include administrators: ✅

Allow force pushes: ❌
Allow deletions: ❌
```

## Environment Configuration

Create environments at **Settings > Environments**:

### 1. `staging` Environment

```yaml
Deployment protection rules:
  - Wait timer: 0 minutes
  - Required reviewers: [development team]

Environment secrets:
  - STAGING_API_URL
  - STAGING_DATABASE_URL
```

### 2. `production` Environment

```yaml
Deployment protection rules:
  - Wait timer: 5 minutes
  - Required reviewers: [senior developers, tech lead]

Environment secrets:
  - PRODUCTION_API_URL
  - PRODUCTION_DATABASE_URL
  - PRODUCTION_MONITORING_TOKEN
```

## Security Features Setup

### 1. Enable GitHub Security Features

Go to **Settings > Security & analysis** and enable:

- [ ] Dependency graph
- [ ] Dependabot alerts
- [ ] Dependabot security updates
- [ ] Dependabot version updates
- [ ] Code scanning alerts
- [ ] Secret scanning alerts

### 2. Create Dependabot Configuration

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "04:00"
    open-pull-requests-limit: 5
    reviewers:
      - "tech-lead"
    assignees:
      - "security-team"
    commit-message:
      prefix: "deps"
      include: "scope"

  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 3

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "monthly"
    open-pull-requests-limit: 2
```

## Testing the Setup

### 1. Create Test Branch

```bash
git checkout -b test-ci-setup
echo "# Test CI" >> test-ci.md
git add test-ci.md
git commit -m "test: verify CI/CD setup"
git push origin test-ci-setup
```

### 2. Create Pull Request

Create a PR to trigger all workflows and verify:

- [ ] All status checks appear and pass
- [ ] Security scans complete successfully
- [ ] Code coverage reports are generated
- [ ] Docker images build successfully

### 3. Verify Security Features

- [ ] Dependabot creates security update PRs
- [ ] Code scanning detects test vulnerabilities
- [ ] Secret scanning catches test secrets

## Common Issues and Solutions

### Issue: Workflows don't trigger

**Solution:**
- Ensure workflow files are in `.github/workflows/`
- Check YAML syntax with `yamllint`
- Verify branch names in trigger conditions

### Issue: Status checks not appearing in branch protection

**Solution:**
- Workflows must run at least once to appear in the list
- Wait for initial workflow completion
- Refresh branch protection settings page

### Issue: Docker build fails

**Solution:**
- Verify Dockerfile syntax
- Check base image availability
- Ensure all COPY paths exist

### Issue: Secret not available in workflow

**Solution:**
- Verify secret name matches exactly (case-sensitive)
- Check if secret is available in the correct environment
- Ensure proper workflow context for secret access

## Monitoring and Maintenance

### Weekly Tasks

- [ ] Review failed workflows and address issues
- [ ] Check security scan results
- [ ] Update dependencies via Dependabot PRs
- [ ] Monitor resource usage (Actions minutes)

### Monthly Tasks

- [ ] Review and update workflow configurations
- [ ] Audit repository secrets and remove unused ones
- [ ] Update branch protection rules if needed
- [ ] Check compliance with security policies

### Quarterly Tasks

- [ ] Review and update security scanning tools
- [ ] Audit access permissions and environments
- [ ] Update workflow documentation
- [ ] Performance review of CI/CD pipeline

## Support and Documentation

- **Internal Documentation**: See `docs/workflows/` for detailed workflow documentation
- **GitHub Actions Documentation**: [docs.github.com/actions](https://docs.github.com/en/actions)
- **Security Best Practices**: `docs/SECURITY_POLICIES.md`
- **Troubleshooting Guide**: `docs/workflows/troubleshooting.md`

## Contact Information

For questions about this setup:

- **CI/CD Issues**: #engineering-ci channel
- **Security Concerns**: #security-team channel  
- **General Questions**: #neorl-industrial channel

---

**Note**: This setup provides enterprise-grade CI/CD with comprehensive security scanning. All workflows are designed to be compatible with the existing project structure and safety requirements.