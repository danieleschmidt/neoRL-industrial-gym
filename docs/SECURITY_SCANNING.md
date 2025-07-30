# Security & Dependency Scanning Guide

This document outlines the comprehensive security and dependency scanning setup for neoRL-industrial-gym, designed for industrial applications with strict security requirements.

## Overview

As an industrial RL library dealing with safety-critical systems, we implement multi-layered security scanning:

- **Dependency Vulnerability Scanning**: Identify known vulnerabilities in dependencies
- **Code Security Analysis**: Static analysis for security issues and vulnerabilities  
- **Secret Detection**: Prevent accidental credential commits
- **Container Security**: Scan Docker images for vulnerabilities
- **Supply Chain Security**: Verify integrity of dependencies and build process

## Automated Security Scanning Tools

### 1. Dependency Vulnerability Scanning

#### Safety (Python Package Vulnerabilities)
```bash
# Install safety
pip install safety

# Basic vulnerability scan
safety check

# Scan requirements files
safety check -r requirements.txt
safety check -r requirements-dev.txt

# JSON output for CI integration
safety check --json --output safety-report.json

# Ignore specific vulnerabilities (use cautiously)
safety check --ignore 12345
```

#### pip-audit (Modern Alternative)
```bash
# Install pip-audit  
pip install pip-audit

# Scan installed packages
pip-audit

# Scan requirements files
pip-audit -r requirements.txt

# Fix vulnerable packages automatically
pip-audit --fix

# Generate reports
pip-audit --format=json --output=audit-report.json
```

#### OSV-Scanner (Google's Open Source Vulnerabilities)
```bash
# Install osv-scanner
go install github.com/google/osv-scanner/cmd/osv-scanner@v1

# Scan project directory
osv-scanner --lockfiles ./

# Scan specific files
osv-scanner -r requirements.txt

# Output formats
osv-scanner --format json --output osv-report.json ./
```

### 2. Static Code Security Analysis

#### Bandit (Python Security Linting)
```bash
# Install bandit
pip install bandit[toml]

# Scan source code
bandit -r src/

# Generate reports
bandit -r src/ -f json -o bandit-report.json

# Configuration file: pyproject.toml
[tool.bandit]
exclude_dirs = ["tests", "docs"]
skips = ["B101", "B601"]  # Skip specific tests if justified
```

#### Semgrep (Multi-language Security Analysis)
```bash
# Install semgrep
pip install semgrep

# Run default security rules
semgrep --config=auto src/

# Run Python-specific rules
semgrep --config=python src/

# Industry-specific rules
semgrep --config=security-audit src/

# Generate reports
semgrep --config=auto --json --output=semgrep-report.json src/
```

#### CodeQL (GitHub's Semantic Analysis)
- Integrated via GitHub Actions
- Semantic analysis of code for security vulnerabilities
- Custom queries for industrial RL specific patterns
- Automatic sarif report generation

### 3. Secret Detection

#### GitLeaks (Git History Secret Scanning)
```bash
# Install gitleaks
# Download from https://github.com/zricethezav/gitleaks/releases

# Scan entire repository history
gitleaks detect --source . --verbose

# Scan specific commits
gitleaks detect --log-opts="--since=2023-01-01"

# Generate reports
gitleaks detect --source . --report-format json --report-path gitleaks-report.json
```

#### TruffleHog (Comprehensive Secret Scanning)
```bash
# Install truffleHog
pip install truffleHog

# Scan git repository
truffleHog --regex --entropy=False .

# Scan specific files
truffleHog --regex requirements.txt

# JSON output
truffleHog --json --regex . > trufflehog-report.json
```

### 4. Container Security Scanning

#### Trivy (Container & Filesystem Scanner)
```bash
# Install trivy
# See: https://aquasecurity.github.io/trivy/latest/getting-started/installation/

# Scan Docker image
trivy image neorl-industrial:latest

# Scan filesystem
trivy fs .

# Generate reports
trivy image --format json --output trivy-report.json neorl-industrial:latest

# Scan for specific vulnerabilities
trivy image --severity HIGH,CRITICAL neorl-industrial:latest
```

#### Docker Scout (Docker's Native Scanner)
```bash
# Enable Docker Scout
docker scout --help

# Scan image
docker scout cves neorl-industrial:latest

# Compare images
docker scout compare --to neorl-industrial:v1.0.0 neorl-industrial:latest
```

## CI/CD Integration

### GitHub Actions Security Workflow

```yaml
name: Security Scanning

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]
  schedule:
    - cron: '0 2 * * 1'  # Weekly Monday 2 AM

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install safety pip-audit bandit[toml]
          
      - name: Safety Check
        run: |
          safety check --json --output safety-report.json
          
      - name: pip-audit
        run: |
          pip-audit --format=json --output=audit-report.json
          
      - name: Upload Security Reports
        uses: actions/upload-artifact@v3
        with:
          name: security-reports
          path: |
            safety-report.json
            audit-report.json

  code-security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Bandit Security Scan
        run: |
          pip install bandit[toml]
          bandit -r src/ -f json -o bandit-report.json
          
      - name: Semgrep Security Analysis
        uses: returntocorp/semgrep-action@v1
        with:
          config: auto
          
  secret-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
          
      - name: GitLeaks Scan
        uses: zricethezav/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  container-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker Image
        run: docker build -t neorl-industrial:test .
        
      - name: Trivy Container Scan
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'neorl-industrial:test'
          format: 'sarif'
          output: 'trivy-results.sarif'
          
      - name: Upload Trivy Results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

## Security Configuration Files

### .gitleaks.toml
```toml
[extend]
useDefault = true

[[rules]]
description = "Industrial config files"
id = "industrial-config"
regex = '''(?i)(factory_config|plc_config|scada_password|industrial_key)'''
tags = ["industrial", "config"]

[[rules]]
description = "Database connections"
id = "database-connection"
regex = '''(?i)(database_url|db_password|connection_string)'''
tags = ["database", "connection"]

[[rules.allowlist]]
description = "Test files"
paths = ['''tests/.*''']
```

### bandit.yml
```yaml
# Bandit configuration for security scanning
exclude_dirs:
  - tests
  - docs
  - scripts

skips:
  # Skip these only if justified by security review
  # B101: Test for use of assert
  # B601: Shell injection (if using subprocess securely)

tests:
  - B201  # flask_debug_true
  - B501  # request_with_no_cert_validation
  - B506  # yaml_load
  - B602  # subprocess_popen_with_shell_equals_true
  - B608  # possible_sql_injection
  - B703  # django_mark_safe
```

## Industrial Security Requirements

### Safety-Critical Scanning

For industrial applications, implement additional security measures:

```bash
# Custom industrial security rules
semgrep --config=rules/industrial-safety.yaml src/

# PLC/SCADA specific scanning  
semgrep --config=rules/industrial-control.yaml src/

# Safety constraint validation
python scripts/validate_safety_constraints.py --security-scan
```

### Compliance Scanning

```bash
# IEC 62443 cybersecurity standards
python scripts/iec62443_compliance_check.py

# NIST cybersecurity framework alignment
python scripts/nist_compliance_scan.py

# Industrial automation security
python scripts/automation_security_scan.py
```

## Vulnerability Management

### Severity Classification

| Severity | Action Required | Timeline |
|----------|----------------|----------|
| **Critical** | Immediate hotfix | 24 hours |
| **High** | Priority patch | 7 days |
| **Medium** | Scheduled update | 30 days |
| **Low** | Next release | 90 days |

### Response Procedures

1. **Critical Vulnerabilities**:
   - Immediate security team notification
   - Emergency deployment approval
   - Customer security advisory
   - Post-incident review

2. **High Vulnerabilities**:
   - Security team review within 24h
   - Patch development priority
   - Stakeholder notification
   - Testing in staging environment

3. **Medium/Low Vulnerabilities**:
   - Regular security review process
   - Bundled with scheduled releases
   - Documentation updates

## Reporting and Monitoring

### Security Dashboards

```python
# Example security metrics collection
security_metrics = {
    "vulnerabilities_by_severity": {
        "critical": 0,
        "high": 2,
        "medium": 5,
        "low": 8
    },
    "dependency_age": {
        "outdated_count": 3,
        "avg_age_days": 45,
        "max_age_days": 120
    },
    "scan_results": {
        "last_scan": "2024-01-15T10:30:00Z",
        "scan_duration": "00:05:30",
        "files_scanned": 250
    }
}
```

### Alerting Configuration

```yaml
# Example alerting rules
alerts:
  - name: critical_vulnerability_detected
    condition: severity == "CRITICAL"
    actions:
      - slack_notification
      - email_security_team
      - create_incident_ticket
      
  - name: dependency_severely_outdated
    condition: dependency_age > 365
    actions:
      - create_security_issue
      - assign_to_team
```

## Security Automation Scripts

### scripts/security_scan.py
```python
#!/usr/bin/env python3
"""Comprehensive security scanning script."""

import subprocess
import json
import sys
from pathlib import Path

def run_safety_check():
    """Run safety vulnerability check."""
    try:
        result = subprocess.run(
            ["safety", "check", "--json"],
            capture_output=True,
            text=True,
            check=False
        )
        return json.loads(result.stdout) if result.stdout else []
    except Exception as e:
        print(f"Safety check failed: {e}")
        return []

def run_bandit_scan():
    """Run bandit security scan."""
    try:
        result = subprocess.run(
            ["bandit", "-r", "src/", "-f", "json"],
            capture_output=True,
            text=True,
            check=False
        )
        return json.loads(result.stdout) if result.stdout else {}
    except Exception as e:
        print(f"Bandit scan failed: {e}")
        return {}

def main():
    """Run all security scans."""
    print("🔒 Running comprehensive security scan...")
    
    safety_results = run_safety_check()
    bandit_results = run_bandit_scan()
    
    # Process and report results
    critical_issues = []
    
    # Check safety results
    if safety_results:
        critical_issues.extend([
            f"Vulnerable dependency: {vuln['package_name']}"
            for vuln in safety_results
        ])
    
    # Check bandit results
    if bandit_results.get('results'):
        high_severity = [
            issue for issue in bandit_results['results']
            if issue.get('issue_severity') == 'HIGH'
        ]
        critical_issues.extend([
            f"Security issue: {issue['test_name']}"
            for issue in high_severity
        ])
    
    if critical_issues:
        print("❌ Critical security issues found:")
        for issue in critical_issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("✅ No critical security issues found")
        sys.exit(0)

if __name__ == "__main__":
    main()
```

## Emergency Security Procedures

### Incident Response

1. **Detection**: Automated scanning identifies critical vulnerability
2. **Assessment**: Security team evaluates impact and exploitability  
3. **Containment**: Immediate measures to limit exposure
4. **Eradication**: Develop and deploy patches
5. **Recovery**: Restore normal operations with monitoring
6. **Lessons Learned**: Post-incident review and process improvement

### Emergency Contacts

- **Security Team**: security@terragon.ai
- **Industrial Safety**: safety@terragon.ai  
- **Incident Response**: incident@terragon.ai
- **Customer Support**: support@terragon.ai

## Best Practices

### Development Security

1. **Secure Coding**: Follow OWASP guidelines
2. **Dependency Management**: Regular updates and vulnerability scanning
3. **Secret Management**: Use environment variables and secret managers
4. **Code Review**: Security-focused review process
5. **Least Privilege**: Minimal permissions for all components

### Industrial Security

1. **Air-Gapped Testing**: Isolated environment for security testing
2. **Change Control**: Rigorous change management for security updates
3. **Emergency Procedures**: Documented emergency response plans
4. **Compliance Monitoring**: Continuous compliance verification
5. **Third-party Assessment**: Regular security audits

## Maintenance Schedule

- **Daily**: Automated vulnerability scanning
- **Weekly**: Dependency update review
- **Monthly**: Security tool updates and configuration review
- **Quarterly**: Comprehensive security assessment
- **Annually**: Third-party security audit

Remember: Security is not a one-time task but an ongoing process that requires continuous attention and improvement, especially in industrial applications where safety and security are paramount.