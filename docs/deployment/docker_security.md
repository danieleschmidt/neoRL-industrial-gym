# Docker Security Guide

This document outlines security best practices and configurations for neoRL-industrial-gym containerized deployments.

## Security Architecture

### Multi-layered Security Approach

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Security                      │
├─────────────────────────────────────────────────────────────┤
│                  Container Security                         │
├─────────────────────────────────────────────────────────────┤
│                   Runtime Security                          │
├─────────────────────────────────────────────────────────────┤
│                     Host Security                           │
├─────────────────────────────────────────────────────────────┤
│                  Network Security                           │
└─────────────────────────────────────────────────────────────┘
```

## Container Security Features

### 1. Non-Root User Execution

All containers run as non-privileged user `neorl`:

```dockerfile
# Create non-root user
RUN groupadd -r neorl && useradd -r -g neorl neorl

# Switch to non-root user before CMD
USER neorl
```

**Security Benefits:**
- Limits blast radius of potential exploits
- Prevents privilege escalation attacks
- Follows principle of least privilege

### 2. Base Image Security

Using minimal, security-hardened base images:

```dockerfile
# Use official Python slim image (smaller attack surface)
FROM python:3.10-slim as base

# Apply security updates during build
RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
```

**Security Benefits:**
- Reduced attack surface
- Fewer installed packages
- Regular security updates

### 3. Multi-stage Builds

Separate stages for development, production, and GPU:

```dockerfile
# Development stage (includes dev tools)
FROM base as development

# Production stage (minimal dependencies only)
FROM base as production

# GPU stage (CUDA-specific optimizations)
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as gpu
```

**Security Benefits:**
- Production images exclude development tools
- Smaller final image size
- Better separation of concerns

### 4. File System Permissions

Proper ownership and permissions:

```dockerfile
# Set proper permissions on app directory
RUN chown -R neorl:neorl /app

# Copy files with correct ownership
COPY --chown=neorl:neorl src/ ./src/
```

### 5. Environment Variable Security

Secure handling of sensitive environment variables:

```bash
# Use secrets management instead of plain env vars for production
# Example with Docker Swarm secrets:
docker service create \
  --name neorl-prod \
  --secret db-password \
  --secret api-key \
  neorl-industrial:prod
```

## Runtime Security

### 1. Resource Limits

Container resource constraints to prevent DoS:

```yaml
# docker-compose.yml
services:
  neorl-prod:
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
```

### 2. Read-only Root Filesystem

Prevent runtime modifications:

```yaml
services:
  neorl-prod:
    read_only: true
    tmpfs:
      - /tmp
      - /var/tmp
```

### 3. No New Privileges

Prevent privilege escalation:

```yaml
services:
  neorl-prod:
    security_opt:
      - no-new-privileges:true
```

### 4. Capabilities Dropping

Drop unnecessary Linux capabilities:

```yaml
services:
  neorl-prod:
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - DAC_OVERRIDE
      - SETUID
      - SETGID
```

### 5. AppArmor/SELinux

Additional mandatory access controls:

```yaml
services:
  neorl-prod:
    security_opt:
      - apparmor:docker-default
      # or for SELinux:
      # - label:type:container_t
```

## Network Security

### 1. Network Segmentation

Isolated Docker networks:

```yaml
networks:
  neorl-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### 2. Port Exposure Minimization

Only expose necessary ports:

```yaml
# Only expose ports that need external access
ports:
  - "127.0.0.1:5000:5000"  # Bind to localhost only
```

### 3. Internal Service Communication

Use service names for internal communication:

```yaml
environment:
  - MLFLOW_TRACKING_URI=http://mlflow:5000  # Internal service name
```

## Secrets Management

### 1. Docker Secrets (Swarm Mode)

```bash
# Create secrets
echo "secure_password" | docker secret create db_password -
echo "api_key_value" | docker secret create api_key -

# Use secrets in service
docker service create \
  --name neorl-prod \
  --secret db_password \
  --secret api_key \
  neorl-industrial:prod
```

### 2. External Secret Management

Integration with enterprise secret managers:

```yaml
# Example with external secrets operator
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
spec:
  provider:
    vault:
      server: "https://vault.example.com:8200"
      path: "secret"
```

### 3. Environment File Security

Secure handling of `.env` files:

```bash
# Set restrictive permissions on .env files
chmod 600 .env
chown root:root .env

# Never commit .env files to version control
echo ".env" >> .gitignore
```

## Image Security Scanning

### 1. Build-time Scanning

Scan images during build:

```bash
# Using Docker Scout
docker scout cves neorl-industrial:latest

# Using Trivy
trivy image neorl-industrial:latest

# Using Clair
clair-scanner neorl-industrial:latest
```

### 2. Registry Scanning

Continuous scanning in registry:

```yaml
# GitHub Actions example
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: 'neorl-industrial:${{ github.sha }}'
    format: 'sarif'
    output: 'trivy-results.sarif'
```

### 3. Runtime Scanning

Monitor running containers:

```bash
# Use Falco for runtime security monitoring
sudo falco -r /etc/falco/falco_rules.yaml
```

## Industrial Security Considerations

### 1. Air-gapped Environments

For isolated industrial networks:

```bash
# Build and save images for offline transfer
docker build -t neorl-industrial:prod .
docker save neorl-industrial:prod > neorl-industrial-prod.tar

# Load on air-gapped system
docker load < neorl-industrial-prod.tar
```

### 2. Compliance Requirements

Meeting industrial standards:

- **IEC 62443**: Industrial communication networks security
- **NIST Cybersecurity Framework**: Risk management
- **ISO 27001**: Information security management
- **SOC 2**: Service organization controls

### 3. Audit Logging

Comprehensive logging for compliance:

```yaml
services:
  neorl-prod:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"
        labels: "service,version"
        env: "NODE_ENV,ENVIRONMENT"
```

## Security Monitoring and Alerting

### 1. Health Check Security

Secure health check endpoints:

```dockerfile
# Health check without exposing sensitive info
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import neorl_industrial; print('OK')" || exit 1
```

### 2. Security Metrics

Monitor security-relevant metrics:

```yaml
# Prometheus monitoring for security events
services:
  security-exporter:
    image: security-metrics-exporter
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
```

### 3. Intrusion Detection

Runtime threat detection:

```bash
# Example using OSSEC
docker run -d \
  --name ossec \
  -v /var/log:/var/log:ro \
  -v /var/run/docker.sock:/var/run/docker.sock:ro \
  ossec/ossec-docker
```

## Incident Response

### 1. Container Isolation

Isolate compromised containers:

```bash
# Remove from network
docker network disconnect neorl-network compromised-container

# Stop container
docker stop compromised-container

# Preserve for forensics
docker commit compromised-container forensics-snapshot
```

### 2. Emergency Procedures

Automated emergency response:

```bash
#!/bin/bash
# emergency-response.sh

# Stop all non-essential services
docker-compose stop neorl-dev jupyter

# Maintain only critical services
docker-compose up -d neorl-prod mlflow postgres

# Alert security team
curl -X POST "https://security-webhook.example.com/alert" \
  -d '{"event": "container_compromise", "timestamp": "'$(date -Iseconds)'"}'
```

## Security Best Practices Checklist

### Build Security

- [ ] Use minimal base images
- [ ] Apply security updates during build
- [ ] Scan images for vulnerabilities
- [ ] Remove unnecessary packages and files
- [ ] Use multi-stage builds
- [ ] Set proper file permissions
- [ ] Use non-root user

### Runtime Security

- [ ] Run with non-root user
- [ ] Drop unnecessary capabilities
- [ ] Use read-only root filesystem where possible
- [ ] Set resource limits
- [ ] Enable AppArmor/SELinux
- [ ] Use network segmentation
- [ ] Implement proper logging

### Secrets Management

- [ ] Never embed secrets in images
- [ ] Use external secret management
- [ ] Rotate secrets regularly
- [ ] Audit secret access
- [ ] Use encrypted communication

### Monitoring and Response

- [ ] Implement security monitoring
- [ ] Set up alerting for security events
- [ ] Regular security assessments
- [ ] Incident response procedures
- [ ] Compliance auditing

## Security Tools Integration

### 1. Container Scanning Tools

```bash
# Integrate multiple scanners for comprehensive coverage
make security-scan  # Runs all configured scanners
```

### 2. Policy Enforcement

```yaml
# Open Policy Agent (OPA) Gatekeeper policies
apiVersion: config.gatekeeper.sh/v1alpha1
kind: Config
metadata:
  name: config
spec:
  security:
    - apiGroups: [""]
      kinds: ["Pod"]
      name: "security-policy"
```

### 3. Compliance Automation

```bash
# Automated compliance checking
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  docker/docker-bench-security
```

This security configuration ensures that neoRL-industrial-gym containers meet industrial-grade security requirements while maintaining operational efficiency and compliance with regulatory standards.