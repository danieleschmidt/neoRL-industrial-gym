# neoRL-industrial-gym Comprehensive Deployment Guide

## 🚀 Production Deployment Guide

This guide provides comprehensive instructions for deploying neoRL-industrial-gym in production environments with enterprise-grade reliability, security, and scalability.

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Environment Setup](#environment-setup)
3. [Container Deployment](#container-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Security Configuration](#security-configuration)
6. [Monitoring & Observability](#monitoring--observability)
7. [High Availability Setup](#high-availability-setup)
8. [Performance Optimization](#performance-optimization)
9. [Compliance & Data Protection](#compliance--data-protection)
10. [Disaster Recovery](#disaster-recovery)
11. [Troubleshooting](#troubleshooting)

## Pre-Deployment Checklist

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores (x86_64 or ARM64)
- RAM: 8GB
- Storage: 50GB SSD
- Network: 1Gbps
- OS: Ubuntu 20.04+ / RHEL 8+ / macOS 11+

**Recommended for Production:**
- CPU: 16+ cores with AVX support
- RAM: 32GB+
- Storage: 200GB+ NVMe SSD
- Network: 10Gbps
- GPU: NVIDIA V100/A100 (for large-scale training)

### Software Dependencies

```bash
# Python 3.8+ with development headers
sudo apt-get update
sudo apt-get install -y python3.8-dev python3-pip

# JAX dependencies
sudo apt-get install -y build-essential

# For GPU support (optional)
# Follow NVIDIA CUDA installation guide

# System monitoring tools
sudo apt-get install -y htop iotop nethogs
```

### Security Prerequisites

- [ ] SSL/TLS certificates configured
- [ ] Firewall rules defined
- [ ] Security groups configured (cloud environments)
- [ ] Service accounts created with minimal permissions
- [ ] Secret management system (HashiCorp Vault, AWS Secrets Manager, etc.)
- [ ] Network segmentation implemented
- [ ] Intrusion detection system configured

## Environment Setup

### 1. Python Environment

```bash
# Create dedicated environment
python3 -m venv neorl-prod
source neorl-prod/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install neoRL-industrial-gym
pip install -e .

# Install production dependencies
pip install gunicorn supervisor redis prometheus-client
```

### 2. Environment Variables

Create `.env` file:

```bash
# Application Configuration
NEORL_ENV=production
NEORL_LOG_LEVEL=INFO
NEORL_LOG_FILE=/var/log/neorl/application.log

# Security
NEORL_SECRET_KEY=your-256-bit-secret-key
NEORL_ENCRYPTION_KEY=your-encryption-key

# Database (if using persistent storage)
NEORL_DB_URL=postgresql://user:pass@localhost/neorl_db

# Monitoring
NEORL_PROMETHEUS_PORT=9090
NEORL_HEALTH_CHECK_PORT=8080

# Performance
NEORL_MAX_WORKERS=4
NEORL_BATCH_SIZE=256
NEORL_CACHE_SIZE_MB=1024

# Compliance
NEORL_GDPR_ENABLED=true
NEORL_DATA_RETENTION_DAYS=365
NEORL_AUDIT_LOG_PATH=/var/log/neorl/audit.log
```

### 3. Directory Structure

```bash
# Create production directory structure
sudo mkdir -p /opt/neorl/{app,data,logs,config}
sudo mkdir -p /var/log/neorl
sudo mkdir -p /var/lib/neorl

# Set permissions
sudo chown -R neorl:neorl /opt/neorl /var/log/neorl /var/lib/neorl
sudo chmod 755 /opt/neorl
sudo chmod 640 /opt/neorl/config/*
```

## Container Deployment

### 1. Dockerfile Optimization

```dockerfile
# Multi-stage production Dockerfile
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r neorl && useradd -r -g neorl neorl

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN groupadd -r neorl && useradd -r -g neorl neorl

# Application code
WORKDIR /app
COPY --chown=neorl:neorl . .

# Security hardening
RUN chown -R neorl:neorl /app
USER neorl

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Expose ports
EXPOSE 8000 8080 9090

# Run application
CMD ["python", "-m", "neorl_industrial.server", "--production"]
```

### 2. Docker Compose for Production

```yaml
version: '3.8'

services:
  neorl-app:
    build: .
    container_name: neorl-production
    restart: unless-stopped
    
    environment:
      - NEORL_ENV=production
      - NEORL_LOG_LEVEL=INFO
    
    volumes:
      - neorl-data:/var/lib/neorl
      - neorl-logs:/var/log/neorl
      - ./config:/opt/neorl/config:ro
    
    ports:
      - "8000:8000"  # Application
      - "8080:8080"  # Health checks
      - "9090:9090"  # Metrics
    
    networks:
      - neorl-network
    
    depends_on:
      - redis
      - postgres
    
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
    
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"

  redis:
    image: redis:7-alpine
    container_name: neorl-redis
    restart: unless-stopped
    
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    
    volumes:
      - redis-data:/data
    
    networks:
      - neorl-network
    
    deploy:
      resources:
        limits:
          memory: 512M

  postgres:
    image: postgres:14-alpine
    container_name: neorl-postgres
    restart: unless-stopped
    
    environment:
      POSTGRES_DB: neorl_db
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    
    volumes:
      - postgres-data:/var/lib/postgresql/data
    
    networks:
      - neorl-network
    
    deploy:
      resources:
        limits:
          memory: 2G

  prometheus:
    image: prom/prometheus:latest
    container_name: neorl-prometheus
    restart: unless-stopped
    
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    
    ports:
      - "9091:9090"
    
    networks:
      - neorl-network

  grafana:
    image: grafana/grafana:latest
    container_name: neorl-grafana
    restart: unless-stopped
    
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
    
    volumes:
      - grafana-data:/var/lib/grafana
    
    ports:
      - "3000:3000"
    
    networks:
      - neorl-network

volumes:
  neorl-data:
  neorl-logs:
  redis-data:
  postgres-data:
  prometheus-data:
  grafana-data:

networks:
  neorl-network:
    driver: bridge
```

## Kubernetes Deployment

### 1. Namespace and RBAC

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: neorl-production
  labels:
    name: neorl-production
    tier: production

---
# rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: neorl-service-account
  namespace: neorl-production

---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: neorl-production
  name: neorl-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch"]

---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: neorl-role-binding
  namespace: neorl-production
subjects:
- kind: ServiceAccount
  name: neorl-service-account
  namespace: neorl-production
roleRef:
  kind: Role
  name: neorl-role
  apiGroup: rbac.authorization.k8s.io
```

### 2. ConfigMap and Secrets

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: neorl-config
  namespace: neorl-production
data:
  NEORL_ENV: "production"
  NEORL_LOG_LEVEL: "INFO"
  NEORL_MAX_WORKERS: "4"
  NEORL_BATCH_SIZE: "256"
  NEORL_CACHE_SIZE_MB: "1024"

---
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: neorl-secrets
  namespace: neorl-production
type: Opaque
data:
  # Base64 encoded values
  NEORL_SECRET_KEY: <base64-encoded-secret>
  NEORL_ENCRYPTION_KEY: <base64-encoded-key>
  DB_PASSWORD: <base64-encoded-password>
```

### 3. Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neorl-deployment
  namespace: neorl-production
  labels:
    app: neorl
    tier: production
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  
  selector:
    matchLabels:
      app: neorl
  
  template:
    metadata:
      labels:
        app: neorl
        tier: production
    spec:
      serviceAccountName: neorl-service-account
      
      containers:
      - name: neorl
        image: neorl-industrial:latest
        imagePullPolicy: Always
        
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8080
          name: health
        - containerPort: 9090
          name: metrics
        
        envFrom:
        - configMapRef:
            name: neorl-config
        - secretRef:
            name: neorl-secrets
        
        resources:
          requests:
            cpu: 1000m
            memory: 2Gi
          limits:
            cpu: 4000m
            memory: 8Gi
        
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        
        volumeMounts:
        - name: neorl-data
          mountPath: /var/lib/neorl
        - name: neorl-logs
          mountPath: /var/log/neorl
      
      volumes:
      - name: neorl-data
        persistentVolumeClaim:
          claimName: neorl-data-pvc
      - name: neorl-logs
        persistentVolumeClaim:
          claimName: neorl-logs-pvc

---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: neorl-service
  namespace: neorl-production
  labels:
    app: neorl
spec:
  selector:
    app: neorl
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: metrics
    port: 9090
    targetPort: 9090
  type: ClusterIP

---
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: neorl-hpa
  namespace: neorl-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neorl-deployment
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Security Configuration

### 1. Network Security

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: neorl-network-policy
  namespace: neorl-production
spec:
  podSelector:
    matchLabels:
      app: neorl
  policyTypes:
  - Ingress
  - Egress
  
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8000
  
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

### 2. Pod Security Policy

```yaml
# pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: neorl-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  
  requiredDropCapabilities:
    - ALL
  
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  
  runAsUser:
    rule: 'MustRunAsNonRoot'
  
  seLinux:
    rule: 'RunAsAny'
  
  fsGroup:
    rule: 'RunAsAny'
```

### 3. TLS Configuration

```bash
# Generate TLS certificates
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout tls.key -out tls.crt \
    -subj "/CN=neorl.yourdomain.com/O=YourOrg"

# Create TLS secret
kubectl create secret tls neorl-tls \
    --cert=tls.crt --key=tls.key \
    -n neorl-production
```

## Monitoring & Observability

### 1. Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "neorl_rules.yml"

scrape_configs:
  - job_name: 'neorl'
    static_configs:
      - targets: ['neorl-service:9090']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - neorl-production
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

### 2. Grafana Dashboard

```json
{
  "dashboard": {
    "title": "neoRL Industrial - Production Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(neorl_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "title": "Training Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "neorl_training_samples_per_second",
            "legendFormat": "Samples/sec"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "neorl_memory_usage_bytes / 1024 / 1024",
            "legendFormat": "Memory (MB)"
          }
        ]
      },
      {
        "title": "Safety Violations",
        "type": "singlestat",
        "targets": [
          {
            "expr": "sum(rate(neorl_safety_violations_total[1h]))",
            "legendFormat": "Violations/hour"
          }
        ]
      }
    ]
  }
}
```

### 3. Alerting Rules

```yaml
# neorl_rules.yml
groups:
- name: neorl.rules
  rules:
  - alert: NeoRLHighErrorRate
    expr: rate(neorl_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate in neoRL"
      description: "Error rate is {{ $value }} errors per second"

  - alert: NeoRLHighMemoryUsage
    expr: neorl_memory_usage_bytes / neorl_memory_limit_bytes > 0.9
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage in neoRL"
      description: "Memory usage is {{ $value }}%"

  - alert: NeoRLSafetyViolations
    expr: rate(neorl_safety_violations_total[1h]) > 10
    for: 0m
    labels:
      severity: critical
    annotations:
      summary: "High rate of safety violations"
      description: "{{ $value }} safety violations in the last hour"
```

## High Availability Setup

### 1. Database High Availability

```yaml
# postgres-ha.yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: postgres-cluster
  namespace: neorl-production
spec:
  instances: 3
  
  postgresql:
    parameters:
      max_connections: "200"
      shared_buffers: "256MB"
      effective_cache_size: "1GB"
  
  bootstrap:
    initdb:
      database: neorl_db
      owner: neorl_user
      secret:
        name: postgres-credentials
  
  storage:
    size: 100Gi
    storageClass: fast-ssd
  
  monitoring:
    enabled: true
```

### 2. Redis Cluster

```yaml
# redis-cluster.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
  namespace: neorl-production
spec:
  serviceName: redis-cluster
  replicas: 6
  selector:
    matchLabels:
      app: redis-cluster
  template:
    metadata:
      labels:
        app: redis-cluster
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command:
          - redis-server
          - /etc/redis/redis.conf
          - --cluster-enabled
          - "yes"
          - --cluster-config-file
          - nodes.conf
          - --cluster-node-timeout
          - "5000"
        ports:
        - containerPort: 6379
          name: client
        - containerPort: 16379
          name: gossip
        volumeMounts:
        - name: redis-data
          mountPath: /data
        - name: redis-config
          mountPath: /etc/redis
  volumeClaimTemplates:
  - metadata:
      name: redis-data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 10Gi
```

### 3. Load Balancer Configuration

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: neorl-ingress
  namespace: neorl-production
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/backend-protocol: "HTTP"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - neorl.yourdomain.com
    secretName: neorl-tls
  rules:
  - host: neorl.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: neorl-service
            port:
              number: 80
```

## Performance Optimization

### 1. JVM Tuning (for JAX)

```bash
# Set JAX environment variables
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export JAX_ENABLE_X64=True
export JAX_PLATFORM_NAME=gpu  # or cpu

# Memory optimization
export JAX_PYTHON_EXECUTABLE_CACHE_SIZE=1000000000  # 1GB
```

### 2. Resource Limits

```yaml
# resource-quotas.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: neorl-quota
  namespace: neorl-production
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "40"
    limits.memory: 80Gi
    persistentvolumeclaims: "10"
```

### 3. Caching Strategy

```python
# Production caching configuration
CACHE_CONFIG = {
    "default": {
        "BACKEND": "redis",
        "LOCATION": "redis://redis-cluster:6379/0",
        "OPTIONS": {
            "CONNECTION_POOL_KWARGS": {
                "max_connections": 50,
                "retry_on_timeout": True,
            }
        },
        "KEY_PREFIX": "neorl",
        "TIMEOUT": 3600,  # 1 hour
    },
    "models": {
        "BACKEND": "redis",
        "LOCATION": "redis://redis-cluster:6379/1",
        "TIMEOUT": 86400,  # 24 hours
    }
}
```

## Compliance & Data Protection

### 1. GDPR Configuration

```python
# GDPR settings
GDPR_SETTINGS = {
    "enabled": True,
    "data_retention_days": 365,
    "auto_cleanup": True,
    "consent_required": True,
    "audit_log_path": "/var/log/neorl/gdpr_audit.log",
    "encryption_enabled": True,
}
```

### 2. Audit Logging

```yaml
# audit-policy.yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
- level: Namespace
  namespaces: ["neorl-production"]
  verbs: ["get", "list", "create", "update", "patch", "delete"]
  resources:
  - group: ""
    resources: ["secrets", "configmaps"]
  - group: "apps"
    resources: ["deployments", "replicasets"]
```

## Disaster Recovery

### 1. Backup Strategy

```bash
#!/bin/bash
# backup.sh

# Database backup
kubectl exec -n neorl-production postgres-cluster-1 -- \
    pg_dump -U neorl_user neorl_db | \
    gzip > "/backups/db-$(date +%Y%m%d-%H%M%S).sql.gz"

# Volume backup
kubectl create -n neorl-production job backup-job --image=backup-tool \
    --restart=Never -- /scripts/volume-backup.sh

# Configuration backup
kubectl get all,configmaps,secrets -n neorl-production -o yaml > \
    "/backups/k8s-config-$(date +%Y%m%d).yaml"
```

### 2. Restore Procedures

```bash
#!/bin/bash
# restore.sh

# Restore database
gunzip -c /backups/db-latest.sql.gz | \
    kubectl exec -i -n neorl-production postgres-cluster-1 -- \
    psql -U neorl_user neorl_db

# Restore configurations
kubectl apply -f /backups/k8s-config-latest.yaml
```

## Troubleshooting

### Common Issues

1. **Memory Issues**
   ```bash
   # Check memory usage
   kubectl top pods -n neorl-production
   
   # Increase memory limits
   kubectl patch deployment neorl-deployment -n neorl-production -p \
     '{"spec":{"template":{"spec":{"containers":[{"name":"neorl","resources":{"limits":{"memory":"16Gi"}}}]}}}}'
   ```

2. **Training Performance**
   ```bash
   # Check GPU utilization
   kubectl exec -it -n neorl-production neorl-deployment-xxx -- nvidia-smi
   
   # Monitor training metrics
   kubectl logs -f -n neorl-production neorl-deployment-xxx | grep "training_throughput"
   ```

3. **Network Issues**
   ```bash
   # Test connectivity
   kubectl exec -it -n neorl-production neorl-deployment-xxx -- \
     curl -I http://postgres-cluster:5432
   
   # Check network policies
   kubectl describe networkpolicy -n neorl-production
   ```

### Debugging Tools

```bash
# Production debugging toolkit
kubectl create -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: debug-toolkit
  namespace: neorl-production
spec:
  containers:
  - name: debug
    image: nicolaka/netshoot
    command: ["sleep", "3600"]
  restartPolicy: Never
EOF

# Access debug pod
kubectl exec -it -n neorl-production debug-toolkit -- bash
```

### Performance Monitoring

```bash
# Real-time performance monitoring
watch -n 1 'kubectl top pods -n neorl-production'

# Get detailed resource usage
kubectl describe nodes | grep -A 5 "Allocated resources"

# Monitor application logs
kubectl logs -f -n neorl-production deployment/neorl-deployment
```

## Maintenance

### Rolling Updates

```bash
# Update image
kubectl set image deployment/neorl-deployment \
    neorl=neorl-industrial:v2.0.0 -n neorl-production

# Monitor rollout
kubectl rollout status deployment/neorl-deployment -n neorl-production

# Rollback if needed
kubectl rollout undo deployment/neorl-deployment -n neorl-production
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment neorl-deployment --replicas=5 -n neorl-production

# Auto-scaling configuration
kubectl autoscale deployment neorl-deployment \
    --cpu-percent=70 --min=3 --max=10 -n neorl-production
```

This deployment guide provides a comprehensive foundation for running neoRL-industrial-gym in production environments with enterprise-grade reliability, security, and observability.