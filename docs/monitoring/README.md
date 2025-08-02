# Monitoring and Observability Guide

This document provides comprehensive guidance for monitoring and observability in neoRL-industrial-gym deployments.

## Overview

Industrial RL systems require robust monitoring to ensure:
- **Safety**: Continuous monitoring of safety constraints and violations
- **Performance**: Tracking of system performance and resource utilization
- **Reliability**: Early detection of failures and anomalies
- **Compliance**: Audit trails and regulatory requirement fulfillment

## Monitoring Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   Metrics       │    │   Alerting      │
│                 │───▶│   Collection    │───▶│                 │
│ • Safety Events │    │                 │    │ • PagerDuty     │
│ • Performance   │    │ • Prometheus    │    │ • Slack         │
│ • Errors        │    │ • InfluxDB      │    │ • Email         │
│ • Business      │    │ • Custom        │    │ • SMS           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Logging       │    │   Visualization │    │   Analysis      │
│                 │    │                 │    │                 │
│ • Structured    │    │ • Grafana       │    │ • Jupyter       │
│ • Centralized   │    │ • Custom        │    │ • MLflow        │
│ • Searchable    │    │ • Dashboards    │    │ • Reports       │
│ • Audit Trail   │    │ • Real-time     │    │ • Trends        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Metrics

### Safety Metrics
- **Constraint Violations**: Number and rate of safety constraint violations
- **Emergency Shutdowns**: Frequency and reasons for emergency stops
- **Safety Response Time**: Time from violation detection to response
- **Human Interventions**: Manual overrides and their reasons

### Performance Metrics
- **Inference Latency**: Time to compute actions (target: <100ms)
- **Throughput**: Actions per second, episodes per hour
- **Resource Utilization**: CPU, memory, GPU usage
- **Queue Depths**: Backlog in processing pipelines

### Quality Metrics
- **Reward Tracking**: Episode rewards and trends
- **Policy Performance**: Success rates and task completion
- **Data Quality**: Dataset statistics and anomalies
- **Model Drift**: Changes in model behavior over time

### System Metrics
- **Uptime**: System availability percentage
- **Error Rates**: Application and system error frequencies
- **Network**: Latency, packet loss, bandwidth usage
- **Storage**: Disk usage, I/O rates

## Implementation

### Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "neorl_industrial_rules.yml"

scrape_configs:
  - job_name: 'neorl-industrial'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
    
  - job_name: 'safety-monitor'
    static_configs:
      - targets: ['localhost:8001']
    metrics_path: '/safety/metrics'
    scrape_interval: 1s  # High frequency for safety
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['localhost:9093']
```

### Metrics Collection in Code

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Define metrics
safety_violations = Counter('safety_violations_total', 
                          'Total safety constraint violations',
                          ['constraint_type', 'severity'])

inference_latency = Histogram('inference_latency_seconds',
                            'Time spent on inference',
                            buckets=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0])

active_environments = Gauge('active_environments',
                          'Number of currently active environments')

episode_reward = Histogram('episode_reward',
                         'Reward distribution per episode',
                         buckets=[-100, -10, -1, 0, 1, 10, 100])

# Usage in application
class MonitoredAgent:
    def predict(self, state):
        with inference_latency.time():
            action = self._compute_action(state)
        return action
    
    def check_safety(self, state, action):
        violations = self.safety_checker.validate(state, action)
        for violation in violations:
            safety_violations.labels(
                constraint_type=violation.type,
                severity=violation.severity
            ).inc()
        return len(violations) == 0
```

### Structured Logging

```python
import structlog
import logging.config

# Configure structured logging
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": structlog.stdlib.ProcessorFormatter,
            "processor": structlog.dev.ConsoleRenderer(colors=False),
        },
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "json",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": True,
        }
    }
})

# Configure structlog
structlog.configure(
    processors=[
        structlog.threadlocal.merge_threadlocal,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Usage
logger = structlog.get_logger("neorl.agent")

logger.info("Agent started", 
           agent_type="CQL",
           environment="ChemicalReactor-v0",
           safety_enabled=True)

logger.warning("Safety violation detected",
              constraint="temperature",
              current_value=325.5,
              max_allowed=320.0,
              violation_id="safety_001",
              timestamp=time.time())
```

## Alerting Rules

### Prometheus Alerting Rules

Create `neorl_industrial_rules.yml`:

```yaml
groups:
  - name: safety_alerts
    rules:
      - alert: SafetyViolationRate
        expr: rate(safety_violations_total[1m]) > 0.01
        for: 0s
        labels:
          severity: critical
        annotations:
          summary: "High safety violation rate detected"
          description: "Safety violation rate is {{ $value }} violations per second"
      
      - alert: EmergencyShutdown
        expr: increase(emergency_shutdowns_total[1m]) > 0
        for: 0s
        labels:
          severity: critical
        annotations:
          summary: "Emergency shutdown triggered"
          description: "Emergency shutdown detected in the last minute"

  - name: performance_alerts
    rules:
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, inference_latency_seconds) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High inference latency"
          description: "95th percentile latency is {{ $value }}s"
      
      - alert: LowThroughput
        expr: rate(actions_computed_total[5m]) < 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low system throughput"
          description: "Action computation rate is {{ $value }} actions/second"

  - name: system_alerts
    rules:
      - alert: HighMemoryUsage
        expr: process_resident_memory_bytes / 1024 / 1024 > 1000
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}MB"
      
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "{{ $labels.job }} service is not responding"
```

## Dashboards

### Grafana Dashboard Configuration

Key dashboard panels:

1. **Safety Overview**
   - Safety violation count (last 24h)
   - Time since last violation
   - Violation rate trend
   - Emergency shutdown count

2. **Performance Monitoring**
   - Inference latency percentiles
   - Throughput (actions/sec)
   - Resource utilization
   - Queue depths

3. **Quality Metrics**
   - Episode reward trends
   - Success rates
   - Model performance metrics
   - Data quality indicators

4. **System Health**
   - Service uptime
   - Error rates
   - Resource usage
   - Network metrics

### Example Dashboard JSON

```json
{
  "dashboard": {
    "title": "neoRL Industrial - Safety Monitor",
    "panels": [
      {
        "title": "Safety Violations (24h)",
        "type": "stat",
        "targets": [
          {
            "expr": "increase(safety_violations_total[24h])",
            "legendFormat": "Violations"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "green", "value": 0},
                {"color": "yellow", "value": 1},
                {"color": "red", "value": 5}
              ]
            }
          }
        }
      },
      {
        "title": "Inference Latency",
        "type": "timeseries",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, inference_latency_seconds)",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, inference_latency_seconds)",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.99, inference_latency_seconds)",
            "legendFormat": "99th percentile"
          }
        ]
      }
    ]
  }
}
```

## Health Checks

### Application Health Checks

```python
from flask import Flask, jsonify
import time
from typing import Dict, Any

app = Flask(__name__)

class HealthChecker:
    def __init__(self):
        self.start_time = time.time()
        self.checks = {
            'database': self._check_database,
            'safety_system': self._check_safety_system,
            'model_loaded': self._check_model,
            'memory_usage': self._check_memory,
        }
    
    def _check_database(self) -> Dict[str, Any]:
        # Check database connectivity
        try:
            # Database ping logic here
            return {"status": "healthy", "response_time": 0.05}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def _check_safety_system(self) -> Dict[str, Any]:
        # Verify safety monitoring is active
        try:
            # Safety system check logic
            return {"status": "healthy", "monitoring": True}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def _check_model(self) -> Dict[str, Any]:
        # Verify model is loaded and functional
        try:
            # Model availability check
            return {"status": "healthy", "model_version": "1.0.0"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def _check_memory(self) -> Dict[str, Any]:
        # Check memory usage
        import psutil
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            return {"status": "unhealthy", "usage": f"{memory.percent}%"}
        return {"status": "healthy", "usage": f"{memory.percent}%"}
    
    def get_health_status(self) -> Dict[str, Any]:
        results = {}
        overall_status = "healthy"
        
        for check_name, check_func in self.checks.items():
            result = check_func()
            results[check_name] = result
            if result["status"] != "healthy":
                overall_status = "unhealthy"
        
        return {
            "status": overall_status,
            "uptime": time.time() - self.start_time,
            "timestamp": time.time(),
            "checks": results
        }

health_checker = HealthChecker()

@app.route('/health')
def health():
    """Basic health check endpoint."""
    return jsonify({"status": "healthy", "service": "neorl-industrial"})

@app.route('/health/detailed')
def health_detailed():
    """Detailed health check with system status."""
    return jsonify(health_checker.get_health_status())

@app.route('/health/ready')
def readiness():
    """Readiness probe for Kubernetes."""
    status = health_checker.get_health_status()
    if status["status"] == "healthy":
        return jsonify(status), 200
    else:
        return jsonify(status), 503

@app.route('/health/live')
def liveness():
    """Liveness probe for Kubernetes."""
    return jsonify({
        "status": "alive",
        "uptime": time.time() - health_checker.start_time
    })
```

## Runbooks

### Safety Violation Response

1. **Immediate Actions** (0-2 minutes)
   - Verify alert is not false positive
   - Check if emergency shutdown triggered
   - Assess current system state
   - Notify safety team

2. **Investigation** (2-15 minutes)
   - Review violation details in logs
   - Check related metrics and trends
   - Identify root cause
   - Document findings

3. **Resolution** (15+ minutes)
   - Implement fix if identified
   - Update safety constraints if needed
   - Test system recovery
   - Document lessons learned

### Performance Degradation Response

1. **Assessment** (0-5 minutes)
   - Check system resource usage
   - Verify network connectivity
   - Review recent deployments
   - Check for external dependencies

2. **Mitigation** (5-30 minutes)
   - Scale resources if needed
   - Restart services if appropriate
   - Switch to backup systems
   - Implement circuit breakers

3. **Recovery** (30+ minutes)
   - Monitor system recovery
   - Conduct root cause analysis
   - Update monitoring thresholds
   - Document incident

## Deployment

### Docker Compose with Monitoring

```yaml
version: '3.8'
services:
  neorl-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PROMETHEUS_ENABLED=true
      - METRICS_PORT=8000
    depends_on:
      - prometheus
      - grafana
  
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/rules:/etc/prometheus/rules
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
  
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
  
  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml

volumes:
  grafana-storage:
```

### Kubernetes Deployment

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    # Prometheus configuration here

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neorl-industrial
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neorl-industrial
  template:
    metadata:
      labels:
        app: neorl-industrial
    spec:
      containers:
      - name: neorl-app
        image: terragonlabs/neorl-industrial:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

## Best Practices

### Monitoring
- Use pull-based metrics collection for reliability
- Implement proper cardinality limits to avoid metric explosion
- Monitor the monitoring system itself
- Use service discovery when possible

### Alerting
- Follow the principle of "alert on symptoms, not causes"
- Implement proper alert routing and escalation
- Use runbooks for consistent incident response
- Regular testing of alert configurations

### Logging
- Use structured logging with consistent schemas
- Include correlation IDs for distributed tracing
- Log at appropriate levels (avoid log spam)
- Implement log rotation and retention policies

### Security
- Secure monitoring endpoints with authentication
- Use encrypted connections for metric transmission
- Implement proper access controls for dashboards
- Regular security audits of monitoring infrastructure

For more detailed information, see:
- [Prometheus Setup Guide](prometheus-setup.md)
- [Grafana Dashboard Templates](grafana-templates.md)
- [Alerting Configuration](alerting-config.md)
- [Security Monitoring](security-monitoring.md)