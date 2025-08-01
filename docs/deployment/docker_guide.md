# Docker Deployment Guide for neoRL-industrial-gym

## Overview

This guide provides comprehensive instructions for deploying neoRL-industrial-gym using Docker containers across different environments: development, production, and GPU-accelerated setups.

## Container Architecture

The project uses a multi-stage Docker build strategy:

- **Base Stage**: Common dependencies and security setup
- **Development Stage**: Full development tools and debugging capabilities
- **Production Stage**: Minimal runtime with production optimizations
- **GPU Stage**: CUDA-enabled environment for accelerated training

## Quick Start

### Development Environment

Start the complete development stack:

```bash
# Start development environment with all services
docker-compose up neorl-dev mlflow postgres

# Access Jupyter Lab
open http://localhost:8888

# Access MLflow UI
open http://localhost:5000
```

### Production Deployment

Deploy the production-ready stack:

```bash
# Build and start production services
docker-compose up neorl-prod mlflow postgres -d

# Check service health
docker-compose ps
docker-compose logs neorl-prod
```

### GPU Training

Enable GPU acceleration (requires NVIDIA Docker runtime):

```bash
# Start GPU-enabled training environment
docker-compose --profile gpu up neorl-gpu mlflow postgres -d

# Monitor GPU usage
docker exec neorl-industrial-gpu nvidia-smi
```

## Detailed Configuration

### 1. Development Setup

The development container includes:
- Full Python development environment
- Jupyter Lab with ML extensions
- Testing and debugging tools
- Hot-reload capability
- Pre-commit hooks

**Key features:**
- Volume mounting for live code changes
- Comprehensive logging and debugging
- Integration with MLflow and PostgreSQL
- Port forwarding for all development services

```bash
# Interactive development shell
docker-compose exec neorl-dev bash

# Run tests in container
docker-compose exec neorl-dev pytest tests/ -v

# Install additional packages
docker-compose exec neorl-dev pip install <package>
```

### 2. Production Deployment

The production container is optimized for:
- Minimal attack surface
- Fast startup times
- Resource efficiency
- Security hardening

**Security features:**
- Non-root user execution
- Minimal system dependencies
- Security-updated base images
- Health checks and monitoring

```bash
# Scale production instances
docker-compose up neorl-prod --scale neorl-prod=3

# View production logs
docker-compose logs -f neorl-prod

# Update production deployment
docker-compose pull neorl-prod
docker-compose up -d neorl-prod
```

### 3. GPU Configuration

GPU containers support:
- CUDA 11.8 runtime
- JAX GPU acceleration
- Memory management
- Multi-GPU setups

**Prerequisites:**
- NVIDIA Docker runtime installed
- Compatible GPU drivers
- CUDA toolkit (handled in container)

```bash
# Verify GPU setup
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Monitor GPU memory during training
docker exec neorl-industrial-gpu watch -n 1 nvidia-smi
```

## Service Configuration

### MLflow Tracking Server

MLflow provides experiment tracking and model registry:

```bash
# Access MLflow UI
open http://localhost:5000

# API endpoint for programmatic access
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# Backup MLflow data
docker-compose exec postgres pg_dump -U mlflow mlflow > mlflow_backup.sql
```

### PostgreSQL Database

PostgreSQL stores MLflow metadata:

```bash
# Connect to database
docker-compose exec postgres psql -U mlflow -d mlflow

# Database backup
docker-compose exec postgres pg_dump -U mlflow mlflow > backup.sql

# Database restore
docker-compose exec -T postgres psql -U mlflow -d mlflow < backup.sql
```

### Monitoring Stack (Optional)

Enable Prometheus and Grafana monitoring:

```bash
# Start monitoring stack
docker-compose --profile monitoring up prometheus grafana -d

# Access Grafana dashboard
open http://localhost:3000
# Login: admin/admin

# View Prometheus metrics
open http://localhost:9090
```

## Environment Variables

### Core Configuration

```bash
# JAX Configuration
JAX_PLATFORM_NAME=cpu|gpu
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

# MLflow Integration
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_EXPERIMENT_NAME=neorl-industrial-default

# neoRL Configuration
NEORL_DEV_MODE=true|false
NEORL_DATA_ROOT=/app/data
NEORL_LOG_LEVEL=INFO
```

### Security Variables

```bash
# Database credentials (use secrets in production)
POSTGRES_DB=mlflow
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=<secure-password>

# API keys and tokens
MLFLOW_TRACKING_TOKEN=<secure-token>
NEORL_API_KEY=<api-key>
```

## Volume Management

### Persistent Data

The stack uses named volumes for persistent data:

```bash
# List volumes
docker volume ls | grep neorl

# Inspect volume contents
docker run --rm -v neorl-mlflow-data:/data alpine ls -la /data

# Backup volumes
docker run --rm -v neorl-mlflow-data:/data -v $(pwd):/backup alpine tar czf /backup/mlflow-backup.tar.gz /data
```

### Volume Types

- **mlflow-data**: MLflow experiments and artifacts
- **postgres-data**: Database storage
- **model-artifacts**: Trained model files
- **jupyter-data**: Jupyter configuration and notebooks
- **neorl-cache**: Python package cache
- **gpu-cache**: GPU-specific cache files

## Networking

### Service Communication

Services communicate through the `neorl-network` bridge:

```bash
# Inspect network
docker network inspect neorl-industrial-gym_neorl-network

# Test connectivity
docker-compose exec neorl-dev ping mlflow
docker-compose exec neorl-dev curl http://mlflow:5000/health
```

### Port Mapping

| Service | Internal Port | External Port | Purpose |
|---------|---------------|---------------|---------|
| neorl-dev | 8888 | 8888 | Jupyter Lab |
| neorl-dev | 8080 | 8080 | Development server |
| neorl-prod | 8000 | 8000 | Production API |
| neorl-gpu | 8000 | 8001 | GPU training API |
| mlflow | 5000 | 5000 | MLflow UI |
| postgres | 5432 | 5432 | Database |
| prometheus | 9090 | 9090 | Metrics |
| grafana | 3000 | 3000 | Dashboards |

## Build Optimization

### Multi-stage Benefits

1. **Smaller Production Images**: Only necessary files included
2. **Layer Caching**: Efficient rebuilds during development
3. **Security**: Minimal attack surface in production
4. **Flexibility**: Different configurations for different use cases

### Build Arguments

```bash
# Build specific stage
docker build --target development -t neorl-dev .
docker build --target production -t neorl-prod .
docker build --target gpu -t neorl-gpu .

# Build with cache
docker build --cache-from neorl-dev --target development -t neorl-dev .

# Build without cache (clean build)
docker build --no-cache -t neorl-prod .
```

## Health Checks

All services include comprehensive health checks:

```bash
# Check service health
docker-compose ps

# View health check logs
docker inspect neorl-industrial-dev | jq '.[0].State.Health'

# Manual health check
docker-compose exec neorl-dev python -c "import neorl_industrial; print('OK')"
```

## Troubleshooting

### Common Issues

#### 1. Permission Errors

```bash
# Fix volume permissions
docker-compose exec neorl-dev chown -R neorl:neorl /app

# Check user context
docker-compose exec neorl-dev id
```

#### 2. Memory Issues

```bash
# Monitor memory usage
docker stats neorl-industrial-dev

# Adjust memory limits in docker-compose.yml
services:
  neorl-dev:
    deploy:
      resources:
        limits:
          memory: 4G
```

#### 3. GPU Not Available

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Check Docker daemon configuration
sudo systemctl status docker
cat /etc/docker/daemon.json
```

#### 4. Database Connection Issues

```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Test database connectivity
docker-compose exec neorl-dev python -c "
import psycopg2
conn = psycopg2.connect(
    host='postgres',
    database='mlflow',
    user='mlflow',
    password='mlflow'
)
print('Database connection OK')
"
```

### Debugging Commands

```bash
# Interactive debugging
docker-compose exec neorl-dev bash
docker-compose exec neorl-dev python -i

# View logs
docker-compose logs -f neorl-dev
docker-compose logs --tail=100 mlflow

# Container inspection
docker inspect neorl-industrial-dev
docker-compose exec neorl-dev env
```

## Production Best Practices

### 1. Security

- Use Docker secrets for sensitive data
- Enable TLS for all external connections
- Regular security updates
- Non-root container execution
- Network segmentation

### 2. Monitoring

- Implement comprehensive health checks
- Set up log aggregation
- Monitor resource usage
- Alert on service failures

### 3. Backup Strategy

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)

# Database backup
docker-compose exec postgres pg_dump -U mlflow mlflow > "backup_db_${DATE}.sql"

# Volume backup
docker run --rm -v neorl-mlflow-data:/data -v $(pwd):/backup alpine tar czf "/backup/mlflow_${DATE}.tar.gz" /data

# Model artifacts backup
docker run --rm -v neorl-model-artifacts:/data -v $(pwd):/backup alpine tar czf "/backup/models_${DATE}.tar.gz" /data
```

### 4. Performance Optimization

- Use appropriate resource limits
- Enable JIT compilation
- Optimize batch sizes
- Monitor GPU utilization
- Cache expensive operations

## Integration with CI/CD

### Docker Registry

```bash
# Tag for registry
docker tag neorl-prod ghcr.io/terragon-labs/neorl-industrial-gym:latest

# Push to registry
docker push ghcr.io/terragon-labs/neorl-industrial-gym:latest

# Pull in production
docker pull ghcr.io/terragon-labs/neorl-industrial-gym:latest
```

### Automated Deployment

```yaml
# Example deployment pipeline
deploy:
  script:
    - docker pull $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - docker-compose up -d --no-deps neorl-prod
    - docker-compose exec -T neorl-prod python -c "import neorl_industrial; print('Health check OK')"
```

---

For additional support or questions about Docker deployment, please refer to the main documentation or create an issue in the GitHub repository.