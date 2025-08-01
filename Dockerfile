# Multi-stage build for neoRL-industrial-gym
FROM python:3.10-slim as base

# Security: Run as non-root user
RUN groupadd -r neorl && useradd -r -g neorl neorl

# Install system dependencies with security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create app directory
WORKDIR /app

# Security: Set proper permissions
RUN chown -R neorl:neorl /app

# ================================
# Development stage
# ================================
FROM base as development

# Install development dependencies
COPY requirements-dev.txt requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements-dev.txt \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=neorl:neorl . .

# Install package in development mode
RUN pip install -e .[dev,test,docs]

# Switch to non-root user
USER neorl

# Environment variables
ENV PYTHONPATH=/app/src \
    JAX_PLATFORM_NAME=cpu \
    NEORL_DEV_MODE=true \
    PYTHONBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import neorl_industrial; print('OK')" || exit 1

# Default command for development
CMD ["python", "-m", "pytest", "tests/", "-v"]

# ================================
# Production stage
# ================================
FROM base as production

# Install only production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# Copy only necessary files for production
COPY --chown=neorl:neorl src/ ./src/
COPY --chown=neorl:neorl pyproject.toml ./
COPY --chown=neorl:neorl README.md ./
COPY --chown=neorl:neorl LICENSE ./

# Install package
RUN pip install --no-cache-dir . \
    && pip cache purge

# Switch to non-root user
USER neorl

# Environment variables for production
ENV PYTHONPATH=/app/src \
    JAX_PLATFORM_NAME=cpu \
    NEORL_DEV_MODE=false \
    PYTHONBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import neorl_industrial; print('OK')" || exit 1

# Default command for production
CMD ["python", "-c", "import neorl_industrial; print('neoRL-industrial-gym ready')"]

# ================================
# GPU stage
# ================================
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as gpu

# Install Python 3.10
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    build-essential \
    git \
    curl \
    ca-certificates \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py | python3.10

# Create non-root user
RUN groupadd -r neorl && useradd -r -g neorl neorl

# Create app directory
WORKDIR /app
RUN chown -R neorl:neorl /app

# Install JAX with CUDA support
RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python3.10 -m pip install --no-cache-dir "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Copy and install requirements
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY --chown=neorl:neorl src/ ./src/
COPY --chown=neorl:neorl pyproject.toml README.md LICENSE ./

# Install package
RUN python3.10 -m pip install --no-cache-dir . \
    && python3.10 -m pip cache purge

# Switch to non-root user
USER neorl

# Environment variables for GPU
ENV PYTHONPATH=/app/src \
    JAX_PLATFORM_NAME=gpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
    CUDA_VISIBLE_DEVICES=0 \
    PYTHONBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check with GPU validation
HEALTHCHECK --interval=30s --timeout=15s --start-period=10s --retries=3 \
    CMD python3.10 -c "import jax; print(f'JAX devices: {jax.devices()}'); import neorl_industrial; print('GPU setup OK')" || exit 1

# Default command for GPU
CMD ["python3.10", "-c", "import jax; print(f'JAX devices: {jax.devices()}'); import neorl_industrial; print('neoRL-industrial-gym GPU ready')"]

# ================================
# Labels for metadata
# ================================
LABEL maintainer="daniel@terragon.ai" \
      description="neoRL-industrial-gym: Industrial-grade Offline RL benchmark" \
      version="0.1.0" \
      org.opencontainers.image.title="neoRL-industrial-gym" \
      org.opencontainers.image.description="First industrial-grade Offline RL benchmark & library" \
      org.opencontainers.image.url="https://github.com/terragon-labs/neoRL-industrial-gym" \
      org.opencontainers.image.source="https://github.com/terragon-labs/neoRL-industrial-gym" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="MIT"