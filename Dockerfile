FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Set environment variables
ENV PYTHONPATH=/app/src
ENV JAX_PLATFORM_NAME=cpu

# Default command
CMD ["python", "-m", "pytest", "tests/"]

# Labels for metadata
LABEL maintainer="daniel@terragon.ai"
LABEL description="neoRL-industrial-gym: Industrial-grade Offline RL benchmark"
LABEL version="0.1.0"