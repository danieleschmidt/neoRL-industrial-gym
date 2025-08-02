# Development Container for neoRL-industrial-gym
FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    graphviz \
    graphviz-dev \
    libffi-dev \
    libssl-dev \
    pkg-config \
    software-properties-common \
    sudo \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to non-root user
USER $USERNAME

# Initialize conda for the user
RUN conda init bash

# Create conda environment with Python 3.10
RUN conda create -n neorl python=3.10 -y && \
    echo "conda activate neorl" >> ~/.bashrc

# Install JAX (CPU version by default)
SHELL ["/bin/bash", "-c"]
RUN source ~/.bashrc && conda activate neorl && \
    pip install --upgrade pip && \
    pip install "jax[cpu]>=0.4.0" jaxlib>=0.4.0

# Install development tools
RUN source ~/.bashrc && conda activate neorl && \
    pip install \
    black \
    isort \
    ruff \
    mypy \
    pytest \
    pytest-cov \
    pytest-xdist \
    pre-commit \
    ipython \
    jupyter \
    notebook \
    jupyterlab

# Install additional development utilities
RUN source ~/.bashrc && conda activate neorl && \
    pip install \
    mlflow \
    wandb \
    tensorboard \
    plotly \
    seaborn \
    pandas \
    numpy \
    scipy \
    matplotlib

# Set working directory
WORKDIR /workspace

# Copy environment template and create .env file
COPY .env.template .env.template

# Make sure conda environment is activated by default
RUN echo 'if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then source "/opt/conda/etc/profile.d/conda.sh"; conda activate neorl; fi' >> ~/.bashrc

# Install git hooks helper
RUN pip install pre-commit

# Expose common development ports
EXPOSE 8000 8080 5000 6006 8888

# Set default command
CMD ["/bin/bash"]