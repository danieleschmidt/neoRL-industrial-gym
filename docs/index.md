# neoRL-industrial-gym Documentation

Welcome to the documentation for **neoRL-industrial-gym**, the first industrial-grade Offline RL benchmark & library built on NeoRL-2 plus real factory traces.

## Overview

neoRL-industrial-gym bridges the gap between academic offline RL research and real-world industrial control systems. Building on NeoRL-2's foundation, we provide production-ready benchmarks using actual factory PID/PLC loops, with a focus on safety-critical applications where online exploration is prohibitively expensive.

## Quick Links

- [Getting Started](tutorials/01_getting_started.md)
- [API Reference](api/index.md)
- [Safety Constraints](tutorials/02_safety_constraints.md)
- [Deployment Guide](tutorials/03_deployment.md)

## Key Features

- **7 Industrial Simulators**: Real-world PID/PLC control loops from manufacturing environments
- **D4RL-Style Datasets**: Standardized offline trajectories with varying quality levels
- **JAX-Accelerated Agents**: High-performance implementations via Optree
- **Safety Monitoring**: Real-time violation tracking and constraint satisfaction metrics
- **MLflow Integration**: Comprehensive experiment tracking and visualization

## Industrial Safety Notice

⚠️ **CRITICAL**: This software is intended for research and simulation only. Always validate policies extensively before deploying to real industrial systems.