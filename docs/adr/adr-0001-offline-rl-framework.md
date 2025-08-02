# ADR-0001: Offline RL Framework Selection

**Status**: Accepted  
**Date**: 2025-08-02  
**Decision Makers**: Daniel Schmidt (CTO), Technical Team

## Context

The neoRL-industrial-gym project requires a robust foundation for implementing offline reinforcement learning algorithms suitable for industrial control systems. The choice of framework significantly impacts performance, safety guarantees, research compatibility, and long-term maintainability.

Key requirements:
- High-performance computing for large-scale simulations
- Safety-critical algorithm implementations
- Research community compatibility (D4RL standards)
- Industrial deployment readiness
- Extensibility for custom algorithms

## Decision

We will build on the NeoRL-2 framework foundation while implementing a custom architecture optimized for industrial applications, using JAX for acceleration and incorporating safety-first design principles.

## Rationale

1. **NeoRL-2 Foundation**: Provides proven algorithm implementations and research community acceptance
2. **Industrial Focus**: Custom safety constraints and monitoring systems not available in general-purpose frameworks
3. **Performance Requirements**: Industrial applications demand real-time inference and large-scale batch training
4. **Safety Guarantees**: Need formal verification capabilities for safety-critical deployments
5. **Ecosystem Integration**: Must work with existing industrial protocols and MLflow tracking

## Consequences

### Positive
- Complete control over safety-critical components
- Optimized performance for industrial use cases
- Research community compatibility through D4RL standards
- Flexible architecture for custom environments and algorithms
- Strong theoretical foundations from NeoRL-2

### Negative
- Increased development effort compared to using existing frameworks
- Need to maintain custom codebase alongside upstream changes
- Potential compatibility issues with new research developments
- Higher initial learning curve for contributors

### Neutral
- Framework choice impacts all future technical decisions
- Requires dedicated team expertise in both RL and industrial systems

## Alternatives Considered

### Option 1: Pure D4RL + Stable-Baselines3
- **Description**: Use existing D4RL datasets with Stable-Baselines3 implementations
- **Pros**: Minimal development effort, broad community support, proven algorithms
- **Cons**: No industrial safety features, limited performance optimization, not designed for production
- **Rejected because**: Lacks safety guarantees and industrial-specific features

### Option 2: Ray RLlib
- **Description**: Build on Ray's distributed RL platform
- **Pros**: Excellent scalability, industry adoption, comprehensive algorithm suite
- **Cons**: Complex deployment, limited safety framework, focuses on online RL
- **Rejected because**: Not optimized for offline RL or safety-critical applications

### Option 3: Acme (DeepMind)
- **Description**: Use DeepMind's agent framework with JAX backend
- **Pros**: JAX-native, research-grade implementations, modular design
- **Cons**: Limited industrial features, academic focus, complex architecture
- **Rejected because**: Requires significant modification for industrial safety requirements

## Implementation Notes

1. **Migration Strategy**: Implement algorithms incrementally, starting with CQL and IQL
2. **Safety Integration**: Design safety constraints as first-class citizens in the architecture
3. **Performance Optimization**: Use JAX JIT compilation and vectorization throughout
4. **Testing Framework**: Comprehensive validation against D4RL benchmarks and industrial datasets
5. **Documentation**: Maintain compatibility matrix with research frameworks

## References

- [NeoRL-2 Paper](https://arxiv.org/abs/2302.00665)
- [D4RL: Datasets for Deep Data-Driven RL](https://arxiv.org/abs/2004.07219)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Industrial RL Safety Standards](https://www.isa.org/standards-and-publications/)

---

*This ADR establishes the foundational framework decision that influences all subsequent architectural choices in the project.*