# ADR-0003: JAX Implementation Strategy

## Status

Accepted

## Context

Having selected JAX as our primary deep learning framework (ADR-0001), we need to define a comprehensive implementation strategy that maximizes JAX's benefits while addressing its unique challenges in the context of industrial offline reinforcement learning.

Key implementation considerations include:

1. **Functional Programming Paradigm**: JAX requires pure functions and immutable data structures
2. **JIT Compilation**: Optimal performance requires careful design for XLA compilation
3. **Vectorization**: Efficient computation through vmap and other JAX transformations
4. **Memory Management**: JAX's lazy evaluation and XLA compilation affect memory usage patterns
5. **Debugging and Development**: Functional programming requires different debugging approaches
6. **Integration**: Seamless integration with existing Python scientific computing ecosystem

The implementation must balance JAX's performance benefits with development productivity and maintainability for a complex industrial RL system.

## Decision

We adopt a layered JAX implementation strategy:

### Core Implementation Layers

1. **Pure JAX Core**: Low-level numerical computations using pure JAX
2. **Functional Abstractions**: Higher-level functional abstractions for RL algorithms
3. **Stateful Wrappers**: Thin stateful wrappers for ease of use and integration
4. **Safety Integration**: JAX-aware safety monitoring and constraint validation

### Key Implementation Patterns

- **Functional State Management**: Use PyTrees for all stateful components
- **JIT-First Design**: Design all core functions for JIT compilation from the start
- **Vectorized Operations**: Leverage vmap for parallel environment execution
- **Immutable Data Flows**: Pure functional data transformations throughout
- **Type Safety**: Comprehensive typing with JAX-compatible type annotations

## Rationale

### Performance Optimization Strategy

JAX's performance advantages require specific implementation patterns:

```python
# JIT-compiled policy execution
@jax.jit
def policy_step(params: PolicyParams, state: EnvState, key: PRNGKey) -> Tuple[Action, PolicyState]:
    """JIT-compiled policy step for maximum performance."""
    pass

# Vectorized environment execution
@jax.vmap
def vectorized_env_step(env_states: EnvStates, actions: Actions) -> EnvStates:
    """Parallel environment execution across batch dimension."""
    pass

# Efficient batch training
@jax.jit
def training_step(optimizer_state: OptState, batch: Batch) -> Tuple[OptState, Metrics]:
    """JIT-compiled training step with automatic differentiation."""
    pass
```

### Functional Programming Benefits

The functional approach provides several advantages for industrial RL:

1. **Reproducibility**: Pure functions eliminate hidden state dependencies
2. **Testing**: Easier unit testing with predictable input/output relationships
3. **Parallelization**: Safe parallel execution without race conditions
4. **Debugging**: Deterministic behavior simplifies debugging
5. **Safety**: Immutable data structures prevent accidental state corruption

### Integration with Safety Systems

JAX implementation must seamlessly integrate with safety-critical requirements:

```python
@jax.jit
def safe_policy_execution(
    policy_params: PolicyParams,
    state: EnvState,
    safety_constraints: SafetyConstraints,
    key: PRNGKey
) -> Tuple[Action, SafetyReport]:
    """Policy execution with integrated safety validation."""
    
    # Get policy action
    action = policy_forward(policy_params, state, key)
    
    # Validate safety constraints (JIT-compiled)
    safety_report = validate_constraints(state, action, safety_constraints)
    
    # Apply safety override if necessary
    safe_action = jax.lax.cond(
        safety_report.is_safe,
        lambda: action,
        lambda: emergency_action(state)
    )
    
    return safe_action, safety_report
```

## Consequences

### Positive

- **Performance**: 2-3x speedup compared to PyTorch implementations
- **Reproducibility**: Deterministic behavior across different hardware
- **Scalability**: Efficient scaling to multi-GPU/TPU environments
- **Memory Efficiency**: XLA optimizations reduce memory footprint
- **Parallelization**: Native support for vectorized operations
- **Safety**: Functional programming reduces potential for bugs

### Negative

- **Learning Curve**: Team requires training on functional programming concepts
- **Debugging Complexity**: JIT compilation can make debugging more challenging
- **Development Speed**: Initial development may be slower due to functional constraints
- **Library Ecosystem**: Smaller ecosystem compared to PyTorch
- **Memory Spikes**: XLA compilation can cause temporary memory spikes

### Neutral

- **Code Style**: Requires consistent functional programming patterns
- **Testing Strategy**: Different testing approaches for functional code
- **Documentation**: Need for comprehensive JAX-specific documentation
- **Performance Profiling**: Different profiling tools and techniques

## Implementation Architecture

### Core Module Structure

```
src/neorl_industrial/
├── core/                    # Pure JAX implementations
│   ├── agents/             # RL algorithm implementations
│   ├── environments/       # Environment dynamics
│   ├── safety/            # Safety constraint validation
│   └── utils/             # JAX utility functions
├── functional/             # Functional abstractions
│   ├── state_management/  # PyTree state handling
│   ├── transformations/   # JAX transformations (vmap, pmap)
│   └── optimizers/        # Custom JAX optimizers
├── wrappers/              # Stateful API wrappers
│   ├── agents/           # Easy-to-use agent interfaces
│   ├── environments/     # Gymnasium-compatible wrappers
│   └── training/         # Training loop abstractions
└── integration/          # External integrations
    ├── mlflow/          # MLflow tracking integration
    ├── safety/          # Safety system integration
    └── deployment/      # Production deployment tools
```

### Data Structure Design

All data structures use JAX PyTrees for efficient transformation:

```python
@dataclass
class PolicyState:
    """Immutable policy state using JAX PyTree."""
    params: chex.ArrayTree
    optimizer_state: optax.OptState
    step_count: int
    metrics: Dict[str, float]

@dataclass
class EnvironmentState:
    """Immutable environment state."""
    observation: chex.Array
    reward: float
    done: bool
    info: Dict[str, Any]
    safety_metrics: SafetyMetrics
```

### Performance Optimization Patterns

1. **Compilation Strategy**: Compile all performance-critical functions
2. **Memory Management**: Use JAX's memory allocation strategies
3. **Batch Processing**: Vectorize operations across environment/batch dimensions
4. **Gradient Computation**: Efficient automatic differentiation patterns

```python
# Efficient batch gradient computation
@jax.jit
def compute_gradients(params: Params, batch: Batch) -> Tuple[Grads, Metrics]:
    """Compute gradients with automatic vectorization."""
    
    def loss_fn(params: Params, sample: Sample) -> Tuple[float, Metrics]:
        """Per-sample loss computation."""
        pass
    
    # Vectorize loss computation across batch
    vectorized_loss = jax.vmap(loss_fn, in_axes=(None, 0))
    losses, metrics = vectorized_loss(params, batch)
    
    # Compute gradients of mean loss
    mean_loss = jnp.mean(losses)
    gradients = jax.grad(lambda p: jnp.mean(vectorized_loss(p, batch)[0]))(params)
    
    return gradients, metrics
```

## Development Guidelines

### Code Organization Principles

1. **Pure Core**: Keep all core algorithms as pure functions
2. **Immutable Data**: Use immutable data structures throughout
3. **Type Safety**: Comprehensive type annotations with chex
4. **JIT Boundaries**: Clearly define JIT compilation boundaries
5. **Testing**: Separate unit tests for pure functions and integration tests

### Performance Best Practices

1. **JIT Everything**: Compile all performance-critical paths
2. **Avoid Python Loops**: Use JAX control flow primitives
3. **Vectorize Operations**: Use vmap instead of explicit loops
4. **Memory Efficiency**: Avoid unnecessary data copying
5. **Profile Regularly**: Use JAX profiling tools to identify bottlenecks

### Safety Integration

1. **Compile Safety Checks**: Include safety validation in JIT functions
2. **Immutable Constraints**: Safety constraints as immutable data structures
3. **Functional Validation**: Pure functions for constraint checking
4. **Error Handling**: JAX-compatible error handling patterns

## Alternatives Considered

### Object-Oriented JAX Wrapper
- **Pros**: Familiar programming model, easier migration from PyTorch
- **Cons**: Loses JAX performance benefits, harder to reason about state
- **Decision**: Functional approach maximizes JAX advantages

### Hybrid JAX/PyTorch Implementation
- **Pros**: Use JAX for compute, PyTorch for convenience
- **Cons**: Integration complexity, data transfer overhead
- **Decision**: Pure JAX provides better performance and consistency

### Minimal JAX Usage (NumPy-like)
- **Pros**: Easier adoption, familiar numerical computing patterns
- **Cons**: Misses JAX's key benefits (JIT, vmap, grad)
- **Decision**: Full JAX adoption necessary for performance goals

### External State Management
- **Pros**: Cleaner separation of pure and stateful code
- **Cons**: Integration complexity, potential performance overhead
- **Decision**: JAX PyTrees provide best balance of performance and usability

## References

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX Performance Tips](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [Functional Programming for RL](https://arxiv.org/abs/2010.09163)
- [JAX RL Implementations](https://github.com/deepmind/acme)
- [Industrial JAX Deployment](https://blog.tensorflow.org/2021/12/accelerated-linear-algebra-xla-in-production.html)