# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the neoRL-industrial-gym project. ADRs document important architectural decisions made during the development of this industrial reinforcement learning benchmark suite.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences. ADRs help teams:

- Document the reasoning behind architectural choices
- Provide context for future developers
- Track the evolution of architectural decisions
- Enable informed decision-making when revisiting choices

## ADR Template

When creating a new ADR, use the following template:

```markdown
# ADR-XXXX: [Title]

## Status

[Proposed | Accepted | Deprecated | Superseded]

## Context

[What is the issue that we're seeing that is motivating this decision or change?]

## Decision

[What is the change that we're proposing and/or doing?]

## Rationale

[Why are we making this decision? What are the driving factors?]

## Consequences

### Positive
- [What becomes easier or better after this change?]

### Negative
- [What becomes more difficult or worse after this change?]

### Neutral
- [What are the other implications or considerations?]

## Alternatives Considered

[What other options were evaluated?]

## References

[Links to relevant resources, discussions, or documentation]
```

## Current ADRs

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](0001-technology-stack-selection.md) | Technology Stack Selection | Accepted | 2025-01-15 |
| [0002](0002-safety-first-design.md) | Safety-First Design Principles | Accepted | 2025-01-15 |
| [0003](0003-jax-implementation.md) | JAX Implementation Strategy | Accepted | 2025-01-15 |

## Guidelines for Writing ADRs

1. **Be Concise**: ADRs should be brief but comprehensive
2. **Focus on Decisions**: Document decisions, not implementations
3. **Include Context**: Explain the problem being solved
4. **Consider Alternatives**: Show what options were evaluated
5. **Document Consequences**: Be honest about trade-offs
6. **Update Status**: Keep the status current as decisions evolve

## Proposing New ADRs

To propose a new ADR:

1. Copy the template above
2. Fill in the sections with your proposal
3. Set status to "Proposed"
4. Submit for review via pull request
5. Update status to "Accepted" once approved

## ADR Lifecycle

- **Proposed**: Initial draft under discussion
- **Accepted**: Decision has been approved and implemented
- **Deprecated**: Decision is no longer recommended but not superseded
- **Superseded**: Decision has been replaced by a newer ADR