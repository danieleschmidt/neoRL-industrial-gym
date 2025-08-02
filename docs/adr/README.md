# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records (ADRs) for the neoRL-industrial-gym project.

## What are ADRs?

Architecture Decision Records (ADRs) are short text documents that capture important architectural decisions made during the development of the project, along with their context and consequences.

## ADR Format

Each ADR follows this structure:

```markdown
# ADR-XXXX: [Title]

**Status**: [Proposed | Accepted | Deprecated | Superseded by ADR-YYYY]
**Date**: YYYY-MM-DD
**Decision Makers**: [List of people involved]

## Context

[Description of the issue/problem and its context]

## Decision

[The decision that was made]

## Rationale

[Why this decision was made]

## Consequences

[Results of the decision, both positive and negative]

## Alternatives Considered

[Other options that were evaluated]
```

## Current ADRs

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-0001](./adr-0001-offline-rl-framework.md) | Offline RL Framework Selection | Accepted | 2025-08-02 |
| [ADR-0002](./adr-0002-jax-acceleration.md) | JAX for High-Performance Computing | Accepted | 2025-08-02 |
| [ADR-0003](./adr-0003-safety-constraint-architecture.md) | Safety Constraint Architecture | Accepted | 2025-08-02 |

## Creating New ADRs

1. Copy the template: `cp adr-template.md adr-XXXX-descriptive-title.md`
2. Fill in the sections with relevant information
3. Update the table above with the new ADR
4. Create a pull request for review