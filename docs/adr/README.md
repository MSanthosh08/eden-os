# Architecture Decision Records

Each ADR captures one decision, the options rejected, and — most importantly —
what would make us revisit it. They are append-only: a superseded ADR is marked
`Superseded by ADR-NNNN` rather than edited.

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-provider-abstraction.md) | One adapter per protocol family, behind a Template Method base | Accepted |
| [0002](0002-routing-strategy.md) | Routing separates hard filters from weighted preferences | Accepted |
| [0003](0003-memory-architecture.md) | Five memory kinds are one data model with five retention policies | Accepted |
| [0004](0004-execution-pipeline.md) | Actions are inert data, and reversibility is proven before permission | Accepted |
| [0005](0005-agent-architecture.md) | The five-method agent contract, enforced by a template lifecycle | Accepted |
| [0006](0006-phase5-surfaces.md) | Hardware actuation is an execution action; automation is data; the interface is loopback-only | Accepted |

Format follows the template in the `engineering:architecture` skill.
