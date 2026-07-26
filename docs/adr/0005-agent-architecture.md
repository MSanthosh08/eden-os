# ADR-0005: The five-method agent contract, enforced by a template lifecycle

**Status:** Accepted
**Date:** 2026-07-26
**Deciders:** Lead Architect

## Context

The requirement was explicit: *agents must inherit from `BaseAgent`; every agent
implements `can_handle()`, `plan()`, `execute()`, `verify()`, `report()`; no
exceptions.*

Two facts about the setting shape how that is best honoured.

**Agents are the component most likely to be written by somebody else.** A
provider adapter is written once per vendor; agents are written continuously, by
people extending EDEN for their own domain. Whatever the base class makes easy
is what will exist in six months.

**An agent is the component we trust least.** Its behaviour is steered by model
output, which is steered by text an attacker may have written. Everything Phases
1–3 built — provider routing, memory isolation, the execution pipeline — is
load-bearing precisely at this boundary.

There is also a direct tension inside the specification itself. "Every agent
implements all five, no exceptions" pulls one way; rule 1, *never duplicate
code*, pulls the other. `execute` is the same loop in every agent. `report` is
the same assembly. Making both abstract means every agent ever written
reimplements them, slightly differently, with the security-relevant parts
drifting.

## Decision

**All five methods exist on every agent, and `run()` calls all five, in order,
every time.** The lifecycle is a template method the agent does not control:

```
can_handle → plan → execute → verify → report
```

An agent cannot skip verification, because it never chooses the sequence.

**Only `can_handle` and `plan` are abstract.** `execute`, `verify` and `report`
ship with complete, overridable implementations. This is a deliberate reading of
"no exceptions": the contract that matters is *all five are present and all five
run*, not *all five are retyped in every subclass*.

**`can_handle` returns a score, not a boolean.** `Suitability` carries a
confidence in `[0, 1]` plus a reason. The orchestrator ranks willing agents, so
a specialist outranks a generalist by scoring higher rather than by being named
in a conditional — the same discipline as ADR-0002 and ADR-0003.

**Plans are inert data.** A `PlanStep` either carries an `Action` — which goes
through the Phase 3 pipeline — or a prompt. There is no third possibility, which
keeps the set of things an agent can do enumerable and reviewable.

**`AgentContext` is the entire capability surface.** An agent receives it and has
no other route outward. It exposes `think`, `recall`, `remember`, `observe`,
`act`, `preview`, `undo` — and no handler, no engine, no `submit`. A test asserts
the absence.

**Agent verification is distinct from execution verification.** Execution asks
*may this action run?* before the fact. The agent asks *did the work achieve the
goal?* after it, and may require its own completed work to be compensated.

## Options considered

### Option A: All five abstract, per the literal wording

| Dimension | Assessment |
|---|---|
| Spec fidelity | Literal |
| Duplication | **Severe** — the step loop in every agent |
| Safety | Poor — each agent reimplements compensation |
| Third-party authoring | Hostile: ~200 lines before anything works |

**Pros:** Nobody can misread the requirement.
**Cons:** The execute loop contains failure handling, step skipping and rollback
triggering. Duplicated per agent, those diverge, and the divergence is invisible
until something needs undoing.

### Option B: A protocol with no base class, composed helpers

| Dimension | Assessment |
|---|---|
| Spec fidelity | Violates "must inherit from `BaseAgent`" |
| Duplication | Low |
| Safety | **Poor** — nothing guarantees `verify` runs |
| Flexibility | High |

**Pros:** Composition over inheritance, easier to test in pieces.
**Cons:** Nothing forces the sequence. An agent could call `execute` and skip
`verify`, which is exactly the failure the requirement exists to prevent.

### Option C: Template lifecycle, two abstract methods *(chosen)*

| Dimension | Assessment |
|---|---|
| Spec fidelity | Honours the intent; deviates on abstractness |
| Duplication | None |
| Safety | Good — the sequence is not the agent's to choose |
| Third-party authoring | Two methods to a working agent |

**Pros:** All five present, all five run, none duplicated. A new agent is
`can_handle` plus `plan`. Improvements to compensation benefit every agent.
**Cons:** A reader checking the requirement literally will find three concrete
methods and must read this ADR to see why. Recorded here for exactly that
reason.

## Trade-off analysis

The deciding question was which failure is worse: an agent author who has to read
an ADR, or an agent author who reimplements rollback triggering incorrectly. The
second is a silent correctness bug in the code path that runs when things have
already gone wrong — the worst place to have one.

Two subsidiary decisions.

**`AgentContext.act` prepares twice.** It calls `engine.review()` to retain undo
state, then `engine.submit()`, which prepares again internally. Preparation is
read-only by contract, so this is safe, but it doubles the pre-action I/O. The
alternative — caching preparations inside the engine keyed by action id — adds
mutable state to the component with the strongest correctness requirements.
Duplicated reads were the cheaper price.

**`dispatch_all` is sequential, not concurrent.** Agents share one workspace and
one memory namespace. Running them in parallel would let two agents race on the
same file with no coordination, and the execution layer has no locking yet
(ADR-0004, action item 8). Concurrency belongs after locking, not before it.

## Consequences

**Easier**
- Writing an agent: two methods.
- Reviewing what an agent intends: `Plan.describe()` before anything runs.
- Auditing: every effect an agent causes is already in the execution journal.
- Read-only deployments: with execution disabled, agents still run and
  `FileTaskAgent` simply declines — a valid configuration, not a broken one.

**Harder**
- An agent needing a genuinely different lifecycle must override `run`, which is
  possible but loses the guarantee. That should be treated as a design smell.
- Multi-agent collaboration has no representation. One task, one agent.
- Re-planning after failure is not supported: a plan is produced once. An agent
  wanting to adapt mid-task must express that as steps, not as a loop.

**To revisit**
- **Re-planning.** The single most likely extension. It belongs in `run` as a
  bounded loop, with the iteration ceiling in `AgentConfig`.
- **Multi-agent delegation.** An agent should be able to raise a sub-task through
  the orchestrator; the natural shape is a `PlanStep` carrying a `Task`, with a
  depth limit to prevent recursion.
- **Plans are sequential.** A dependency graph would allow independent steps to
  run in parallel, but only once the execution layer can lock.
- **`FileTaskAgent` routes by keyword.** Adequate for a demonstration, brittle in
  general. Model-based routing is the obvious upgrade and needs a confidence
  floor so it degrades to the generalist rather than guessing.

## Action items

1. [x] `Task`, `Suitability`, `Plan`, `PlanStep`, `StepOutcome`, `Verification`, `AgentReport`.
2. [x] `BaseAgent` with a template `run()` invoking all five methods in order.
3. [x] `AgentContext` as the sole capability surface, with no route around the pipeline.
4. [x] Three built-in agents spanning read-only, effectful and diagnostic.
5. [x] `AgentOrchestrator` with score-based routing and explained declines.
6. [x] Memory consolidation, closing the ADR-0003 carry-over.
7. [x] Test asserting an agent cannot reach an execution handler.
8. [ ] Bounded re-planning loop with an iteration ceiling in `AgentConfig`.
9. [ ] Sub-task delegation with a recursion depth limit.
10. [ ] Replace keyword routing in `FileTaskAgent` with model-based classification.
