# ADR-0004: Actions are inert data, and reversibility is proven before permission

**Status:** Accepted
**Date:** 2026-07-25
**Deciders:** Lead Architect

## Context

EDEN must be able to change the world — write files, run programs, drive
hardware — while a language model is deciding what those changes should be. The
requirement was stated as *never execute immediately; everything passes through
verification, permission, execution and rollback*.

Two properties of the setting make the naive design unworkable.

**The instruction source is untrusted.** A model's output is influenced by
whatever text it has read, including text an attacker wrote. "Write this file"
must therefore be treated the way a web server treats a POST body, not the way
a program treats its own function calls.

**Undo has to be arranged in advance.** After a file is overwritten, its
previous contents are gone. Any design that decides how to roll back *after*
something fails is deciding too late.

A third, quieter force: an approval step is worthless if the human approving
cannot see what they are approving. That constrains the representation of an
action, not just the pipeline around it.

## Decision

**Actions are data.** `Action` is a frozen dataclass describing an intended
effect. It has no `run`, `execute` or `apply` method — a test asserts this — and
no handler is reachable from outside the module. The only path from intent to
effect is `ExecutionEngine.submit()`.

**Prepare comes before verify.** The pipeline is:

```
prepare  →  verify  →  permit  →  execute  →  (rollback)
```

`prepare` is read-only. Its job is to capture undo state and return a
`RollbackPlan`, or `None` when the effect genuinely cannot be undone.
Reversibility is therefore an *input* to the risk assessment rather than a hope
held afterwards.

**Rollback steps are Actions.** A plan is a tuple of ordinary actions run
through ordinary handlers. There is no second, less-exercised code path for the
operation that runs when things have already gone wrong.

**Irreversible never auto-approves.** Whatever its risk score, an action with no
rollback plan requires confirmation. Risk and reversibility are independent
axes, and the policy ladder treats them that way.

**Absence of an approver is refusal.** The default `ApprovalGate` is
`DenyingGate`. An unattended EDEN cannot perform anything needing confirmation.

**Everything is journalled, including refusals.** A record of what EDEN declined
to do is part of the audit trail. Journal failures are logged but never abort an
action policy already approved.

## Options considered

### Option A: Capability tokens — grant a handler up front, then call it freely

| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Auditability | Poor — one grant covers many effects |
| Approval granularity | Coarse |
| Rollback | Ad hoc, per handler |

**Pros:** Fast, and familiar from OS capability models.
**Cons:** A capability granted for "write files in the workspace" also covers
writing the one file that matters. Approval happens once, far from the effect,
which is precisely the gap an injected instruction exploits.

### Option B: Interceptor chain around direct calls

| Dimension | Assessment |
|---|---|
| Complexity | Moderate |
| Auditability | Good |
| Approval granularity | Good |
| Rollback | **Retrofitted** |

**Pros:** Familiar middleware shape; effects stay ordinary function calls.
**Cons:** The call *is* the effect, so there is no moment at which an action
exists but has not yet happened. Nothing can be shown to a human, serialised for
later, or batched into a reviewable plan. Rollback must reconstruct prior state
after the fact, which for a file overwrite is impossible.

### Option C: Actions as data through a staged pipeline *(chosen)*

| Dimension | Assessment |
|---|---|
| Complexity | Moderate — six modules |
| Auditability | Good — one record per action, refusals included |
| Approval granularity | Per action, with the effect described |
| Rollback | Proven before execution |

**Pros:** A plan can be inspected, logged, approved, replayed or refused before
anything happens. Reversibility is structural. Transactions and dry-run fall out
almost for free.
**Cons:** More ceremony than a function call. Handler authors must write
`prepare` carefully, and a lazy `prepare` that returns `None` degrades an action
to "always needs approval" rather than failing loudly.

## Trade-off analysis

The decisive question was *when* the system commits to being able to undo
something. Options A and B both answer "after", which for destructive
filesystem work means "never". Option C forces the answer to "before", and pays
for it with an extra phase and a stricter handler contract.

Two subsidiary decisions deserve recording.

**Rollback bypasses verification and permission.** Compensating steps were
derived from state captured before the action ran and were described in the
permit the approver saw, so they are already authorised. Re-asking would let an
approver refuse the undo of something they had just approved, stranding the
system mid-change. The trade is that a compromised `prepare` could smuggle a
step into a plan — which is why `prepare` is read-only by contract and why the
plan's description is surfaced to the approver.

**Shell execution ships, deliberately.** Omitting it would not make an AI OS
safer; it would push command execution into whatever the first user wrote
themselves, outside the pipeline. Shipping it inside the pipeline means it is
allowlist-gated (empty by default), never passed through a shell, run with a
scrubbed environment, time-bounded, and — because `prepare` returns no plan —
never auto-approved.

**Over-matching in the path denylist fails safe.** Glob matching uses `fnmatch`,
whose `*` spans separators and so matches more than a strict glob would. For a
denylist that is the correct direction of error. A test caught the first
implementation matching *everything* because the tail of `**/.git/**` is `**`;
the failure mode was a useless subsystem rather than an open door, which is the
asymmetry this choice is designed to produce.

## Consequences

**Easier**
- Showing a human exactly what will happen before it happens.
- Dry-run: verify and permit, skip the effect. Three lines in the engine.
- Transactions: run in order, compensate in reverse.
- Auditing: one journal entry per action, refusals included.
- Phase 4 agents: a model proposes an `Action`; the pipeline disposes. An agent
  cannot bypass it because there is nothing to bypass to.

**Harder**
- Every new effect needs a handler with a genuine `prepare`.
- Streaming or long-running effects do not fit the single-shot shape.
- Two handlers cannot currently share a transaction-wide lock, so a concurrent
  writer can invalidate captured undo state between prepare and execute.

**To revisit**
- **Time-of-check/time-of-use.** Paths are resolved once and reused, but nothing
  holds a lock across prepare and execute. A file changed in between would be
  rolled back to the state captured at prepare, not the state at execute. Needs
  either advisory locking or a content hash checked immediately before writing.
- Binary files are irreversible for `FILE_WRITE`. A content-addressed backup
  store would fix this and is the natural next step.
- Journal entries are unsigned. Tamper-evidence needs a hash chain.
- `deny_above_risk` defaults to `HIGH`, which permits `CRITICAL`-risk actions to
  reach an approver. That is intentional today; production deployments should
  lower it.

## Action items

1. [x] `Action`, `Verdict`, `Permit`, `Preparation`, `RollbackPlan`, `ExecutionRecord`.
2. [x] Seven composable verifiers, including scope, sensitivity and reversibility.
3. [x] `PolicyEngine` with a documented decision ladder; `DenyingGate` as default.
4. [x] Four handlers spanning reversible, irreversible and inert.
5. [x] `ExecutionEngine` with dry-run, review, saga transactions and auto-compensation.
6. [x] Journal with in-memory and JSONL sinks; failures never abort an action.
7. [x] Adversarial tests: traversal, symlink escape, credential names, metacharacters.
8. [ ] Content-hash check between prepare and execute to close the TOCTOU window.
9. [ ] Content-addressed backup store so binary writes become reversible.
10. [ ] Hash-chain the journal for tamper evidence.
