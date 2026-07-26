# ADR-0003: Five memory kinds are one data model with five retention policies

**Status:** Accepted
**Date:** 2026-07-25
**Deciders:** Lead Architect

## Context

EDEN must implement short-term, long-term, vector, conversation and project
memory. The naming invites a natural but expensive reading: five subsystems,
five schemas, five APIs, and an agent that must know which one to interrogate.

Examining what each actually needs to hold reveals something else. Every one of
them stores *text, plus when it was created, plus who it belongs to, plus how
much it matters*. What genuinely differs is:

- **where it persists** — process memory, a file, a database
- **how it forgets** — capacity, age, token budget, or never

That is a retention policy, not a data model.

A second force: agents will ask "what do I know about X?" far more often than
"what does my vector store contain?". Any design that forces the caller to pick
a store first has pushed a routing decision onto every call site.

## Decision

**One record, one contract, five policies.**

- `MemoryRecord` is the sole record type across all five kinds. `MemoryKind` is
  a field on it, not a different class.
- `MemoryStore` is the sole contract. `BaseMemoryStore` is a Template Method
  owning filtering, recency and importance blending, ordering, validation,
  logging and timing. A concrete store implements four persistence hooks and
  optionally overrides one scoring hook.
- `MemoryManager` is the entry point. `recall()` fans one query across every
  store concurrently and merges the results into a single ranking.
- Persistence is injected as a `RecordRepository`, so JSONL today and SQLite
  tomorrow is a constructor argument.
- Embedding is injected as an `Embedder`, so vector memory works with a real
  provider, with the offline hash embedder, or with anything a host supplies.

Two stores are specialisations rather than new machinery. `ConversationMemory`
adds token-budget windowing. `ProjectMemory` *composes* a long-term store and
adds a fact layer — because "remember that the deploy command is X" is an
access pattern, not a sixth policy.

## Options considered

### Option A: Five independent subsystems

| Dimension | Assessment |
|---|---|
| Complexity | High — five schemas, five APIs |
| Duplication | Severe — filtering and ranking written five times |
| Cross-store recall | Effectively impossible |
| Cost of a new kind | A new subsystem |

**Pros:** Each kind can be optimised in isolation.
**Cons:** No way to answer "what do I know about X?" without the caller
orchestrating five queries and inventing its own merge. Ranking would diverge
between stores, silently.

### Option B: One store with a `kind` column

| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Duplication | None |
| Retention | **Broken** — one policy for all |
| Cross-store recall | Trivial |

**Pros:** Simplest possible implementation.
**Cons:** Short-term memory that never forgets is a leak; conversation history
that ignores token budgets breaks prompting. The differences between kinds are
real and this design erases them.

### Option C: One model, one contract, five policies *(chosen)*

| Dimension | Assessment |
|---|---|
| Complexity | Moderate — one base, five stores |
| Duplication | None |
| Retention | Correct — each policy is explicit and testable |
| Cross-store recall | Built in, via the manager |

**Pros:** Retention differences are explicit and independently testable. Recall
is one call. Adding a sixth kind is a subclass with four small methods.
**Cons:** The base class must stay general enough for all five, so a store with
genuinely unusual semantics would strain it.

## Trade-off analysis

The decisive question was where retention policy lives. Option B puts it
nowhere; Option A puts it in five places that will drift. Option C makes it the
*only* thing a store defines, which means each policy is a handful of lines that
a test can pin precisely — eviction by importance, expiry by age, trimming by
turn count, windowing by tokens.

Two subsidiary decisions are worth recording.

**Merging requires per-store normalisation.** Scores from a cosine index and
from lexical overlap are not comparable. The manager rescales each store's hits
to `[0, 1]` within that store before applying a per-kind weight. Without this, a
store whose scorer happens to emit larger numbers would win purely by scale.

**Zero relevance is a hard exclusion.** The blending step short-circuits when
relevance is zero, so a record with no lexical overlap — or a cosine below the
configured floor — cannot be resurrected by being recent. This is deliberately
the same filter/rank discipline as ADR-0002: a hard constraint must not be
something a strong enough soft signal can outweigh. A test caught the original
implementation getting this wrong.

## Consequences

**Easier**
- "What do I know about X?" is one call across every store.
- Swapping storage: `JsonlRecordRepository` → any `RecordRepository`.
- Running with no credentials: `HashEmbedder` keeps vector memory functional.
- Testing retention: each policy is isolated and deterministic.

**Harder**
- A store needing genuinely different search semantics must override more of
  the base than the single scoring hook comfortably allows.
- Per-kind merge weights are a tuning surface with no obvious right answer.
  Current defaults are editorial, not measured.

**To revisit**
- Brute-force cosine is exact and dependency-free, and is correct up to roughly
  10⁴ records per namespace. Beyond that, inject a real index behind
  `VectorIndex` — the seam already exists.
- `JsonlRecordRepository` reads the whole namespace file per query. Acceptable
  at current scale; move to SQLite when a namespace exceeds a few megabytes.
- Memory has no consolidation or summarisation. Long-running agents will need
  it, and it belongs in the manager, not in a store.
- Per-kind merge weights should be moved into `MemoryConfig` once there is
  evidence about what they ought to be.

## Action items

1. [x] `MemoryRecord`, `MemoryQuery`, `SearchHit` and the scoring primitives.
2. [x] `BaseMemoryStore` Template Method plus the `MemoryStore` protocol.
3. [x] Five stores: short-term, long-term, vector, conversation, project.
4. [x] `RecordRepository` with JSONL and in-memory implementations.
5. [x] `Embedder` with gateway-backed and offline implementations.
6. [x] `MemoryManager` with concurrent fan-out, normalised merge and failure isolation.
7. [x] Kernel wiring, started after the gateway and stopped before it.
8. [ ] Move per-kind merge weights into `MemoryConfig` once measured.
9. [ ] Add consolidation/summarisation before Phase 4 agents ship.
