# ADR-0002: Routing separates hard filters from weighted preferences

**Status:** Accepted
**Date:** 2026-07-25
**Deciders:** Lead Architect

## Context

The Omni Router must choose a provider using availability, speed, cost,
capability, privacy, rate limits and health, and must never hardcode that
choice.

The naive reading is "score everything and take the highest". That reading is
wrong, and dangerously so. Consider a request marked
`minimum_privacy_tier = "local_only"` — patient data, unreleased source, a legal
document. If privacy is merely a weighted signal, a sufficiently cheap and fast
public-cloud provider will eventually outscore the compliant local one. The
weights need only be a little off, once, in production.

Some constraints are not preferences. They are conditions of eligibility.

## Decision

Selection runs in two ordered phases with different semantics.

**Phase 1 — Filter (boolean, non-negotiable).** A provider is excluded if any
of these fail, and no score can resurrect it:

- provider disabled in configuration
- `privacy_tier` below the effective floor (the stricter of the global setting
  and the request's own)
- required capabilities not advertised
- requested model unavailable
- circuit breaker open

**Phase 2 — Rank (continuous, weighted).** Survivors are scored by independent
`Scorer` implementations — cost, latency, health, privacy, preference — each
returning a value in `[0, 1]` normalised *within the candidate set*. A
`Strategy` is a weight vector over those scorers.

Every decision returns a `RoutingDecision` carrying the ranked list *with
per-signal breakdowns* and an exclusion reason string per rejected provider.

## Options considered

### Option A: Single weighted score over all signals

| Dimension | Assessment |
|---|---|
| Complexity | Low |
| Safety | **Poor** — compliance is negotiable |
| Explainability | Poor — one opaque number |
| Tunability | Fragile; weights interact non-obviously |

**Pros:** One code path, easy to describe.
**Cons:** Privacy and capability become tunable, which is the same as saying
they are not guarantees. A capability the model does not have cannot be
compensated for by being cheap.

### Option B: Ordered rule chain (first matching rule wins)

| Dimension | Assessment |
|---|---|
| Complexity | Medium, grows with rule count |
| Safety | Good |
| Explainability | Good |
| Tunability | Poor — no notion of "slightly better" |

**Pros:** Deterministic and auditable.
**Cons:** Cannot express trade-offs. "Prefer cheap unless it is much slower" has
no representation; every distinction becomes a new rule.

### Option C: Filter then rank *(chosen)*

| Dimension | Assessment |
|---|---|
| Complexity | Medium — two phases, five scorers |
| Safety | Good — constraints are structurally inviolable |
| Explainability | Good — reasons plus breakdowns |
| Tunability | Good — weights only affect eligible candidates |

**Pros:** Hard requirements cannot be traded away. Soft preferences remain
genuinely continuous. Named strategies (`cheapest`, `fastest`, `privacy_first`)
are preset weight vectors over the same scorers, so a new profile is data, not
a class.
**Cons:** Two mental models instead of one. An operator can be surprised that
raising a weight changed nothing, because their provider was filtered out — the
exclusion reasons exist precisely to answer that.

## Trade-off analysis

Option C costs one extra concept and buys a structural guarantee. The
alternative — trusting that weights are always tuned correctly — is a guarantee
that depends on nobody making a configuration mistake, which is not a guarantee.

Normalising within the candidate set rather than globally was a secondary
decision. It means scores stay meaningful whether a deployment has two
providers or twenty, and whether costs are denominated in cents or thousandths
of a cent, at the price of scores not being comparable between requests. Since
scores are only ever used to order one candidate set, that price is zero.

## Consequences

**Easier**
- Compliance review: "can this request reach a public cloud provider?" is
  answered by reading the filter, not by simulating the weights.
- Debugging: `decision.excluded` names every rejected provider and why.
- New strategies: a weight vector in `build_strategy`.

**Harder**
- Operators must understand that a filtered provider ignores weights entirely.
- Adding a signal means adding a `Scorer` *and* a weight field, in two files.

**To revisit**
- Latency currently uses an EWMA of observed durations with a configured prior.
  Under bursty traffic a percentile (p95) would be a better signal.
- `RoundRobinStrategy` holds a mutable cursor, so it is not safe to share one
  router instance across processes. When EDEN distributes, that cursor must
  move to shared state or the strategy must become stateless.
- Cost scoring uses a four-characters-per-token estimate. Once real tokenisers
  are available per provider, replace the heuristic.

## Action items

1. [x] `Scorer` protocol with cost, latency, health, privacy and preference implementations.
2. [x] `WeightedStrategy` plus named presets; `RoundRobinStrategy`.
3. [x] `OmniRouter.decide()` returning ranked results and exclusion reasons.
4. [x] Test asserting the privacy floor filters rather than de-prioritises.
5. [ ] Replace EWMA latency with a rolling p95 before Phase 5.
6. [ ] Make round-robin state external ahead of distributed execution.
