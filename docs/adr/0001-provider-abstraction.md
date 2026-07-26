# ADR-0001: One adapter per protocol family, behind a Template Method base

**Status:** Accepted
**Date:** 2026-07-25
**Deciders:** Lead Architect

## Context

EDEN must support OpenAI, Gemini, Claude, Groq, DeepSeek, Together, OpenRouter,
Ollama, local models and unknown future providers, and must let any of them be
replaced without changing business logic.

Two forces pull against each other:

- **Fidelity.** Vendors differ. Anthropic hoists the system prompt to a
  top-level field and requires `max_tokens`; Gemini renames the assistant role
  to `model` and nests text in `parts`; OpenAI-style APIs do neither.
- **Duplication.** Retry, timeout, rate limiting, timing, logging and error
  translation are identical for all of them. Written per vendor, that is ten
  copies of the same subtly-important code, and a bug fixed in one.

A third fact shapes the answer: six of the ten named vendors already implement
the *same* OpenAI chat-completions contract.

## Decision

Two-part structure.

1. **`BaseProvider` is an abstract Template Method.** It owns rate limiting,
   timeout enforcement, retry with backoff, execution timing, structured
   logging, cost accounting and transport-error translation. It delegates
   exactly two operations to subclasses: `_perform_chat` and (optionally)
   `_perform_stream`.
2. **One concrete adapter per *protocol family*, not per vendor.** Vendors
   sharing a wire contract share an adapter and are distinguished purely by
   configuration: `base_url`, `api_key_env`, model catalogue, pricing,
   privacy tier.

`ProviderKind.CUSTOM` plus an `implementation` import path admits third-party
adapters without EDEN's source referencing them.

## Options considered

### Option A: One class per vendor

| Dimension | Assessment |
|---|---|
| Complexity | High — ten classes at launch |
| Fidelity | Excellent |
| Duplication | Severe — retry/timing/logging copied ten times |
| Cost of a new vendor | New file, new tests, new review |

**Pros:** Each vendor's quirks are locally obvious.
**Cons:** The shared safety machinery drifts between copies. A retry bug gets
fixed in three of ten adapters and nobody notices.

### Option B: One universal adapter with per-vendor branching

| Dimension | Assessment |
|---|---|
| Complexity | Deceptively low, then explosive |
| Fidelity | Degrades as vendors diverge |
| Duplication | None |
| Cost of a new vendor | Another branch in a growing conditional |

**Pros:** Minimal file count.
**Cons:** Becomes a `if vendor == ...` switch — the exact coupling this system
exists to prevent. Testing one vendor means loading all of them.

### Option C: Template Method base + one adapter per protocol family *(chosen)*

| Dimension | Assessment |
|---|---|
| Complexity | Moderate — one base, four adapters |
| Fidelity | Excellent where it matters, shared where it doesn't |
| Duplication | None |
| Cost of a new vendor | Usually zero code; a config row |

**Pros:** Six of the eight shipped vendors cost nothing to add. Shared
machinery is improved once for all. Adapters are ~120 lines of pure translation
and are trivially testable.
**Cons:** Requires judging whether a new vendor is "OpenAI-compatible enough".
Inheritance is a tighter coupling than composition would be.

## Trade-off analysis

Option C accepts inheritance — normally worth resisting — in exchange for
eliminating the duplication that Option A guarantees. The coupling is
acceptable because the base class's contract is narrow and stable: subclasses
see only `_perform_chat`, `_perform_stream`, configuration and a logger. They
cannot reach into routing, health or transport internals.

The "OpenAI-compatible enough" judgement is a real cost, but it fails safely: a
vendor that turns out to diverge simply graduates to its own adapter, and the
base class carries over unchanged.

## Consequences

**Easier**
- Adding a vendor speaking a known protocol: a TOML entry.
- Improving retry, timing or redaction: one file, every vendor benefits.
- Testing: `FakeTransport` plus canned JSON, no network anywhere in the suite.

**Harder**
- Vendor-specific features that do not fit the neutral types (server-side
  tool execution, vendor-native caching) need explicit accommodation in
  `eden.core.types` rather than a quick passthrough.
- A base-class change touches every adapter, so the base's contract must be
  treated as public API and versioned accordingly.

**To revisit**
- If more than two adapters need to opt *out* of shared behaviour, inheritance
  is the wrong tool and the base should become a composed pipeline of
  middleware.
- Tool use, embeddings and multimodal content will each require a neutral type
  before they can be added; do not let a vendor shape leak through.

## Action items

1. [x] `BaseProvider` with template-method `chat` / `stream` / `health_check`.
2. [x] `OpenAICompatibleProvider`, `AnthropicProvider`, `GeminiProvider`, `MockProvider`.
3. [x] `ProviderFactory` with registrable builders and a `CUSTOM` import path.
4. [x] Parametrised test asserting identical business logic across five vendor configurations.
5. [ ] Neutral `ToolCall` / `ToolResult` types before any tool-use adapter work.
6. [ ] Neutral embedding request/response types ahead of Phase 2 (memory).
