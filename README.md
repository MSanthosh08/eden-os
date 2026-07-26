# EDEN

A modular AI Operating System.

EDEN's organising idea is a single seam: **business logic never knows which AI
vendor it is talking to.** Agents, memory, automation and the interface layer
speak one neutral vocabulary; a thin adapter translates that vocabulary into
each vendor's dialect. Swapping OpenAI for Ollama is a configuration edit.

**Status:** Complete — all five phases. Configuration, utilities, logging,
exceptions, core kernel, the AI Gateway, memory, the execution pipeline, agents,
hardware, automation and the interface layer are all built, tested and
production-ready. See [Roadmap](#roadmap) for what each phase delivered and
[`docs/adr/`](docs/adr/) for why each decision was made.

---

## Quickstart

```bash
pip install -e ".[dev]"
cp eden.example.toml eden.toml
export OPENAI_API_KEY=...          # credentials live in the environment, never in the file
```

Use it without writing any Python:

```bash
eden status                        # what is on, which providers are healthy
eden ask "explain dependency injection in one line"
eden chat                          # interactive, remembers the conversation
eden task "write release notes" --path notes.md
eden memory "what did we decide about routing"
eden devices                       # the device fleet
eden rules                         # what can fire on a schedule
eden serve                         # web console at http://127.0.0.1:8420
```

Or as a library:

```python
import asyncio
from eden import ChatRequest, EdenKernel, Message, load_config

async def main() -> None:
    async with EdenKernel(load_config()).session() as kernel:
        response = await kernel.gateway.chat(
            ChatRequest(messages=[Message.user("Explain dependency injection in one line.")])
        )
        print(f"{response.provider}/{response.model} in {response.latency_ms:.0f}ms: {response.content}")

asyncio.run(main())
```

Nothing in that snippet names a vendor. The router picks one.

Streaming is the same shape:

```python
async for chunk in await kernel.gateway.stream(request):
    print(chunk.delta, end="", flush=True)
```

---

## Architecture

Layers may import downward and never upward. This is enforced by review, and
violations show up immediately as import cycles.

```
                    ┌───────────────────────────────────────────┐
                    │  interface     CLI · web console · JSON API│
                    │                WebApprovalGate (the human) │
                    └────────────────────┬──────────────────────┘
                    ┌────────────────────▼──────────────────────┐
                    │  automation    Rule = Trigger + payload,   │
                    │                a Task or an Action —       │
                    │                never a callable            │
                    └────────────────────┬──────────────────────┘
                    ┌────────────────────▼──────────────────────┐
                    │  hardware      reads: direct               │
                    │                writes: → execution         │
                    └────────────────────┬──────────────────────┘
                    ┌────────────────────▼──────────────────────┐
                    │  agents        AgentOrchestrator          │
                    │  can_handle → plan → execute → verify     │
                    │                            → report       │
                    │  AgentContext = the whole capability set  │
                    └────────────────────┬──────────────────────┘
                    ┌────────────────────▼──────────────────────┐
                    │  execution     ExecutionEngine            │
                    │  prepare → verify → permit → execute      │
                    │                          └→ rollback      │
                    │  handlers · verifiers · policy · journal  │
                    └────────────────────┬──────────────────────┘
                    ┌────────────────────▼──────────────────────┐
                    │  memory        MemoryManager (fan-out)    │
                    │  ┌──────────┬──────────┬───────────────┐  │
                    │  │short_term│long_term │ vector        │  │
                    │  │conversa- │ project  │ (Embedder ────┼──┼─┐
                    │  │tion      │          │  + VectorIndex)│ │ │
                    │  └──────────┴────┬─────┴───────────────┘  │ │
                    │       RecordRepository (jsonl · memory)   │ │
                    └────────────────────┬──────────────────────┘ │
                                         │  ┌─────────────────────┘
                                         │  speaks only eden.core.types
                    ┌────────────────────▼──────────────────────┐
                    │  gateway                                  │
                    │  ┌─────────────┐  ┌────────────────────┐  │
                    │  │ OmniRouter  │  │  GatewayClient     │  │
                    │  │ filter→rank │──▶  failover, health  │  │
                    │  └─────────────┘  └─────────┬──────────┘  │
                    │       ┌─────────────────────▼──────────┐  │
                    │       │ BaseProvider (Template Method) │  │
                    │       │ retry · timeout · rate limit   │  │
                    │       │ timing · logging · error map   │  │
                    │       └──┬────────┬────────┬────────┬──┘  │
                    │      openai_   anthropic  gemini   mock   │
                    │      compatible                           │
                    └────────────────────┬──────────────────────┘
                    ┌────────────────────▼──────────────────────┐
                    │  transport (HttpTransport protocol)       │
                    └────────────────────┬──────────────────────┘
                    ┌────────────────────▼──────────────────────┐
                    │  core   container · registry · kernel     │
                    │         interfaces · types                │
                    └────────────────────┬──────────────────────┘
                    ┌────────────────────▼──────────────────────┐
                    │  errors · logging · utils · config        │
                    └───────────────────────────────────────────┘
```

### Why the layers sit in this order

| Layer | Owns | Depends on |
|---|---|---|
| `config` | Every tunable, as frozen dataclasses. Enums for all named values. | nothing |
| `errors` | The exception hierarchy, with `code`, `context`, `retryable`. | nothing |
| `utils` | Clock, retry/backoff, rate limiting, dynamic import, redaction. | config, errors |
| `logging` | Handler setup, correlation IDs, redaction filter, timing. | config, utils |
| `core` | DI container, registry, kernel, neutral domain types, protocols. | all of the above |
| `transport` | `HttpTransport` protocol + one httpx implementation. | core |
| `gateway` | Provider adapters, health tracking, Omni Router, client façade. | transport, core |
| `memory` | Five retention policies over one record shape, behind one manager. | gateway, core |
| `execution` | The only path from an intent to an effect on the world. | core, config |
| `agents` | The five-method contract; `AgentContext` is an agent's whole capability surface. | execution, memory, gateway |
| `hardware` | Drivers and safety envelope; actuation routes through execution. | execution, transport |
| `automation` | Triggers and rules; dispatches work, owns no machinery of its own. | agents, execution |
| `interface` | CLI and a dependency-free web console. | all of the above |
| `agents` | Goal-directed work, confined to the capability surface it is given. | all of the above |

---

## Design decisions worth knowing

**One adapter per protocol family, not per vendor.** OpenAI, Groq, DeepSeek,
Together, OpenRouter, Fireworks, vLLM and Ollama all speak the OpenAI
chat-completions contract, so they share `OpenAICompatibleProvider` and differ
only in `base_url`, credential and model catalogue. Anthropic and Gemini have
genuinely different wire shapes and get their own adapters.

**`BaseProvider` is a Template Method.** Retry, timeout, rate limiting, timing,
structured logging and error translation live there once. A vendor adapter
implements two things: build the payload, parse the response. Improving the
shared machinery improves every vendor simultaneously.

**Routing separates hard filters from soft preferences.** Capability coverage,
model availability, the privacy floor and circuit-breaker admission are
*filters* — failing one means ineligible, not merely worse. Cost, latency,
health, privacy and operator preference are *weighted scores*. This is what
stops a cheap, non-compliant provider from ever winning on price.

**Every routing decision is explainable.** `OmniRouter.decide()` returns the
ranked shortlist *with per-signal score breakdowns* and a reason string for each
excluded provider. Routing is never a black box.

**Streaming does not fail over mid-stream.** Once bytes have reached the caller,
silently switching providers would splice two different generations together.
A failure *before* the first chunk does fail over — that is where retrying is
actually safe.

**Credentials are never in configuration.** Config stores `api_key_env`, the
*name* of an environment variable. Values are resolved lazily into `SecretStr`,
whose `str` and `repr` are redacted, and a logging filter scrubs both sensitive
field names and credential-shaped substrings before any record reaches a sink.

**Time is injected.** Retry backoff, the circuit breaker and the rate limiter
all take a `Clock`. The suite runs in 0.3 seconds with zero real sleeping, and
the timing logic is deterministic.

**Five memory kinds, one data model.** Short-term, long-term, vector,
conversation and project memory share `MemoryRecord` and the `MemoryStore`
contract. They differ only in *where* they persist and *how* they forget.
`MemoryManager.recall()` fans one query across all five concurrently, normalises
each store's scores before merging, and drops a store that fails rather than
failing the recall.

**Vector memory works without credentials.** The `Embedder` seam has two
implementations: `GatewayEmbedder` routes through the AI Gateway (inheriting
failover, health and privacy filtering) and falls back to `HashEmbedder` — a
deterministic offline embedder — when no provider is available. Losing the
ability to *store* a memory because a vendor is down is worse than storing it
with a weaker vector.

**Actions are data, not code.** An `Action` describes an intended effect and has
no `run` method — a test asserts this. The only path from intent to effect is
`ExecutionEngine.submit()`, which runs `prepare → verify → permit → execute`.
`prepare` is read-only and comes *first*, because its job is to capture how the
action would be undone before anyone decides whether to allow it. Reversibility
is an input to the risk assessment, not a hope held afterwards.

**Silence is refusal.** The default `ApprovalGate` denies. An unattended EDEN
cannot perform anything that needs confirmation, and an irreversible action
never auto-approves whatever its risk score.

**Agents get a capability surface, not the system.** An agent receives an
`AgentContext` and has no other route outward — no gateway constructor, no
memory store, and crucially no execution handler. `AgentContext.act()` goes
through the Phase 3 pipeline, so "an agent cannot act unsupervised" is
structural rather than a convention. A test asserts the absence.

---

## Memory at a glance

| Kind | Persistence | Forgetting |
|---|---|---|
| `short_term` | volatile, in process | capacity + time-to-live, evicting least *important* first |
| `long_term` | durable, via `RecordRepository` | explicit deletion; refuses writes at the ceiling |
| `vector` | durable + embeddings | explicit deletion |
| `conversation` | volatile or durable | turn limit + token-budget windowing |
| `project` | durable, via `RecordRepository` | explicit deletion; adds a fact layer |

```python
async with EdenKernel(load_config()).session() as kernel:
    await kernel.memory.remember(
        "the deploy command is `make ship`",
        kind=MemoryKind.PROJECT, namespace="eden", importance=0.9,
    )
    await kernel.memory.observe(Message.user("how do I deploy?"), namespace="chat-1")

    hits = await kernel.memory.recall(MemoryQuery(text="deploy", namespace="eden"))
    window = await kernel.memory.conversation.window("chat-1")
    answer = await kernel.gateway.chat(ChatRequest(messages=window))
```

---

## Execution at a glance

```python
kernel = EdenKernel(load_config(), approval_gate=CallbackGate(ask_the_user))
async with kernel.session() as k:
    proposal = Action(                      # inert: describes, does not do
        kind=ActionKind.FILE_WRITE,
        summary="Write the drafted release notes",
        parameters={"path": "notes.md", "content": drafted},
        actor="drafting-agent",
    )
    verdict, _ = await k.execution.review(proposal)   # inspect, nothing happens
    record = await k.execution.submit(proposal)       # the only way to act
```

Policy ladder, in order:

1. A **blocking finding** refuses outright — no threshold or approval overrides it.
2. Risk above `deny_above_risk` refuses outright.
3. **Irreversible** actions never auto-approve, whatever their risk.
4. Risk at or below `auto_approve_max_risk` proceeds automatically.
5. Everything else asks the gate.

Shipped guards: workspace confinement (symlink-resolved), credential-filename
denylist, command allowlist that starts *empty*, no shell invocation ever,
scrubbed child environment, payload and output ceilings, and a journal that
records refusals as well as successes.

---

## Agents at a glance

Every agent has the five methods, and `run()` calls all five in order — an agent
cannot skip verification because it does not control the sequence.

```python
async with EdenKernel(load_config(), approval_gate=gate).session() as k:
    report = await k.agents.dispatch(
        Task(goal="write a release summary", context={"path": "notes.md"})
    )
    print(report.status, report.plan.describe(), report.final_output)
```

| Method | Question it answers | Abstract? |
|---|---|---|
| `can_handle` | Is this task mine, and how sure am I? | yes |
| `plan` | What inert steps would achieve it? | yes |
| `execute` | Run them — effects via the pipeline, thoughts via a model | no |
| `verify` | Did the work achieve the goal? | no |
| `report` | What happened, in full | no |

`can_handle` returns a *score*, so a specialist outranks a generalist by scoring
higher rather than by being named in a conditional. Routing decisions record why
every agent declined.

Agent verification is a different question from execution verification: the
pipeline asks *may this run?* beforehand, the agent asks *did it work?* after —
and may require its own completed work to be rolled back.

Three agents ship: `ConversationAgent` (read-only), `FileTaskAgent` (effectful,
and notably contains no path checks of its own — those live in the pipeline it
cannot influence), and `EchoAgent` (diagnostics).

---

## Hardware, automation and the interface

**Reads are free; writes are actions.** `device.read()` is an ordinary call —
observing a sensor changes nothing. Actuation is only reachable as an `Action`
of kind `device_command` submitted to the execution pipeline, so a servo command
is verified, authorised, journalled and rollback-planned by the same machinery
that guards file writes. Safety limits live in configuration rather than driver
code, and `DeviceSafetyError` is non-retryable by design: a command outside the
envelope does not become safe on a second attempt.

**A rule is data, never a callable.** `Rule` pairs a trigger with exactly one of
a `Task` or an `Action`. It cannot hold a function — otherwise automation could
bypass five phases of safeguards in one lambda — and `eden rules` therefore
lists exactly what is able to fire unattended.

**Silence is refusal.** `WebApprovalGate` is the human that Phase 3's
`ApprovalGate` seam was designed to wait for. A pending action blocks, surfaces
in the console with its risk findings and rollback plan, and times out into a
*refusal*. An interface that cannot reach a human must behave exactly like one
with no human behind it.

```bash
eden serve
```

| Route | Purpose |
|---|---|
| `GET /` | single-page console |
| `GET /api/status` | subsystems, providers, mode |
| `GET /api/journal` | execution audit trail |
| `GET /api/memory?q=` | cross-store recall |
| `GET /api/devices` · `GET /api/device/read` | fleet state and sensor reads |
| `GET /api/rules` · `POST /api/rule/run` | automation |
| `GET /api/approvals` · `POST /api/approve` | the human in the loop |
| `POST /api/chat` · `/api/task` · `/api/action` | gated by `allow_actions` |

> The console has **no authentication** and binds loopback by default. Do not
> expose it to a network. See ADR-0006, action item 8.

All three subsystems default to **off**. Software that can move things, act on a
schedule, or answer HTTP should not do any of those the moment it is installed.

---

## Extending EDEN

### Add a vendor that speaks the OpenAI contract

Configuration only:

```toml
[[gateway.providers]]
name = "fireworks"
kind = "openai_compatible"
base_url = "https://api.fireworks.ai/inference/v1"
api_key_env = "FIREWORKS_API_KEY"
default_model = "accounts/fireworks/models/llama-v3p3-70b-instruct"
```

### Add a vendor with its own wire format

Subclass `BaseProvider`, implement `_perform_chat` (and optionally
`_perform_stream`), then point configuration at it:

```toml
[[gateway.providers]]
name = "in-house"
kind = "custom"
implementation = "my_company.eden_plugins:InHouseProvider"
privacy_tier = "on_premise"
```

Nothing in EDEN's source mentions your package.

### Add a new kind of effect

Subclass `ActionHandler`, implement `prepare` (read-only, returns the rollback
plan) and `execute`, then register it. Verification, permission, journalling,
transactions and compensation apply automatically.

### Write an agent

Subclass `BaseAgent` and implement two methods:

```python
class TriageAgent(BaseAgent):
    @property
    def description(self) -> str:
        return "Sorts incoming issues by severity."

    def can_handle(self, task: Task) -> Suitability:
        return Suitability(score=0.9 if "triage" in task.goal else 0.0)

    async def plan(self, task: Task) -> Plan:
        return Plan(task_id=task.id, steps=(PlanStep(description="...", prompt="..."),))
```

`execute`, `verify` and `report` are inherited complete. Register it with
`kernel.agents.register(TriageAgent(context))`.

### Add a device

Declare it in `eden.toml`. A `simulated` device needs no equipment and is how a
rig is dry-run before it is wired up:

```toml
[[hardware.devices]]
name = "rig"
kind = "simulated"
channels = ["servo", "temp"]

  [hardware.devices.limits]
  servo = [0.0, 90.0]
```

For a protocol EDEN does not speak, subclass `BaseDevice`, implement the four
transport hooks, and point `kind = "custom"` at its import path.

### Swap where memory persists

`LongTermMemory` and `VectorMemory` take a `RecordRepository`. Implement the
five-method protocol against SQLite, Postgres or S3 and inject it — no store
changes.

### Override any wiring from a host application

```python
container = Container()
container.register_instance(GatewayClient, my_own_client)
kernel = EdenKernel(config, container=container)   # EDEN will not overwrite it
```

---

## Development

```bash
ruff check src tests      # 22 rule families, including bandit and docstrings
black src tests           # line length 100
mypy src tests            # --strict, no implicit Any
pytest                    # 436 tests, ~2s, no network
```

All four are required to pass before merge. The suite makes **no network
calls**: provider tests run against an in-memory `FakeTransport` and memory
tests against `InMemoryRecordRepository` and `HashEmbedder`. That is possible
only because both layers depend on protocols rather than concrete backends.

Architecture decisions are recorded in [`docs/adr/`](docs/adr/).

---

## Roadmap

| Phase | Scope | State |
|---|---|---|
| 1 | Config, utils, logging, errors, core, AI Gateway | **Complete** |
| 2 | Memory — five retention policies, embeddings, cross-store recall | **Complete** |
| 3 | Execution — prepare → verify → permit → execute → rollback | **Complete** |
| 4 | Agents — the five-method contract, routing, orchestration | **Complete** |
| 5 | Hardware, automation, interface | **Complete** |

Phase 3 deliberately preceded phase 4: the execution pipeline exists before
agents do, so an agent is born unable to run anything unsupervised. An agent
will not be given a bypass, because there is nothing to bypass *to* — handlers
are unreachable from outside `eden.execution`.
