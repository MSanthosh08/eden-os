# ADR-0006: Hardware actuation is an execution action; automation is data; the interface is loopback-only

**Status:** Accepted
**Date:** 2026-07-26
**Deciders:** Lead Architect

## Context

Phase 5 adds the three subsystems that let EDEN touch the world outside its own
process: hardware, automation, and a user interface. Each raises the same
question in a different form — *what stops this from doing something nobody
asked for?* — and the answer in all three cases turned out to be "route it
through machinery that already exists" rather than "add new safeguards".

Three specific pressures shaped the design.

**Hardware failure is not recoverable.** A wrong file is restorable from a
rollback plan. A wrong servo angle can break a machine, or a person. Whatever
guards exist must be in front of the actuator, not around it.

**Automation removes the human from the loop by definition.** A scheduled rule
runs at 03:00 with nobody watching. Anything a rule can do, it can do
unobserved, repeatedly.

**A UI is an attack surface.** The moment EDEN answers HTTP, everything behind
it is reachable by whatever can reach that port.

## Decision

**Hardware: reads are direct, writes are actions.**

`BaseDevice.read()` is an ordinary call — observing a sensor changes nothing,
and requiring approval to look at a thermometer would make the system unusable
without making it safer. Actuation is different: the only route to a moving part
is an `Action` of kind `device_command` submitted to the Phase 3 pipeline.
`DeviceCommandHandler` is an ordinary `ActionHandler`, registered by the kernel,
so a servo command inherits verification, permission, journalling and rollback
from machinery that is already written and already tested.

`prepare` reads the channel's current value, which makes most device commands
*reversible* and therefore eligible for the normal policy ladder. Safety limits
live in `DeviceConfig`, not in driver code, so an operator can tighten them
without a deploy. `DeviceSafetyError` is explicitly non-retryable: a command
outside the envelope does not become safe on a second attempt.

`hardware.enabled` defaults to `False`.

**Automation: a rule is data, never a callable.**

`Rule` pairs a `Trigger` with exactly one of a `Task` or an `Action`. It cannot
hold a function. That makes every rule listable, inspectable, disableable and
auditable, and — decisively — means automation cannot reach past the same
façades a human operator uses. A rule that acts goes through the execution
pipeline; a rule that delegates goes through the agent orchestrator.

The scheduler holds two injected coroutines rather than the subsystems
themselves, so it is decoupled from what it drives and testable with recording
doubles. Time comes from the injected `Clock`; the entire scheduler is tested
without waiting.

`automation.enabled` defaults to `False`, and `catch_up` defaults to `False` so
a restart does not fire a day of missed hourly jobs at once.

**Interface: loopback, no framework, and a human approval gate.**

A ~350-line asyncio HTTP server rather than a web framework, because EDEN has
carried zero runtime dependencies for five phases and a dozen JSON routes is not
worth breaking that for.

`WebApprovalGate` is the payoff of ADR-0004's `ApprovalGate` seam. A pending
action blocks, surfaces in the UI with its risk findings and rollback plan, and
**times out into refusal**. An interface that cannot reach a human must behave
exactly like an interface with no human behind it.

`interface.enabled` defaults to `False` and `host` defaults to `127.0.0.1`. The
server has no authentication; that is stated in the config docstring, enforced
by the default, and repeated in the startup log.

## Options considered

### Hardware

| Option | Complexity | Safety | Auditability |
|---|---|---|---|
| A: Direct driver API, guards inside drivers | Low | Poor — each driver reimplements them | None |
| B: Separate hardware permission system | High | Good | Duplicated |
| **C: Actuation as an execution action** | **Low** | **Good** | **Free** |

Option A puts the guard inside the thing being guarded; a driver bug removes
both. Option B is Option C with a second, less-tested copy of the pipeline —
and two permission systems means two places to get a policy wrong. Option C
costs one handler class.

The trade Option C accepts: device commands must be expressible as
`(device, channel, value)` triples. A protocol needing rich structured commands
would need a richer action payload. That is a real limit, recorded below.

### Automation

| Option | Inspectable | Sandboxed | Expressive |
|---|---|---|---|
| A: Rules hold callables | No | No | Very |
| B: Rules hold a script/DSL | Partly | Needs an interpreter | Very |
| **C: Rules hold a Task or an Action** | **Yes** | **Inherited** | **Adequate** |

Option A is the obvious one and is wrong: a callable is opaque to inspection and
can do anything the process can, bypassing five phases of safeguards in one
lambda. Option B means writing and securing an interpreter. Option C gives up
arbitrary expressiveness and gets sandboxing for free — anything a rule needs to
do that a task or action cannot express should probably be an agent.

### Interface

| Option | Dependencies | Effort | Fit |
|---|---|---|---|
| A: FastAPI/Starlette | +3 packages | Low | Good |
| B: stdlib `http.server` | 0 | Low | **Poor — synchronous** |
| **C: Minimal asyncio server** | **0** | **Moderate** | **Good** |

Option B is disqualified: a sync server inside an async kernel means a thread
bridge for every request. Option A is the pragmatic industry choice and would be
right for a larger surface. Option C was chosen because the surface is small and
the zero-dependency property has repeatedly paid off — it is why the test suite
needs no network and why installation is one command.

## Consequences

**Easier**
- One audit trail covers files, commands *and* physical actuation.
- Tightening a servo's range is a config edit, not a deploy.
- Automation is inspectable: `eden rules` lists exactly what can fire.
- Approval finally has a face; `CallbackGate` was designed for this in Phase 3.
- `pip install -e .` still pulls nothing.

**Harder**
- Device commands are limited to `(device, channel, value)`.
- The HTTP server implements a deliberate subset: one request per connection,
  no keep-alive, no TLS, no compression, no streaming responses.
- No authentication at all, which is why loopback is the enforced default.

**To revisit**
- **The interface has no authentication.** Acceptable only while it is loopback.
  A token or local socket is required before it binds anything else.
- Serial and GPIO devices need drivers; the `DeviceKind.CUSTOM` seam exists but
  `pyserial` was kept out of the dependency list. Revisit when a real rig lands.
- `DailyTrigger` is UTC-only; local time and DST need a real timezone database.
- The scheduler is single-process. Distributed execution needs the fire history
  in shared state, same as the round-robin cursor from ADR-0002.
- SSE or WebSocket streaming would make the console feel live; today it polls.

## Action items

1. [x] `BaseDevice` with connect/read/send, state tracking, rate limiting, bounds.
2. [x] `SimulatedDevice` and `HttpDevice`; `CUSTOM` import seam.
3. [x] `DeviceCommandHandler` registered by the kernel into the Phase 3 pipeline.
4. [x] Triggers (interval, daily, event, manual), `Rule`, `AutomationScheduler`.
5. [x] Dependency-free HTTP server, JSON API, single-page console.
6. [x] `WebApprovalGate` that times out into refusal.
7. [x] CLI with `status/ask/chat/task/memory/devices/rules/serve`.
8. [ ] Authentication before the interface may bind a non-loopback address.
9. [ ] Serial/GPIO driver once target hardware is chosen.
10. [ ] Timezone-aware daily triggers.
