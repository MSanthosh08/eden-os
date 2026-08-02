"""API routes and the web console.

Routes are assembled here rather than inside the server so the transport stays
ignorant of EDEN, and so the whole API surface is visible in one place. Every
route reads from a running kernel; none of them reach past a subsystem's public
façade.

``allow_actions`` gates the mutating routes. An operator who wants a
read-only dashboard sets it to ``false`` and the task and action endpoints
return ``403`` — one flag, checked in one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eden.agents.types import Task
from eden.config.enums import ActionKind
from eden.core.types import ChatRequest, Message
from eden.errors import EdenError, InterfaceError
from eden.execution.types import Action
from eden.interface.server import Request, Response, Router, WebApprovalGate
from eden.memory.types import MemoryQuery

if TYPE_CHECKING:  # pragma: no cover - typing only
    from eden.core.kernel import EdenKernel

_DEFAULT_RECALL_LIMIT = 10
_JOURNAL_LIMIT = 50


def build_router(kernel: EdenKernel, gate: WebApprovalGate | None = None) -> Router:
    """Assemble the full API surface for ``kernel``.

    Args:
        kernel: A started kernel.
        gate: Approval gate whose pending queue is exposed, if one is in use.

    Returns:
        A populated router.
    """
    router = Router()
    config = kernel.config.interface

    def guard_actions() -> None:
        """Raise when mutating routes are disabled.

        Raises:
            InterfaceError: If ``allow_actions`` is off.
        """
        if not config.allow_actions:
            raise InterfaceError(
                "This interface is running in observe-only mode; "
                "set interface.allow_actions to enable tasks and actions."
            )

    # ------------------------------------------------------------------
    # Pages
    # ------------------------------------------------------------------
    async def index(request: Request) -> Response:
        """Serve the console."""
        del request
        return Response.html(CONSOLE_HTML)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    async def status(request: Request) -> Response:
        """Return a snapshot of every subsystem."""
        del request
        payload: dict[str, Any] = {
            "app": kernel.config.app_name,
            "version": kernel.config.version,
            "environment": kernel.config.environment.value,
            "subsystems": {
                "gateway": True,
                "memory": kernel.config.memory.enabled,
                "execution": kernel.config.execution.enabled,
                "agents": kernel.config.agents.enabled,
                "hardware": kernel.config.hardware.enabled,
                "automation": kernel.config.automation.enabled,
            },
            "providers": [line.provider for line in kernel.gateway.health_summary()],
            "allow_actions": config.allow_actions,
        }
        return Response.json(payload)

    async def providers(request: Request) -> Response:
        """Return provider health."""
        del request
        return Response.json(
            [
                {
                    "provider": line.provider,
                    "state": line.state.value,
                    "success_rate": line.success_rate,
                    "latency_ms": line.latency_ms,
                    "circuit": line.circuit,
                }
                for line in kernel.gateway.health_summary()
            ]
        )

    async def journal(request: Request) -> Response:
        """Return recent execution audit entries."""
        del request
        if not kernel.config.execution.enabled:
            return Response.json([])
        return Response.json(await kernel.execution.journal.read(limit=_JOURNAL_LIMIT))

    async def memory_search(request: Request) -> Response:
        """Return memories matching a query."""
        if not kernel.config.memory.enabled:
            return Response.json([])
        hits = await kernel.memory.recall(
            MemoryQuery(
                text=request.query.get("q", ""),
                namespace=request.query.get("namespace", "cli"),
                limit=int(request.query.get("limit", _DEFAULT_RECALL_LIMIT)),
            )
        )
        return Response.json(
            [
                {
                    "content": hit.record.content,
                    "kind": hit.record.kind.value,
                    "score": round(hit.score, 4),
                    "store": hit.store,
                    "created_at": hit.record.created_at.isoformat(),
                }
                for hit in hits
            ]
        )

    async def devices(request: Request) -> Response:
        """Return the device fleet."""
        del request
        if not kernel.config.hardware.enabled:
            return Response.json([])
        return Response.json([item.to_dict() for item in kernel.hardware.statuses()])

    async def rules(request: Request) -> Response:
        """Return automation rules and recent runs."""
        del request
        if not kernel.config.automation.enabled:
            return Response.json({"rules": [], "runs": []})
        scheduler = kernel.automation
        return Response.json(
            {
                "rules": [rule.to_dict() for rule in scheduler.rules],
                "runs": [run.to_dict() for run in scheduler.history[-20:]],
            }
        )

    async def approvals(request: Request) -> Response:
        """Return actions waiting for a human decision."""
        del request
        if gate is None:
            return Response.json([])
        return Response.json([item.to_dict() for item in gate.pending])

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    async def chat(request: Request) -> Response:
        """Generate a reply through the gateway, remembering the exchange.

        Mirrors the CLI's ``eden chat`` behaviour: the turn is observed, a
        durable-facts preamble is prepended so a stated name survives
        regardless of the token window, and the reply is remembered in turn.
        """
        payload = request.json()
        message = str(payload.get("message", "")).strip()
        if not message:
            raise InterfaceError("A 'message' field is required.")
        namespace = str(payload.get("namespace") or "cli")

        messages: list[Message] = [Message.user(message)]
        remembers = kernel.config.memory.enabled
        if remembers:
            await kernel.memory.observe(Message.user(message), namespace=namespace)
            messages = await kernel.memory.conversation.window(namespace)
            facts = await kernel.memory.facts_message(namespace)
            if facts is not None:
                messages = [facts, *messages]

        response = await kernel.gateway.chat(ChatRequest(messages=messages))
        if remembers:
            await kernel.memory.observe(Message.assistant(response.content), namespace=namespace)
        return Response.json(
            {
                "content": response.content,
                "provider": response.provider,
                "model": response.model,
                "latency_ms": round(response.latency_ms, 1),
                "cost": response.cost,
            }
        )

    async def dispatch_task(request: Request) -> Response:
        """Hand a goal to the agent orchestrator."""
        guard_actions()
        payload = request.json()
        goal = str(payload.get("goal", "")).strip()
        if not goal:
            raise InterfaceError("A 'goal' field is required.")
        context = payload.get("context")
        report = await kernel.agents.dispatch(
            Task(goal=goal, context=dict(context) if isinstance(context, dict) else {})
        )
        return Response.json(report.to_dict())

    async def submit_action(request: Request) -> Response:
        """Submit one action to the execution pipeline."""
        guard_actions()
        payload = request.json()
        try:
            kind = ActionKind(str(payload.get("kind", "")))
        except ValueError as exc:
            raise InterfaceError(
                f"Unknown action kind. Allowed: {[k.value for k in ActionKind]}"
            ) from exc
        parameters = payload.get("parameters")
        record = await kernel.execution.submit(
            Action(
                kind=kind,
                summary=str(payload.get("summary") or f"{kind.value} via web interface"),
                parameters=dict(parameters) if isinstance(parameters, dict) else {},
                actor="web",
            )
        )
        return Response.json(record.to_dict())

    async def resolve_approval(request: Request) -> Response:
        """Record a human approval decision."""
        guard_actions()
        if gate is None:
            raise InterfaceError("No web approval gate is configured.")
        payload = request.json()
        approval_id = str(payload.get("id", ""))
        approved = bool(payload.get("approved", False))
        resolved = gate.resolve(approval_id, approved=approved)
        return Response.json({"resolved": resolved, "approved": approved})

    async def run_rule(request: Request) -> Response:
        """Run one automation rule immediately."""
        guard_actions()
        if not kernel.config.automation.enabled:
            raise InterfaceError("The automation subsystem is disabled.")
        payload = request.json()
        name = str(payload.get("rule", ""))
        if not name:
            raise InterfaceError("A 'rule' field is required.")
        return Response.json((await kernel.automation.run(name)).to_dict())

    async def read_device(request: Request) -> Response:
        """Read one device channel.

        Reads are permitted even in observe-only mode: looking at a sensor
        changes nothing.
        """
        if not kernel.config.hardware.enabled:
            raise InterfaceError("The hardware subsystem is disabled.")
        device = request.query.get("device", "")
        channel = request.query.get("channel", "")
        if not device or not channel:
            raise InterfaceError("Both 'device' and 'channel' are required.")
        try:
            reading = await kernel.hardware.read(device, channel)
        except EdenError as exc:
            return Response.json(exc.to_dict(), status=400)
        return Response.json(reading.to_dict())

    router.add("GET", "/", index)
    router.add("GET", "/api/status", status)
    router.add("GET", "/api/providers", providers)
    router.add("GET", "/api/journal", journal)
    router.add("GET", "/api/memory", memory_search)
    router.add("GET", "/api/devices", devices)
    router.add("GET", "/api/device/read", read_device)
    router.add("GET", "/api/rules", rules)
    router.add("GET", "/api/approvals", approvals)
    router.add("POST", "/api/chat", chat)
    router.add("POST", "/api/task", dispatch_task)
    router.add("POST", "/api/action", submit_action)
    router.add("POST", "/api/approve", resolve_approval)
    router.add("POST", "/api/rule/run", run_rule)
    return router


CONSOLE_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EDEN</title>
<style>
:root{--bg:#0e1116;--panel:#161b22;--line:#262c36;--ink:#e6edf3;--dim:#8b949e;
--accent:#58a6ff;--ok:#3fb950;--warn:#d29922;--bad:#f85149}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;
align-items:baseline;gap:14px;flex-wrap:wrap}
h1{margin:0;font-size:16px;letter-spacing:.14em}
.sub{color:var(--dim);font-size:12px}
main{display:grid;grid-template-columns:1fr 1fr;gap:14px;padding:14px;max-width:1400px}
@media(max-width:900px){main{grid-template-columns:1fr}}
section{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}
h2{margin:0 0 10px;font-size:12px;letter-spacing:.12em;color:var(--dim);
text-transform:uppercase}
input,textarea,button{font:inherit;background:#0d1117;color:var(--ink);
border:1px solid var(--line);border-radius:6px;padding:8px 10px}
input,textarea{width:100%}
button{cursor:pointer;border-color:var(--accent);color:var(--accent)}
button:hover{background:#1c2836}
button.bad{border-color:var(--bad);color:var(--bad)}
button.ok{border-color:var(--ok);color:var(--ok)}
.row{display:flex;gap:8px;margin-top:8px}
.row button{flex:0 0 auto}
pre{white-space:pre-wrap;word-break:break-word;margin:10px 0 0;color:var(--dim);
max-height:340px;overflow:auto;font-size:12.5px}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;border:1px solid var(--line);
font-size:11px;color:var(--dim);margin:0 5px 5px 0}
.pill.on{color:var(--ok);border-color:var(--ok)}
.pill.off{color:var(--dim)}
.card{border:1px solid var(--line);border-radius:6px;padding:10px;margin-top:8px}
.card.risk{border-color:var(--warn)}
.muted{color:var(--dim);font-size:12px}
</style>
</head>
<body>
<header>
  <h1>EDEN</h1>
  <span class="sub" id="sub">connecting…</span>
  <span class="sub" id="subsystems"></span>
</header>
<main>
  <section>
    <h2>Chat</h2>
    <textarea id="msg" rows="3" placeholder="Ask anything…"></textarea>
    <div class="row"><button onclick="send()">Send</button></div>
    <pre id="chatOut"></pre>
  </section>

  <section>
    <h2>Agent task</h2>
    <input id="goal" placeholder="e.g. write a file summarising the release">
    <input id="path" placeholder="optional target path, e.g. notes.md" style="margin-top:8px">
    <div class="row"><button onclick="task()">Dispatch</button></div>
    <pre id="taskOut"></pre>
  </section>

  <section>
    <h2>Pending approvals</h2>
    <div id="approvals" class="muted">none</div>
  </section>

  <section>
    <h2>Memory</h2>
    <input id="q" placeholder="search remembered things…">
    <div class="row"><button onclick="recall()">Recall</button></div>
    <pre id="memOut"></pre>
  </section>

  <section>
    <h2>Execution journal</h2>
    <div class="row"><button onclick="load('/api/journal','journalOut')">Refresh</button></div>
    <pre id="journalOut"></pre>
  </section>

  <section>
    <h2>Providers &middot; devices &middot; rules</h2>
    <div class="row">
      <button onclick="load('/api/providers','infoOut')">Providers</button>
      <button onclick="load('/api/devices','infoOut')">Devices</button>
      <button onclick="load('/api/rules','infoOut')">Rules</button>
    </div>
    <pre id="infoOut"></pre>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
const show = (id, v) => $(id).textContent =
  typeof v === 'string' ? v : JSON.stringify(v, null, 2);

async function api(path, body) {
  const opts = body ? {method:'POST', headers:{'Content-Type':'application/json'},
                       body: JSON.stringify(body)} : {};
  const r = await fetch(path, opts);
  return r.json();
}
async function load(path, target) {
  show(target, 'loading…');
  try { show(target, await api(path)); } catch (e) { show(target, String(e)); }
}
async function send() {
  const m = $('msg').value.trim(); if (!m) return;
  show('chatOut', 'thinking…');
  const r = await api('/api/chat', {message: m});
  show('chatOut', r.content ? `[${r.provider}/${r.model} ${r.latency_ms}ms]\\n\\n${r.content}`
                            : r);
}
async function task() {
  const goal = $('goal').value.trim(); if (!goal) return;
  const path = $('path').value.trim();
  show('taskOut', 'working…');
  show('taskOut', await api('/api/task', {goal, context: path ? {path} : {}}));
}
async function recall() {
  show('memOut', await api('/api/memory?q=' + encodeURIComponent($('q').value)));
}
async function decide(id, approved) {
  await api('/api/approve', {id, approved});
  refreshApprovals();
}
async function refreshApprovals() {
  let items = [];
  try { items = await api('/api/approvals'); } catch (e) { return; }
  const box = $('approvals');
  if (!items.length) { box.innerHTML = '<span class="muted">none</span>'; return; }
  box.innerHTML = items.map(a => `
    <div class="card risk">
      <div><strong>${a.action.summary}</strong></div>
      <div class="muted">${a.action.kind} &middot; risk ${a.verdict.risk}
        &middot; ${a.rollback}</div>
      <div class="muted">${a.verdict.findings.map(f => f.message).join(' ')}</div>
      <div class="row">
        <button class="ok"  onclick="decide('${a.id}', true)">Approve</button>
        <button class="bad" onclick="decide('${a.id}', false)">Refuse</button>
      </div>
    </div>`).join('');
}
async function refreshStatus() {
  try {
    const s = await api('/api/status');
    $('sub').textContent = `${s.app} ${s.version} \\u00b7 ${s.environment}` +
      (s.allow_actions ? '' : ' \\u00b7 observe-only');
    $('subsystems').innerHTML = Object.entries(s.subsystems)
      .map(([k,v]) => `<span class="pill ${v?'on':'off'}">${k}</span>`).join('');
  } catch (e) { $('sub').textContent = 'disconnected'; }
}
refreshStatus(); refreshApprovals();
setInterval(refreshApprovals, 2000);
setInterval(refreshStatus, 10000);
</script>
</body>
</html>
"""
