"""Command-line interface.

The CLI exists so that using EDEN does not require writing Python. Every
command boots a real kernel through the same composition root as any other
entry point — there is no separate "CLI mode" that behaves differently, which
is what stops the CLI and the library from drifting apart.

Commands::

    eden status                       subsystem and provider overview
    eden ask "question"               one-shot generation
    eden chat                         interactive session with memory
    eden task "goal" [--path P]       hand a goal to an agent
    eden memory "query"               search remembered things
    eden devices                      list the device fleet
    eden rules                        list automation rules
    eden serve                        start the web console
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eden.agents.types import Task
from eden.config.enums import LogLevel
from eden.config.loader import DEFAULT_CONFIG_FILENAME, ConfigLoader
from eden.config.schema import EdenConfig
from eden.core.kernel import EdenKernel
from eden.core.types import ChatRequest, Message
from eden.errors import EdenError
from eden.interface.api import build_router
from eden.interface.server import HttpServer, WebApprovalGate
from eden.memory.types import MemoryQuery

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_INTERRUPTED = 130

_PROMPT = "you> "
_QUIT = frozenset({"exit", "quit", ":q"})


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the ``eden`` command."""
    parser = argparse.ArgumentParser(
        prog="eden",
        description="EDEN — a modular AI Operating System.",
    )
    parser.add_argument("--config", type=Path, help="path to eden.toml")
    parser.add_argument(
        "--log-level",
        choices=[level.value for level in LogLevel],
        help="override the configured log level",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="show subsystem and provider status")

    ask = sub.add_parser("ask", help="one-shot generation")
    ask.add_argument("prompt", nargs="+", help="what to ask")

    chat = sub.add_parser("chat", help="interactive session with memory")
    chat.add_argument("--namespace", default="cli", help="conversation namespace")

    task = sub.add_parser("task", help="hand a goal to an agent")
    task.add_argument("goal", nargs="+", help="what to achieve")
    task.add_argument("--path", help="target path, for file tasks")

    memory = sub.add_parser("memory", help="search remembered things")
    memory.add_argument("query", nargs="*", default=[], help="search text")
    memory.add_argument("--namespace", default="cli")
    memory.add_argument("--limit", type=int, default=10)

    sub.add_parser("devices", help="list the device fleet")
    sub.add_parser("rules", help="list automation rules and recent runs")

    serve = sub.add_parser("serve", help="start the local web console")
    serve.add_argument("--host", help="override the configured host")
    serve.add_argument("--port", type=int, help="override the configured port")
    return parser


def _load(args: argparse.Namespace) -> EdenConfig:
    """Build configuration from the file, environment and CLI overrides.

    The CLI is interactive, so it defaults to a quiet log level — the
    subsystem startup/shutdown trace is operational detail a long-running
    service benefits from, but it is noise for someone running one command
    and reading its answer. That quiet default only applies when nothing
    else has asked for a level: a person's own ``eden.toml``, their
    environment, or ``--log-level`` on the command line all still win.
    """
    loader = ConfigLoader()
    path = args.config if args.config else Path(DEFAULT_CONFIG_FILENAME)
    loader.with_toml(path, required=args.config is not None)
    loader.with_environ()
    merged = loader.merged()

    overrides: dict[str, Any] = {}
    level_already_set = isinstance(merged.get("logging"), dict) and "level" in merged["logging"]
    if args.log_level:
        overrides["logging"] = {"level": args.log_level}
    elif not level_already_set:
        overrides["logging"] = {"level": LogLevel.WARNING.value}
    if getattr(args, "host", None) or getattr(args, "port", None):
        interface: dict[str, Any] = {"enabled": True}
        if getattr(args, "host", None):
            interface["host"] = args.host
        if getattr(args, "port", None):
            interface["port"] = args.port
        overrides["interface"] = interface

    loader.with_overrides(overrides)
    return loader.build()


def _emit(payload: Any, *, as_json: bool) -> None:  # noqa: ANN401 - varied payloads
    """Write a result to stdout in the requested format."""
    if as_json:
        sys.stdout.write(json.dumps(payload, default=str, indent=2) + "\n")
        return
    if isinstance(payload, str):
        sys.stdout.write(payload + "\n")
        return
    sys.stdout.write(json.dumps(payload, default=str, indent=2) + "\n")


async def _status(kernel: EdenKernel, *, as_json: bool) -> int:
    """Print subsystem and provider status."""
    config = kernel.config
    health = kernel.gateway.health_summary()
    payload = {
        "app": config.app_name,
        "version": config.version,
        "environment": config.environment.value,
        "subsystems": {
            "gateway": True,
            "memory": config.memory.enabled,
            "execution": config.execution.enabled,
            "agents": config.agents.enabled,
            "hardware": config.hardware.enabled,
            "automation": config.automation.enabled,
            "interface": config.interface.enabled,
        },
        "providers": [
            {
                "name": line.provider,
                "state": line.state.value,
                "latency_ms": line.latency_ms,
                "circuit": line.circuit,
            }
            for line in health
        ],
    }
    if as_json:
        _emit(payload, as_json=True)
        return EXIT_OK
    subsystems: dict[str, bool] = {
        "gateway": True,
        "memory": config.memory.enabled,
        "execution": config.execution.enabled,
        "agents": config.agents.enabled,
        "hardware": config.hardware.enabled,
        "automation": config.automation.enabled,
        "interface": config.interface.enabled,
    }
    lines = [f"{config.app_name} {config.version}  [{config.environment.value}]", ""]
    lines.extend(f"  {'on ' if enabled else 'off'}  {name}" for name, enabled in subsystems.items())
    lines.append("")
    lines.append(f"  providers ({len(health)}):")
    lines.extend(
        f"    {line.provider:<16} {line.state.value:<10} {line.latency_ms:>8.0f}ms  {line.circuit}"
        for line in health
    )
    _emit("\n".join(lines), as_json=False)
    return EXIT_OK


async def _ask(kernel: EdenKernel, prompt: str, *, as_json: bool) -> int:
    """Run one generation and print the reply."""
    response = await kernel.gateway.chat(ChatRequest(messages=[Message.user(prompt)]))
    if as_json:
        _emit(
            {
                "content": response.content,
                "provider": response.provider,
                "model": response.model,
                "latency_ms": response.latency_ms,
                "cost": response.cost,
            },
            as_json=True,
        )
    else:
        _emit(response.content, as_json=False)
        sys.stderr.write(f"[{response.provider}/{response.model} {response.latency_ms:.0f}ms]\n")
    return EXIT_OK


async def _chat(kernel: EdenKernel, namespace: str) -> int:
    """Run an interactive session, remembering the conversation."""
    remembers = kernel.config.memory.enabled
    sys.stderr.write("EDEN chat. 'exit' to leave.\n")
    while True:
        try:
            line = input(_PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            sys.stderr.write("\n")
            return EXIT_OK
        if not line:
            continue
        if line.lower() in _QUIT:
            return EXIT_OK

        history: list[Message] = []
        if remembers:
            await kernel.memory.observe(Message.user(line), namespace=namespace)
            history = await kernel.memory.conversation.window(namespace)
            facts = await kernel.memory.facts_message(namespace)
            if facts is not None:
                # Facts are looked up directly rather than carried in the
                # rolling window, so a name stays available even once the
                # turn that stated it has aged out of the token budget.
                history = [facts, *history]
        messages = history or [Message.user(line)]

        try:
            response = await kernel.gateway.chat(ChatRequest(messages=messages))
        except EdenError as exc:
            sys.stderr.write(f"error: {exc.message}\n")
            continue
        sys.stdout.write(response.content + "\n")
        if remembers:
            await kernel.memory.observe(Message.assistant(response.content), namespace=namespace)


async def _task(kernel: EdenKernel, goal: str, path: str | None, *, as_json: bool) -> int:
    """Dispatch a goal to an agent and print the report."""
    report = await kernel.agents.dispatch(Task(goal=goal, context={"path": path} if path else {}))
    if as_json:
        _emit(report.to_dict(), as_json=True)
        return EXIT_OK if report.succeeded else EXIT_ERROR
    lines = [
        f"agent    : {report.agent}",
        f"status   : {report.status.value}",
        f"summary  : {report.summary}",
    ]
    if report.plan is not None:
        lines.append(f"steps    : {len(report.plan.steps)}")
    output = report.final_output
    if output:
        lines.extend(["", output])
    _emit("\n".join(lines), as_json=False)
    return EXIT_OK if report.succeeded else EXIT_ERROR


async def _memory(kernel: EdenKernel, args: argparse.Namespace) -> int:
    """Search memory and print the hits."""
    if not kernel.config.memory.enabled:
        _emit("memory is disabled in configuration", as_json=args.json)
        return EXIT_ERROR
    hits = await kernel.memory.recall(
        MemoryQuery(
            text=" ".join(args.query),
            namespace=args.namespace,
            limit=args.limit,
        )
    )
    if args.json:
        _emit(
            [
                {
                    "content": hit.record.content,
                    "kind": hit.record.kind.value,
                    "score": hit.score,
                }
                for hit in hits
            ],
            as_json=True,
        )
        return EXIT_OK
    if not hits:
        _emit("no matches", as_json=False)
        return EXIT_OK
    _emit(
        "\n".join(
            f"  [{hit.score:.2f}] {hit.record.kind.value:<13} {hit.record.content[:90]}"
            for hit in hits
        ),
        as_json=False,
    )
    return EXIT_OK


async def _devices(kernel: EdenKernel, *, as_json: bool) -> int:
    """Print the device fleet."""
    if not kernel.config.hardware.enabled:
        _emit("hardware is disabled in configuration", as_json=as_json)
        return EXIT_OK
    statuses = kernel.hardware.statuses()
    if as_json:
        _emit([item.to_dict() for item in statuses], as_json=True)
        return EXIT_OK
    if not statuses:
        _emit("no devices configured", as_json=False)
        return EXIT_OK
    _emit(
        "\n".join(
            f"  {item.name:<18} {item.state.value:<13} channels={','.join(item.channels) or '-'}"
            for item in statuses
        ),
        as_json=False,
    )
    return EXIT_OK


async def _rules(kernel: EdenKernel, *, as_json: bool) -> int:
    """Print automation rules and recent runs."""
    if not kernel.config.automation.enabled:
        _emit("automation is disabled in configuration", as_json=as_json)
        return EXIT_OK
    scheduler = kernel.automation
    if as_json:
        _emit(
            {
                "rules": [rule.to_dict() for rule in scheduler.rules],
                "runs": [run.to_dict() for run in scheduler.history[-10:]],
            },
            as_json=True,
        )
        return EXIT_OK
    lines = [
        f"  {rule.name:<20} {rule.trigger.describe():<24} {rule.description}"
        for rule in scheduler.rules
    ]
    _emit("\n".join(lines) if lines else "no rules registered", as_json=False)
    return EXIT_OK


async def _serve(kernel: EdenKernel, gate: WebApprovalGate) -> int:
    """Start the web console and serve until interrupted."""
    server = HttpServer(kernel.config.interface, build_router(kernel, gate))
    await server.start()
    sys.stderr.write(f"\n  EDEN console: {server.url}\n  Ctrl-C to stop.\n\n")
    try:
        await asyncio.Event().wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await server.stop()
    return EXIT_OK


async def _run(args: argparse.Namespace) -> int:
    """Boot a kernel and dispatch the requested command."""
    config = _load(args)
    gate = WebApprovalGate(config.interface)
    async with EdenKernel(config, approval_gate=gate).session() as kernel:
        if args.command == "status":
            return await _status(kernel, as_json=args.json)
        if args.command == "ask":
            return await _ask(kernel, " ".join(args.prompt), as_json=args.json)
        if args.command == "chat":
            return await _chat(kernel, args.namespace)
        if args.command == "task":
            return await _task(kernel, " ".join(args.goal), args.path, as_json=args.json)
        if args.command == "memory":
            return await _memory(kernel, args)
        if args.command == "devices":
            return await _devices(kernel, as_json=args.json)
        if args.command == "rules":
            return await _rules(kernel, as_json=args.json)
        if args.command == "serve":
            return await _serve(kernel, gate)
    return EXIT_ERROR


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``eden`` console script.

    Args:
        argv: Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        sys.stderr.write("\ninterrupted\n")
        return EXIT_INTERRUPTED
    except EdenError as exc:
        sys.stderr.write(f"error [{exc.code}]: {exc.message}\n")
        if exc.context:
            sys.stderr.write(f"  context: {json.dumps(exc.context, default=str)}\n")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover - console entry point
    with contextlib.suppress(BrokenPipeError):
        raise SystemExit(main())
