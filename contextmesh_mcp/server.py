"""Context Mesh MCP server — stdio transport.

    pip install 'contextmesh-graph[mcp]'
    contextmesh-mcp --demo --rounds 8 --save ./session   # write one
    contextmesh-mcp --session ./session                  # serve it

Requires Python 3.10+, because the MCP SDK does. Writing and inspecting a
session needs neither the SDK nor 3.10, which is why that lives in
``session.py`` and is reachable as ``python -m contextmesh_mcp``. Read tools
live in ``tools.py``; controlled structural writes live in ``writes.py``. Both
remain plain Python so their contracts are tested without the SDK on every
version the core supports. This file is transport and nothing else.

PR #8 keeps client authority narrow: evidence enters as observation; recheck
runs deployment-owned auditors; repair selects deployment-owned TaskRegistry
keys; resume delegates to the native selective scheduler. There is no direct
reject, invalidate, status setter, callable, module path or client audit verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Sequence

from contextmesh.execute import TaskRegistry

try:
    from mcp.server import MCPServer
except ImportError as exc:  # pragma: no cover - depends on the optional extra
    raise SystemExit(
        "the MCP SDK is not installed. Install the extra:\n"
        "    pip install 'contextmesh-graph[mcp]'\n"
        "It needs Python 3.10 or newer; contextmesh itself supports 3.9.\n"
        f"(import failed: {exc})"
    ) from exc

from . import resources, tools, writes
from .session import (
    Checkpointer,
    SessionError,
    add_source_arguments,
    adopt,
    open_session,
    session,
)

SERVER_NAME = "context-mesh"
INSTRUCTIONS = """\
Context Mesh is a typed context graph: entities, claims, sources, decisions,
assumptions and evidence, connected by typed edges that record why they are
connected.

Ask it questions with mesh_ask. An answer comes back as a path you can read,
with the evidence on it — or as one of four typed dead-end reasons rather than a
guess, so "the graph cannot answer this" is a distinguishable outcome.

mesh_blast_radius is a dry run: it tells you what would fall if an assumption
turned out to be false, and what would survive. It changes nothing.

Controlled writes are deliberately split by authority:

- mesh_submit_evidence stores a raw observation. It cannot name an edge,
  assumption, verdict, rejection or invalidation target.
- mesh_recheck asks the registered auditors to interpret current state. The
  client supplies no verdict. A disproof must identify pre-ingested evidence.
- mesh_repair can select only durable worker/auditor keys already registered in
  this deployment. No callable, module path or import string crosses MCP.
- mesh_resume delegates to the native scheduler, which reruns only pending/stale
  work and leaves unaffected DONE work cached.

This instance may be serving a saved session directory or a demo graph rebuilt
for this process. contextmesh://session says which, and whether anything you see
here outlives the server. Controlled writes refuse an ephemeral demo session.

Independent project. No affiliation with, or endorsement by, anyone involved in
the screen capture the dashboard was reverse-engineered from.
"""

mcp = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS)


def _payload(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)


#: Set by ``main`` once the session is open. ``None`` while the module is
#: merely imported, which is how the tool handlers stay callable in tests.
_CHECKPOINTER: Optional[Checkpointer] = None


def _run(
    name: str, arguments: Optional[Dict[str, Any]] = None, *, mutates: bool = False
) -> str:
    """Answer a read tool call, checkpointing telemetry when appropriate.

    ``mesh_ask`` moves walk/resolver telemetry. A failed checkpoint for that
    observational telemetry is reported to stderr without turning a good answer
    into a dead channel. Structural writes use ``_run_write`` instead because
    they have the stronger success-means-durable contract.
    """
    try:
        result = tools.call(session(), name, arguments)
    except tools.MeshToolError as exc:
        return _payload({"error": "not_found", "tool": name, "message": str(exc)})
    except Exception as exc:  # surface engine errors as data, not a dead channel
        return _payload(
            {"error": type(exc).__name__, "tool": name, "message": str(exc)}
        )
    if mutates and _CHECKPOINTER is not None:
        try:
            _CHECKPOINTER.record_mutation()
        except Exception as exc:  # a failed telemetry write is reported, not fatal
            print(f"context-mesh: checkpoint failed: {exc}", file=sys.stderr)
    return _payload(result)


def _run_write(name: str, **arguments: Any) -> str:
    """Run a controlled structural write and adopt only its committed clone."""
    global _CHECKPOINTER
    try:
        entry = writes.WRITE_TOOLS[name]
        result = entry["fn"](session(), _CHECKPOINTER, **arguments)
    except Exception as exc:
        return _payload(
            {"error": type(exc).__name__, "tool": name, "message": str(exc)}
        )
    if result.session is not session():
        adopt(result.session)
        assert _CHECKPOINTER is not None
        _CHECKPOINTER.session = result.session
    return _payload(result.payload)


# ── read tools ───────────────────────────────────────────────────────────
@mcp.tool(name="mesh_ask", description=tools.TOOLS["mesh_ask"]["description"])
def mesh_ask(question: str) -> str:
    return _run("mesh_ask", {"question": question}, mutates=True)


@mcp.tool(name="mesh_get_node", description=tools.TOOLS["mesh_get_node"]["description"])
def mesh_get_node(node_id: str) -> str:
    return _run("mesh_get_node", {"node_id": node_id})


@mcp.tool(name="mesh_health", description=tools.TOOLS["mesh_health"]["description"])
def mesh_health() -> str:
    return _run("mesh_health", {})


@mcp.tool(name="mesh_lineage", description=tools.TOOLS["mesh_lineage"]["description"])
def mesh_lineage(assumption_id: str) -> str:
    return _run("mesh_lineage", {"assumption_id": assumption_id})


@mcp.tool(
    name="mesh_blast_radius",
    description=tools.TOOLS["mesh_blast_radius"]["description"],
)
def mesh_blast_radius(assumption_id: str) -> str:
    return _run("mesh_blast_radius", {"assumption_id": assumption_id})


@mcp.tool(
    name="mesh_explain_as_of",
    description=tools.TOOLS["mesh_explain_as_of"]["description"],
)
def mesh_explain_as_of(question: str, as_of: str) -> str:
    # ``mutates`` stays off. ``mesh_ask`` sets it because a walk moves
    # telemetry; this walks a projection built for the call and thrown away
    # after it, so the live graph counts nothing.
    return _run("mesh_explain_as_of", {"question": question, "as_of": as_of})


@mcp.tool(
    name="mesh_reconstruct_decision",
    description=tools.TOOLS["mesh_reconstruct_decision"]["description"],
)
def mesh_reconstruct_decision(decision_id: str, as_of: str, depth: int = 3) -> str:
    return _run(
        "mesh_reconstruct_decision",
        {"decision_id": decision_id, "as_of": as_of, "depth": depth},
    )


# ── controlled writes ────────────────────────────────────────────────────
@mcp.tool(
    name="mesh_submit_evidence",
    description=writes.WRITE_TOOLS["mesh_submit_evidence"]["description"],
)
def mesh_submit_evidence(
    text: str,
    source_id: str,
    external_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    return _run_write(
        "mesh_submit_evidence",
        text=text,
        source_id=source_id,
        external_id=external_id,
        metadata=metadata,
    )


@mcp.tool(
    name="mesh_recheck",
    description=writes.WRITE_TOOLS["mesh_recheck"]["description"],
)
def mesh_recheck() -> str:
    return _run_write("mesh_recheck")


@mcp.tool(
    name="mesh_repair",
    description=writes.WRITE_TOOLS["mesh_repair"]["description"],
)
def mesh_repair(
    task: str,
    worker_key: str,
    assumes: str,
    auditor_key: Optional[str] = None,
    produces: Optional[List[str]] = None,
    rationale: Optional[str] = None,
) -> str:
    return _run_write(
        "mesh_repair",
        task=task,
        worker_key=worker_key,
        assumes=assumes,
        auditor_key=auditor_key,
        produces=produces,
        rationale=rationale,
    )


@mcp.tool(
    name="mesh_resume",
    description=writes.WRITE_TOOLS["mesh_resume"]["description"],
)
def mesh_resume() -> str:
    return _run_write("mesh_resume")


# ── resources ────────────────────────────────────────────────────────────
@mcp.resource("contextmesh://schema", mime_type="text/markdown")
def resource_schema() -> str:
    return resources.schema_text()


@mcp.resource("contextmesh://health", mime_type="application/json")
def resource_health() -> str:
    return resources.read(session(), "contextmesh://health")


@mcp.resource("contextmesh://session", mime_type="application/json")
def resource_session() -> str:
    return resources.read(session(), "contextmesh://session")


@mcp.resource("contextmesh://assumptions", mime_type="application/json")
def resource_assumptions() -> str:
    return resources.read(session(), "contextmesh://assumptions")


@mcp.resource("contextmesh://node/{node_id}", mime_type="application/json")
def resource_node(node_id: str) -> str:
    return resources.read(session(), f"contextmesh://node/{node_id}")


@mcp.resource("contextmesh://assumption/{assumption_id}", mime_type="application/json")
def resource_assumption(assumption_id: str) -> str:
    return resources.read(session(), f"contextmesh://assumption/{assumption_id}")


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    registry: Optional[TaskRegistry] = None,
) -> int:
    parser = add_source_arguments(
        argparse.ArgumentParser(
            prog="contextmesh-mcp",
            description=(
                "MCP server over a Context Mesh graph with controlled evidence intake."
            ),
            epilog=(
                "contextmesh-mcp --demo --rounds 8 --save ./session   write one\n"
                "contextmesh-mcp --session ./session                  serve it"
            ),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    )
    args = parser.parse_args(argv)

    try:
        opened = open_session(args, registry=registry)
    except SessionError as exc:
        print(f"contextmesh-mcp: {exc}", file=sys.stderr)
        return 2

    if args.save is not None:
        target = opened.save(args.save)
        print(f"contextmesh-mcp: wrote {target}", file=sys.stderr)
        return 0

    global _CHECKPOINTER
    adopt(opened)
    _CHECKPOINTER = Checkpointer(opened, args.checkpoint)
    described = opened.describe()
    where = (
        f"from {opened.path} gen {described['generation']}, checkpoint {args.checkpoint}"
        if described["persistent"]
        else "not persistent"
    )
    print(
        f"context-mesh MCP: {described['nodes_live']}/{described['nodes_total']} nodes live, "
        f"{described['edges_live']}/{described['edges_total']} edges live, "
        f"controlled writes, {where}",
        file=sys.stderr,
    )
    try:
        mcp.run(transport="stdio")
    finally:
        try:
            written = _CHECKPOINTER.close()
        except Exception as exc:
            print(f"context-mesh: final checkpoint failed: {exc}", file=sys.stderr)
        else:
            if written is not None:
                print(f"context-mesh: checkpointed {written}", file=sys.stderr)
    return 0
