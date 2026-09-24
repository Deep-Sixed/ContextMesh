"""Context Mesh MCP — a read-only interface over a Context Mesh graph.

Experimental. It serves either the bundled demo graph, rebuilt per process, or
a session directory that outlives the process — and it makes the caller say
which, because silently serving a throwaway graph in place of someone's real
one is the failure the session format exists to prevent.

The split here is deliberate. ``session`` and ``tools`` are plain Python over
the engine and import nothing from the MCP SDK, so the safety tests that matter
run on every supported version with no dependencies — and so ``python -m
contextmesh_mcp`` can write and inspect a session without the SDK at all. Only
``server`` needs it, and only ``server`` requires Python 3.10+.

The invariant this package is built around:

    A read operation may move walk telemetry. It may not change graph
    structure, ontology state, assumptions, supersession, or invalidation.

Asking a question *is* a write in Context Mesh — a walk bumps ``node.walks`` and
``edge.traversals``, and PRUNE later drops what nothing walked. That is designed
behaviour, so "the graph is byte-identical after a read" is the wrong test. The
right one separates structure and belief from usage, and ``tests/test_mcp.py``
asserts the first two are untouched while allowing the third to move.

No tool here rejects an assumption. ``mesh_blast_radius`` answers "what *would*
fall", which needs no authority; deciding an assumption is false is reserved for
an auditor holding evidence, per GRAPH.md rule 7. A tool that let a client name
an assumption and have it rejected would make that rule a convention.
"""

from .session import Session, SessionError
from .tools import (
    TOOLS,
    MeshToolError,
    mesh_ask,
    mesh_blast_radius,
    mesh_get_node,
    mesh_health,
    mesh_lineage,
)

__version__ = "0.1.1"

__all__ = [
    "TOOLS",
    "MeshToolError",
    "Session",
    "SessionError",
    "mesh_ask",
    "mesh_blast_radius",
    "mesh_get_node",
    "mesh_health",
    "mesh_lineage",
]
