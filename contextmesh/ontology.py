"""The ontology is a file, not a constant.

``GRAPH.md``, shipped inside this package, is the schema. This module parses it on
import so that a change to the documented edge table is immediately a change to
what the engine will accept. That is the whole point of the "read on every
write" line in the dashboard: the ontology cannot drift away from the docs
because there is only one copy of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, Tuple

# Inside the package, not beside it: the wheel used to install this as a
# loose top-level site-packages/GRAPH.md, where any other distribution that
# shipped or removed a file of that name would break `import contextmesh`.
ONTOLOGY_FILE = Path(__file__).resolve().parent / "GRAPH.md"

_ROW = re.compile(r"^\|(?P<cells>.+)\|\s*$")
_PAIR = re.compile(r"(?P<src>[a-z_]+)\s*(?:→|->)\s*(?P<dst>[a-z_]+)")

#: The only values GRAPH.md's edge-table ``Invalidation`` column may hold.
_DIRECTIONS = frozenset({"backward", "forward", "none"})


class OntologyError(Exception):
    """Raised when a write does not typecheck against GRAPH.md."""


@dataclass(frozen=True)
class Ontology:
    node_types: FrozenSet[str]
    edge_types: FrozenSet[str]
    #: edge type -> set of legal (source node type, target node type) pairs
    signatures: Dict[str, FrozenSet[Tuple[str, str]]]
    #: edge type -> "backward" | "forward" | "none", parsed from GRAPH.md's
    #: edge-table ``Invalidation`` column (rule 2). The one place this
    #: direction is expressed; ``contextmesh/assumptions.py`` reads it from
    #: here rather than keeping its own copy.
    invalidation: Dict[str, str]
    #: node type -> the field/attrs names GRAPH.md's ``Must carry`` column
    #: requires to be present (see GRAPH.md's "What `Must carry` means").
    must_carry: Dict[str, FrozenSet[str]]

    @property
    def backward(self) -> FrozenSet[str]:
        """Edge types rule 2 follows target-to-source when an assumption falls."""
        return frozenset(t for t, d in self.invalidation.items() if d == "backward")

    @property
    def forward(self) -> FrozenSet[str]:
        """Edge types rule 2 follows source-to-target when an assumption falls."""
        return frozenset(t for t, d in self.invalidation.items() if d == "forward")

    @property
    def propagating(self) -> FrozenSet[str]:
        """Every edge type an invalidation can travel along, either direction."""
        return self.backward | self.forward

    def check_edge(self, edge_type: str, src_type: str, dst_type: str) -> None:
        if edge_type not in self.edge_types:
            raise OntologyError(
                f"no such edge type {edge_type!r}; GRAPH.md defines "
                f"{sorted(self.edge_types)}"
            )
        legal = self.signatures[edge_type]
        if (src_type, dst_type) not in legal:
            pretty = ", ".join(f"{a}->{b}" for a, b in sorted(legal))
            raise OntologyError(
                f"{src_type}-[{edge_type}]->{dst_type} is not a legal pair; "
                f"GRAPH.md allows {pretty}"
            )

    def check_node(self, node_type: str) -> None:
        if node_type not in self.node_types:
            raise OntologyError(
                f"no such node type {node_type!r}; GRAPH.md defines "
                f"{sorted(self.node_types)}"
            )


def _cells(line: str) -> list[str] | None:
    m = _ROW.match(line.strip())
    if not m:
        return None
    return [c.strip() for c in m.group("cells").split("|")]


def _is_rule_row(cells: Iterable[str]) -> bool:
    return all(set(c) <= set("-: ") and c for c in cells)


def _names(cell: str) -> FrozenSet[str]:
    """Parse a comma-separated, backtick-wrapped cell into a set of names.

    Used for the ``Must carry`` column: ``` `canonical`, `aliases` ``` becomes
    ``{"canonical", "aliases"}``, and an empty cell is a legitimate "carries
    nothing beyond the base record" rather than a parse failure.
    """
    return frozenset(
        name.strip().strip("`") for name in cell.split(",") if name.strip().strip("`")
    )


def load(path: Path | str = ONTOLOGY_FILE) -> Ontology:
    """Parse the node-type and edge-type tables out of GRAPH.md."""
    text = Path(path).read_text(encoding="utf-8")
    section = None
    node_types: set[str] = set()
    must_carry: Dict[str, FrozenSet[str]] = {}
    signatures: Dict[str, FrozenSet[Tuple[str, str]]] = {}
    invalidation: Dict[str, str] = {}

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            section = stripped[3:].strip().lower()
            continue
        cells = _cells(line)
        if not cells or _is_rule_row(cells):
            continue
        head = cells[0].strip("`").lower()
        if section == "node types":
            if head in {"type", ""}:
                continue
            symbol = cells[1].strip("`").strip() if len(cells) > 1 else ""
            if not symbol:
                raise OntologyError(
                    f"node type {cells[0]!r} has no Symbol column"
                )
            if symbol != head:
                raise OntologyError(
                    f"node type {cells[0]!r}: Symbol column is {symbol!r}, which "
                    f"does not match Type {cells[0]!r} lowercased ({head!r})"
                )
            node_types.add(symbol)
            must_carry[symbol] = _names(cells[3]) if len(cells) > 3 else frozenset()
        elif section == "edge types":
            if head in {"edge", ""}:
                continue
            pairs = frozenset(
                (m.group("src"), m.group("dst")) for m in _PAIR.finditer(cells[1])
            )
            if pairs:
                signatures[head] = pairs
            direction = cells[2].strip("`").strip().lower() if len(cells) > 2 else ""
            if direction not in _DIRECTIONS:
                raise OntologyError(
                    f"edge type {head!r}: Invalidation column is {direction!r}; "
                    f"must be one of {sorted(_DIRECTIONS)}"
                )
            invalidation[head] = direction

    if not node_types or not signatures:
        raise OntologyError(f"{path} did not yield a usable ontology")

    unknown = {
        t
        for pairs in signatures.values()
        for pair in pairs
        for t in pair
        if t not in node_types
    }
    if unknown:
        raise OntologyError(f"edge table references undeclared node types: {sorted(unknown)}")

    return Ontology(
        node_types=frozenset(node_types),
        edge_types=frozenset(signatures),
        signatures=signatures,
        invalidation=dict(invalidation),
        must_carry=must_carry,
    )


#: The live ontology. Imported by the graph, the pipeline and the linter.
ONTOLOGY = load()
