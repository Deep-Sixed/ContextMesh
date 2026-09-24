"""GRAPH.md is the schema, so the schema is what these tests check."""

import tempfile
import unittest
from pathlib import Path

from contextmesh.decisions import DecisionLog
from contextmesh.graph import ContextGraph
from contextmesh.model import EdgeType, NodeType, Provenance
from contextmesh.ontology import ONTOLOGY, ONTOLOGY_FILE, OntologyError, load


class OntologyFileTest(unittest.TestCase):
    def test_parses_every_declared_node_type(self):
        self.assertEqual(
            ONTOLOGY.node_types,
            {t.value for t in NodeType},
            "GRAPH.md and NodeType have drifted apart",
        )

    def test_parses_every_declared_edge_type(self):
        self.assertEqual(
            ONTOLOGY.edge_types,
            {t.value for t in EdgeType},
            "GRAPH.md and EdgeType have drifted apart",
        )

    def test_every_edge_type_has_at_least_one_legal_pair(self):
        for edge_type, pairs in ONTOLOGY.signatures.items():
            self.assertTrue(pairs, f"{edge_type} declares no legal pairs")

    def test_reload_is_stable(self):
        self.assertEqual(load().signatures, ONTOLOGY.signatures)

    def test_symbol_column_is_what_populates_node_types(self):
        """Every parsed node type is the Symbol column, not Type lowercased.

        For the real file these currently agree, so this only proves the
        parser reads the right column; ``MalformedFileTest`` below proves it
        actually checks the two against each other.
        """
        self.assertEqual(ONTOLOGY.node_types, {"entity", "claim", "source",
                                                 "decision", "assumption", "evidence"})


class InvalidationDirectionTest(unittest.TestCase):
    """GRAPH.md rule 2, now machine-readable as the edge table's own column."""

    def test_matches_rule_2_exactly(self):
        self.assertEqual(ONTOLOGY.backward, frozenset({"depends_on", "derived_from"}))
        self.assertEqual(ONTOLOGY.forward, frozenset({"produces"}))
        self.assertEqual(
            ONTOLOGY.propagating, frozenset({"depends_on", "derived_from", "produces"})
        )

    def test_every_other_edge_type_is_declared_none(self):
        non_propagating = ONTOLOGY.edge_types - ONTOLOGY.propagating
        self.assertEqual(
            non_propagating,
            {"mentions", "cites", "contradicts", "supports", "supersedes",
             "justified_by", "resolves_to"},
        )
        for edge_type in non_propagating:
            self.assertEqual(ONTOLOGY.invalidation[edge_type], "none")

    def test_every_edge_type_has_a_direction(self):
        self.assertEqual(set(ONTOLOGY.invalidation), ONTOLOGY.edge_types)

    def test_assumptions_module_reads_the_parsed_ontology_not_a_copy(self):
        """The drift this closes: BACKWARD/FORWARD used to be restated by hand."""
        from contextmesh import assumptions

        self.assertEqual(assumptions.BACKWARD, ONTOLOGY.backward)
        self.assertEqual(assumptions.FORWARD, ONTOLOGY.forward)
        self.assertEqual(assumptions.PROPAGATING, ONTOLOGY.propagating)

    def test_blast_radius_follows_the_graphs_own_ontology_not_the_module_default(self):
        """A graph built against a non-default Ontology invalidates by *its*
        Invalidation column, not by the process-global ``ONTOLOGY`` that
        ``assumptions.BACKWARD``/``FORWARD`` happen to equal today.

        ``depends_on`` is declared ``backward`` in the real file; this loads a
        variant where it is ``none`` and proves a decision standing on a
        rejected assumption through ``depends_on`` is *not* pulled into the
        blast radius when the graph's own ontology says that edge does not
        propagate — which only holds if ``blast_radius`` reads
        ``graph.ontology`` at traversal time rather than the module import.
        """
        from contextmesh.assumptions import AssumptionLedger

        text = Path(ONTOLOGY_FILE).read_text(encoding="utf-8").replace(
            "| `depends_on` | decision→assumption, decision→decision, "
            "claim→assumption | backward |",
            "| `depends_on` | decision→assumption, decision→decision, "
            "claim→assumption | none |",
            1,
        )
        self.assertNotEqual(text, Path(ONTOLOGY_FILE).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "GRAPH.md"
            path.write_text(text, encoding="utf-8")
            custom = load(path)
        self.assertEqual(custom.invalidation["depends_on"], "none")

        graph = ContextGraph(ontology=custom)
        ledger = AssumptionLedger(graph)
        assumption = ledger.assume("Shard count grows linearly with corpus size")
        source = graph.add_node(
            NodeType.SOURCE, "Doc", attrs={"origin": "x", "retrieved_at": "x"}
        )
        decision = DecisionLog(graph).decide(
            "Rebuild the index", "x", source_id=source.id, assumptions=[assumption.id]
        )

        radius = ledger.blast_radius(assumption.id)
        self.assertNotIn(decision.id, radius)


class MustCarryTest(unittest.TestCase):
    """GRAPH.md's Must carry column, parsed and enforced at the write boundary."""

    def test_parses_every_node_types_must_carry_set(self):
        self.assertEqual(
            ONTOLOGY.must_carry,
            {
                "entity": frozenset({"canonical", "aliases"}),
                "claim": frozenset({"provenance"}),
                "source": frozenset({"origin", "retrieved_at"}),
                "decision": frozenset({"rationale", "provenance"}),
                "assumption": frozenset({"status", "version"}),
                "evidence": frozenset({"kind"}),
            },
        )

    def test_a_node_missing_a_must_carry_attr_is_refused(self):
        graph = ContextGraph()
        with self.assertRaisesRegex(OntologyError, "missing"):
            graph.add_node(NodeType.SOURCE, "Undocumented")

    def test_a_node_missing_only_one_of_two_must_carry_names_is_still_refused(self):
        graph = ContextGraph()
        with self.assertRaisesRegex(OntologyError, "retrieved_at"):
            graph.add_node(NodeType.SOURCE, "Half documented", attrs={"origin": "x"})

    def test_must_carry_is_satisfied_by_a_node_field_not_only_by_attrs(self):
        """``provenance`` is Must carry for claim, and it is a ``Node`` field."""
        graph = ContextGraph()
        source = graph.add_node(
            NodeType.SOURCE, "Doc", attrs={"origin": "x", "retrieved_at": "x"}
        )
        claim = graph.add_node(
            NodeType.CLAIM, "A claim", provenance=Provenance(source_id=source.id)
        )
        self.assertNotIn("provenance", claim.attrs)
        self.assertIsNotNone(claim.provenance)

    def test_a_node_field_cannot_be_satisfied_by_shadowing_it_in_attrs(self):
        """GRAPH.md: ``provenance`` is a field, not an attribute.

        ``attrs={"provenance": None}`` must not give a caller a second,
        easier route to satisfying Must carry while the real ``Node.provenance``
        field stays unset — that would mean two answers to "does this claim
        have provenance", one of them a plain dict entry nothing else reads.
        """
        graph = ContextGraph()
        with self.assertRaisesRegex(OntologyError, "provenance"):
            graph.add_node(NodeType.CLAIM, "A claim", attrs={"provenance": None})

    def test_presence_is_enough_even_if_the_value_is_not_well_formed(self):
        """GRAPH.md: `retrieved_at="at plan time"` satisfies presence, not dates."""
        graph = ContextGraph()
        node = graph.add_node(
            NodeType.SOURCE,
            "Runner-minted",
            attrs={"origin": "runner", "retrieved_at": "at plan time"},
        )
        self.assertEqual(node.attrs["retrieved_at"], "at plan time")

    def test_a_snapshot_missing_a_must_carry_field_fails_closed_on_load(self):
        """Every node-construction path is add_node, restoring one included."""
        graph = ContextGraph()
        graph.add_node(
            NodeType.SOURCE, "Doc", attrs={"origin": "x", "retrieved_at": "x"}
        )
        payload = graph.to_dict()
        del payload["nodes"][0]["attrs"]["retrieved_at"]
        with self.assertRaises(OntologyError):
            ContextGraph.from_dict(payload)


class MalformedFileTest(unittest.TestCase):
    """A hand-edited GRAPH.md that breaks its own contract fails to load."""

    def _load(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "GRAPH.md"
            path.write_text(text, encoding="utf-8")
            return load(path)

    def _real_text(self):
        return Path(ONTOLOGY_FILE).read_text(encoding="utf-8")

    def test_symbol_disagreeing_with_type_fails_closed(self):
        text = self._real_text().replace(
            "| Entity | `entity` |", "| Entity | `entityx` |", 1
        )
        self.assertNotEqual(text, self._real_text())
        with self.assertRaisesRegex(OntologyError, "Symbol"):
            self._load(text)

    def test_an_empty_symbol_cell_fails_closed(self):
        text = self._real_text().replace("| Entity | `entity` |", "| Entity |  |", 1)
        self.assertNotEqual(text, self._real_text())
        with self.assertRaisesRegex(OntologyError, "Symbol"):
            self._load(text)

    def test_an_invalid_invalidation_value_fails_closed(self):
        text = self._real_text().replace(
            "| `mentions` | source→entity, claim→entity | none |",
            "| `mentions` | source→entity, claim→entity | sideways |",
            1,
        )
        self.assertNotEqual(text, self._real_text())
        with self.assertRaisesRegex(OntologyError, "Invalidation"):
            self._load(text)

    def test_an_empty_invalidation_cell_fails_closed(self):
        text = self._real_text().replace(
            "| `mentions` | source→entity, claim→entity | none |",
            "| `mentions` | source→entity, claim→entity |  |",
            1,
        )
        self.assertNotEqual(text, self._real_text())
        with self.assertRaisesRegex(OntologyError, "Invalidation"):
            self._load(text)


class TypecheckTest(unittest.TestCase):
    def setUp(self):
        self.graph = ContextGraph()
        self.source = self.graph.add_node(
            NodeType.SOURCE,
            "RFC 014",
            attrs={"origin": "fixture", "retrieved_at": "fixture"},
        )
        self.entity = self.graph.add_node(
            NodeType.ENTITY,
            "pgvector",
            attrs={"canonical": "pgvector", "aliases": []},
        )
        self.claim = self.graph.add_node(
            NodeType.CLAIM,
            "pgvector stores embeddings",
            provenance=Provenance(source_id=self.source.id),
        )

    def test_legal_edge_is_accepted(self):
        edge = self.graph.add_edge(self.claim.id, EdgeType.MENTIONS, self.entity.id)
        self.assertIs(edge.type, EdgeType.MENTIONS)

    def test_illegal_pair_is_refused(self):
        with self.assertRaises(OntologyError):
            self.graph.add_edge(self.entity.id, EdgeType.MENTIONS, self.source.id)

    def test_unknown_endpoint_is_refused(self):
        with self.assertRaises(OntologyError):
            self.graph.add_edge(self.claim.id, EdgeType.MENTIONS, "entity:missing")

    def test_self_edge_is_refused(self):
        with self.assertRaises(OntologyError):
            self.graph.add_edge(self.claim.id, EdgeType.CITES, self.claim.id)

    def test_untyped_edges_are_structurally_impossible(self):
        self.graph.add_edge(self.claim.id, EdgeType.MENTIONS, self.entity.id)
        self.graph.add_edge(self.claim.id, EdgeType.DERIVED_FROM, self.source.id)
        self.assertEqual(self.graph.untyped_edges, 0)

    def test_duplicate_edge_merges_rather_than_multiplies(self):
        first = self.graph.add_edge(self.claim.id, EdgeType.MENTIONS, self.entity.id)
        second = self.graph.add_edge(self.claim.id, EdgeType.MENTIONS, self.entity.id)
        self.assertIs(first, second)
        self.assertEqual(len(self.graph.edges), 1)
        self.assertEqual(second.weight, 2.0)

    def test_empty_graph_is_truthy(self):
        # ``graph or ContextGraph()`` must not swap in a fresh graph.
        self.assertTrue(ContextGraph())


class SchemaShipsInsideThePackageTest(unittest.TestCase):
    """0.1.1 installed the schema as a loose top-level site-packages/GRAPH.md
    (package-data "../GRAPH.md"), where another distribution shipping or
    removing a file of that name would break `import contextmesh`."""

    def test_the_schema_is_read_from_inside_the_package(self):
        import contextmesh

        package = Path(contextmesh.__file__).resolve().parent
        self.assertEqual(Path(ONTOLOGY_FILE).parent, package)
        self.assertTrue(Path(ONTOLOGY_FILE).is_file())

    def test_package_data_does_not_reach_outside_the_package(self):
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        text = pyproject.read_text(encoding="utf-8")
        section = text.split("[tool.setuptools.package-data]", 1)[1].split("\n[", 1)[0]
        self.assertNotIn("..", section)
        self.assertIn('"GRAPH.md"', section)


if __name__ == "__main__":
    unittest.main()
