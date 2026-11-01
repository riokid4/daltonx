"""Unit tests for Section 7 – CAGAssembler (daltonx/exporters/cag.py).

Tests verify:
1. make_model() converts Influence statements into a networkx.MultiDiGraph
   with correct nodes and signed edges.
2. export_to_cytoscapejs() produces the expected CytoscapeJS dict structure.
3. _node_name() uses AIS grounding when threshold is set, otherwise concept.name.
4. Polarity logic: same polarity → (circle, green); opposite → (tee, maroon);
   no polarity → (triangle, maroon, dotted).
5. print_tsv() writes a valid TSV with header and data rows.
6. Statements that are not Influence are silently ignored.
"""

import json
import os
import sys
import tempfile
import unittest

# Make sure the repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from daltonx.statements import (
    Concept, Event, Evidence, Influence, QualitativeDelta,
)
from daltonx.exporters.cag import CAGAssembler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_influence(subj_name, obj_name,
                    subj_pol=1, obj_pol=1,
                    subj_adj=None, obj_adj=None,
                    source_api='ais', text='test evidence',
                    subj_grounding=None, obj_grounding=None):
    """Build a minimal Influence statement for testing."""
    subj_concept = Concept(subj_name, db_refs={'AIS': subj_grounding or []})
    obj_concept = Concept(obj_name, db_refs={'AIS': obj_grounding or []})
    subj_delta = QualitativeDelta(polarity=subj_pol, adjectives=subj_adj or [])
    obj_delta = QualitativeDelta(polarity=obj_pol, adjectives=obj_adj or [])
    subj_event = Event(subj_concept, delta=subj_delta)
    obj_event = Event(obj_concept, delta=obj_delta)
    ev = Evidence(source_api=source_api, text=text,
                  annotations={'subj_text': subj_name, 'obj_text': obj_name})
    return Influence(subj_event, obj_event, evidence=[ev])


def _make_influence_no_polarity(subj_name, obj_name):
    """Build an Influence with no polarity on either side."""
    subj_concept = Concept(subj_name)
    obj_concept = Concept(obj_name)
    subj_delta = QualitativeDelta(polarity=None)
    obj_delta = QualitativeDelta(polarity=None)
    return Influence(
        Event(subj_concept, delta=subj_delta),
        Event(obj_concept, delta=obj_delta),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCAGMakeModel(unittest.TestCase):
    """Tests for CAGAssembler.make_model()."""

    def test_returns_multidigraph(self):
        """make_model() returns a networkx.MultiDiGraph."""
        import networkx as nx
        stmt = _make_influence('capability', 'safety')
        asm = CAGAssembler([stmt])
        cag = asm.make_model()
        self.assertIsInstance(cag, nx.MultiDiGraph)

    def test_nodes_present(self):
        """Both subject and object concepts appear as nodes."""
        stmt = _make_influence('alignment research', 'model safety')
        asm = CAGAssembler([stmt])
        cag = asm.make_model()
        node_ids = set(cag.nodes())
        self.assertIn('Alignment research', node_ids)
        self.assertIn('Model safety', node_ids)

    def test_edge_present(self):
        """An edge exists between subject and object nodes."""
        stmt = _make_influence('funding', 'research')
        asm = CAGAssembler([stmt])
        cag = asm.make_model()
        self.assertTrue(cag.has_edge('Funding', 'Research'))

    def test_same_polarity_edge_style(self):
        """Same polarity (both +1) → circle arrow, green colour, solid line."""
        stmt = _make_influence('a', 'b', subj_pol=1, obj_pol=1)
        asm = CAGAssembler([stmt])
        cag = asm.make_model()
        edge_data = list(cag.edges(data=True))[0][2]
        self.assertEqual(edge_data['targetArrowShape'], 'circle')
        self.assertEqual(edge_data['linecolor'], 'green')
        self.assertEqual(edge_data['linestyle'], 'solid')

    def test_opposite_polarity_edge_style(self):
        """Opposite polarity (+1 / -1) → tee arrow, maroon colour, solid line."""
        stmt = _make_influence('capability', 'risk', subj_pol=1, obj_pol=-1)
        asm = CAGAssembler([stmt])
        cag = asm.make_model()
        edge_data = list(cag.edges(data=True))[0][2]
        self.assertEqual(edge_data['targetArrowShape'], 'tee')
        self.assertEqual(edge_data['linecolor'], 'maroon')
        self.assertEqual(edge_data['linestyle'], 'solid')

    def test_no_polarity_edge_style(self):
        """No polarity on either side → triangle arrow, maroon, dotted line."""
        stmt = _make_influence_no_polarity('oversight', 'deployment')
        asm = CAGAssembler([stmt])
        cag = asm.make_model()
        edge_data = list(cag.edges(data=True))[0][2]
        self.assertEqual(edge_data['targetArrowShape'], 'triangle')
        self.assertEqual(edge_data['linecolor'], 'maroon')
        self.assertEqual(edge_data['linestyle'], 'dotted')

    def test_node_simulable_flag(self):
        """Nodes are marked simulable=True when both polarities are set."""
        stmt = _make_influence('x', 'y', subj_pol=1, obj_pol=-1)
        asm = CAGAssembler([stmt])
        cag = asm.make_model()
        for _, node_data in cag.nodes(data=True):
            self.assertTrue(node_data['simulable'])

    def test_node_not_simulable_when_no_polarity(self):
        """Nodes are marked simulable=False when polarity is absent."""
        stmt = _make_influence_no_polarity('p', 'q')
        asm = CAGAssembler([stmt])
        cag = asm.make_model()
        for _, node_data in cag.nodes(data=True):
            self.assertFalse(node_data['simulable'])

    def test_non_influence_stmts_ignored(self):
        """Statements that are not Influence instances are silently skipped."""
        from daltonx.statements import Statement
        stmt = _make_influence('a', 'b')
        asm = CAGAssembler([stmt])
        # Add a raw Statement (no subj/obj) — should be ignored
        asm.statements.append(object())
        # Should not raise
        cag = asm.make_model()
        self.assertEqual(len(list(cag.nodes())), 2)

    def test_multiple_statements(self):
        """Multiple Influence statements produce the correct graph topology."""
        s1 = _make_influence('compute', 'capability')
        s2 = _make_influence('capability', 'risk')
        s3 = _make_influence('regulation', 'risk', subj_pol=1, obj_pol=-1)
        asm = CAGAssembler([s1, s2, s3])
        cag = asm.make_model()
        self.assertGreaterEqual(cag.number_of_nodes(), 4)
        self.assertEqual(cag.number_of_edges(), 3)

    def test_add_statements(self):
        """add_statements() extends the statement list."""
        s1 = _make_influence('a', 'b')
        s2 = _make_influence('c', 'd')
        asm = CAGAssembler([s1])
        asm.add_statements([s2])
        cag = asm.make_model()
        self.assertEqual(cag.number_of_edges(), 2)

    def test_belief_stored_on_edge(self):
        """Belief score from the statement is stored on the edge."""
        stmt = _make_influence('a', 'b')
        stmt.belief = 0.75
        asm = CAGAssembler([stmt])
        cag = asm.make_model()
        edge_data = list(cag.edges(data=True))[0][2]
        self.assertAlmostEqual(edge_data['belief'], 0.75)


class TestCAGNodeName(unittest.TestCase):
    """Tests for CAGAssembler._node_name()."""

    def test_uses_concept_name_by_default(self):
        """Without a grounding threshold, concept.name is returned."""
        concept = Concept('ai risk', db_refs={'AIS': [('AIS/risk', 0.95)]})
        asm = CAGAssembler()
        name = asm._node_name(concept)
        self.assertEqual(name, 'Ai risk')

    def test_uses_grounding_above_threshold(self):
        """When score exceeds threshold, last path fragment is used."""
        concept = Concept('ai risk', db_refs={'AIS': [('AIS/alignment/risk', 0.95)]})
        asm = CAGAssembler()
        asm.grounding_threshold = 0.5
        name = asm._node_name(concept)
        self.assertEqual(name, 'Risk')

    def test_falls_back_below_threshold(self):
        """When score is below threshold, falls back to concept.name."""
        concept = Concept('ai risk', db_refs={'AIS': [('AIS/alignment/risk', 0.2)]})
        asm = CAGAssembler()
        asm.grounding_threshold = 0.5
        name = asm._node_name(concept)
        self.assertEqual(name, 'Ai risk')

    def test_empty_grounding_list(self):
        """Empty grounding list still returns concept.name."""
        concept = Concept('unknown', db_refs={'AIS': []})
        asm = CAGAssembler()
        asm.grounding_threshold = 0.5
        name = asm._node_name(concept)
        self.assertEqual(name, 'Unknown')


class TestCAGExportCytoscapeJS(unittest.TestCase):
    """Tests for CAGAssembler.export_to_cytoscapejs()."""

    def setUp(self):
        s1 = _make_influence('oversight', 'safety', subj_pol=1, obj_pol=1)
        s2 = _make_influence('capability', 'risk', subj_pol=1, obj_pol=-1)
        self.asm = CAGAssembler([s1, s2])
        self.asm.make_model()

    def test_top_level_keys(self):
        """Result has 'nodes' and 'edges' keys."""
        result = self.asm.export_to_cytoscapejs()
        self.assertIn('nodes', result)
        self.assertIn('edges', result)

    def test_node_structure(self):
        """Each node has 'data' with 'id', 'simulable', 'tooltip'."""
        result = self.asm.export_to_cytoscapejs()
        for node in result['nodes']:
            self.assertIn('data', node)
            self.assertIn('id', node['data'])
            self.assertIn('simulable', node['data'])
            self.assertIn('tooltip', node['data'])

    def test_edge_structure(self):
        """Each edge has 'data' with expected keys."""
        result = self.asm.export_to_cytoscapejs()
        required = {
            'id', 'source', 'target', 'linestyle', 'linecolor',
            'targetArrowShape', 'subj_adjectives', 'subj_polarity',
            'obj_adjectives', 'obj_polarity', 'simulable',
        }
        for edge in result['edges']:
            self.assertIn('data', edge)
            for key in required:
                self.assertIn(key, edge['data'],
                              msg=f"Missing key '{key}' in edge data")

    def test_node_count(self):
        """Number of nodes in export matches graph."""
        result = self.asm.export_to_cytoscapejs()
        self.assertEqual(
            len(result['nodes']),
            self.asm.CAG.number_of_nodes(),
        )

    def test_edge_count(self):
        """Number of edges in export matches graph."""
        result = self.asm.export_to_cytoscapejs()
        self.assertEqual(
            len(result['edges']),
            self.asm.CAG.number_of_edges(),
        )

    def test_raise_before_make_model(self):
        """Calling export_to_cytoscapejs() before make_model() raises."""
        asm = CAGAssembler([_make_influence('a', 'b')])
        with self.assertRaises(RuntimeError):
            asm.export_to_cytoscapejs()

    def test_json_serialisable(self):
        """The CytoscapeJS export is fully JSON-serialisable."""
        result = self.asm.export_to_cytoscapejs()
        # Should not raise
        serialised = json.dumps(result)
        self.assertIsInstance(serialised, str)


class TestCAGPrintTSV(unittest.TestCase):
    """Tests for CAGAssembler.print_tsv()."""

    def test_tsv_written(self):
        """print_tsv() creates a file with header and at least one data row."""
        stmt = _make_influence('oversight', 'safety',
                               subj_pol=1, obj_pol=1,
                               source_api='ais', text='oversight increases safety')
        asm = CAGAssembler([stmt])
        asm.make_model()
        with tempfile.NamedTemporaryFile(mode='r', suffix='.tsv',
                                        delete=False) as fh:
            fname = fh.name
        try:
            asm.print_tsv(fname)
            with open(fname) as f:
                lines = f.readlines()
            self.assertGreater(len(lines), 1, "Expected header + at least one data row")
            # Verify header
            header_cols = lines[0].strip().split('\t')
            self.assertIn('Factor A Text', header_cols)
            self.assertIn('Factor B Text', header_cols)
        finally:
            os.unlink(fname)

    def test_tsv_columns(self):
        """TSV rows have same number of columns as header."""
        stmt = _make_influence('compute', 'capability',
                               source_api='ais', text='More compute increases capability')
        asm = CAGAssembler([stmt])
        asm.make_model()
        with tempfile.NamedTemporaryFile(mode='r', suffix='.tsv',
                                        delete=False) as fh:
            fname = fh.name
        try:
            asm.print_tsv(fname)
            with open(fname) as f:
                lines = f.readlines()
            n_cols = len(lines[0].rstrip('\r\n').split('\t'))
            for data_line in lines[1:]:
                self.assertEqual(len(data_line.rstrip('\r\n').split('\t')), n_cols)
        finally:
            os.unlink(fname)


class TestCAGEdgeCases(unittest.TestCase):
    """Edge-case behaviour of CAGAssembler."""

    def test_empty_statements(self):
        """Assembler with no statements produces an empty graph."""
        asm = CAGAssembler()
        cag = asm.make_model()
        self.assertEqual(cag.number_of_nodes(), 0)
        self.assertEqual(cag.number_of_edges(), 0)

    def test_default_grounding_ontology_is_ais(self):
        """grounding_ontology defaults to 'AIS'."""
        asm = CAGAssembler()
        self.assertEqual(asm.grounding_ontology, 'AIS')

    def test_make_model_resets_cag(self):
        """Calling make_model() twice resets the graph."""
        stmt = _make_influence('a', 'b')
        asm = CAGAssembler([stmt])
        asm.make_model()
        asm.make_model()  # second call
        self.assertEqual(asm.CAG.number_of_edges(), 1)

    def test_grounding_ontology_param_override(self):
        """make_model() with grounding_ontology param updates the attribute."""
        stmt = _make_influence('a', 'b')
        asm = CAGAssembler([stmt])
        asm.make_model(grounding_ontology='UN')
        self.assertEqual(asm.grounding_ontology, 'UN')

    def test_node_mods_stored(self):
        """Adjectives are stored in node 'mods' attribute."""
        stmt = _make_influence('a', 'b', subj_adj=['significantly'])
        asm = CAGAssembler([stmt])
        cag = asm.make_model()
        node_data = dict(cag.nodes(data=True))
        self.assertIn('significantly', node_data['A']['mods'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
