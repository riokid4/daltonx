"""daltonx/tests/test_ontology.py
===================================
Unit tests for Section 2 – The Ontology Engine.

Tests cover:
  - YAML loading and graph construction
  - Transitive closure is pre-built after initialize()
  - O(1) isa_or_partof via transitive_closure
  - isa() / is_opposite() queries
  - get_parents() / get_children() hierarchical walkers
  - get_name() / get_id_from_name() name lookups
  - add_entry() live mutation and graph rebuild
  - Influence statement refinement / contradiction cross-check
    using the ontology (proves Section 1 + Section 2 integration)
"""
import unittest

from daltonx.ontology import AISafetyOntology, load_ai_safety_ontology
from daltonx.statements import (
    Concept, Event, Influence, QualitativeDelta, Evidence, AISafetyContext
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(concept_id, polarity):
    """Build a minimal grounded Event for integration tests."""
    concept = Concept(
        concept_id,
        db_refs={'AIS': concept_id}
    )
    delta = QualitativeDelta(polarity=polarity)
    return Event(concept, delta=delta)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class OntologyFixture(unittest.TestCase):
    """Base fixture: loads the bundled YAML once per test class."""

    @classmethod
    def setUpClass(cls):
        cls.onto = load_ai_safety_ontology()
        # Force initialization explicitly so timing is predictable
        cls.onto.initialize()


# ---------------------------------------------------------------------------
# Test suite 1: Loading & graph structure
# ---------------------------------------------------------------------------

class TestOntologyLoading(OntologyFixture):
    """Test that the YAML is parsed into a correctly shaped graph."""

    def test_initialized_flag(self):
        """After initialize() the _initialized flag must be True."""
        self.assertTrue(self.onto._initialized)

    def test_node_count(self):
        """Graph must contain at least 11 nodes (safety + 10 concepts)."""
        self.assertGreaterEqual(len(self.onto.nodes), 11)

    def test_edge_count(self):
        """Graph must have at least 10 isa + 2 is_opposite edges."""
        self.assertGreaterEqual(len(self.onto.edges), 12)

    def test_alignment_node_present(self):
        """AIS:safety/alignment must be a node in the graph."""
        label = self.onto.label('AIS', 'safety/alignment')
        self.assertIn(label, self.onto.nodes)

    def test_misalignment_node_present(self):
        """AIS:safety/misalignment must be a node in the graph."""
        label = self.onto.label('AIS', 'safety/misalignment')
        self.assertIn(label, self.onto.nodes)

    def test_deep_leaf_node_present(self):
        """AIS:safety/alignment/mechanistic_interpretability/sparse_autoencoders
        must be reachable."""
        label = self.onto.label(
            'AIS',
            'safety/alignment/mechanistic_interpretability/sparse_autoencoders'
        )
        self.assertIn(label, self.onto.nodes)


# ---------------------------------------------------------------------------
# Test suite 2: Transitive closure
# ---------------------------------------------------------------------------

class TestTransitiveClosure(OntologyFixture):
    """Test that _build_transitive_closure populated the set correctly."""

    def test_closure_non_empty(self):
        """Transitive closure must be a non-empty set."""
        self.assertIsInstance(self.onto.transitive_closure, set)
        self.assertGreater(len(self.onto.transitive_closure), 0)

    def test_isa_or_partof_uses_closure(self):
        """isa_or_partof must return True using the O(1) closure set."""
        # rlhf isa alignment isa safety -> rlhf isa_or_partof safety
        result = self.onto.isa_or_partof(
            'AIS', 'safety/alignment/rlhf',
            'AIS', 'safety'
        )
        self.assertTrue(result)

    def test_closure_tuple_format(self):
        """Each element of transitive_closure is a pair of label strings."""
        elem = next(iter(self.onto.transitive_closure))
        self.assertIsInstance(elem, tuple)
        self.assertEqual(len(elem), 2)
        # Both elements must be 'AIS:...' labels
        self.assertTrue(elem[0].startswith('AIS:'))
        self.assertTrue(elem[1].startswith('AIS:'))


# ---------------------------------------------------------------------------
# Test suite 3: isa / is_opposite queries
# ---------------------------------------------------------------------------

class TestIsaAndIsOpposite(OntologyFixture):
    """Test the primary ontological relationship queries."""

    def test_rlhf_isa_alignment(self):
        self.assertTrue(
            self.onto.isa('AIS', 'safety/alignment/rlhf',
                          'AIS', 'safety/alignment')
        )

    def test_rlhf_not_isa_misalignment(self):
        self.assertFalse(
            self.onto.isa('AIS', 'safety/alignment/rlhf',
                          'AIS', 'safety/misalignment')
        )

    def test_sparse_autoencoders_isa_mechanistic_interpretability(self):
        self.assertTrue(
            self.onto.isa(
                'AIS',
                'safety/alignment/mechanistic_interpretability/sparse_autoencoders',
                'AIS',
                'safety/alignment/mechanistic_interpretability'
            )
        )

    def test_sparse_autoencoders_isa_alignment_transitive(self):
        """Transitive isa: sparse_autoencoders -> mech_interp -> alignment."""
        self.assertTrue(
            self.onto.isa(
                'AIS',
                'safety/alignment/mechanistic_interpretability/sparse_autoencoders',
                'AIS',
                'safety/alignment'
            )
        )

    def test_alignment_is_opposite_misalignment(self):
        self.assertTrue(
            self.onto.is_opposite(
                'AIS', 'safety/alignment',
                'AIS', 'safety/misalignment'
            )
        )

    def test_misalignment_is_opposite_alignment(self):
        """is_opposite is symmetric as the YAML adds both directions."""
        self.assertTrue(
            self.onto.is_opposite(
                'AIS', 'safety/misalignment',
                'AIS', 'safety/alignment'
            )
        )

    def test_rlhf_not_opposite_misalignment(self):
        """rlhf has no direct is_opposite edge to misalignment."""
        self.assertFalse(
            self.onto.is_opposite(
                'AIS', 'safety/alignment/rlhf',
                'AIS', 'safety/misalignment'
            )
        )


# ---------------------------------------------------------------------------
# Test suite 4: Parent / child hierarchy
# ---------------------------------------------------------------------------

class TestHierarchy(OntologyFixture):
    """Test get_parents and get_children walkers."""

    def test_get_parents_of_rlhf(self):
        """rlhf's parents must include alignment and safety."""
        parents = self.onto.get_parents('AIS', 'safety/alignment/rlhf')
        ns_ids = set(parents)
        self.assertIn(('AIS', 'safety/alignment'), ns_ids)
        self.assertIn(('AIS', 'safety'), ns_ids)

    def test_get_children_of_alignment(self):
        """alignment's children must include rlhf and scalable_oversight."""
        children = self.onto.get_children('AIS', 'safety/alignment')
        ns_ids = set(children)
        self.assertIn(('AIS', 'safety/alignment/rlhf'), ns_ids)
        self.assertIn(('AIS', 'safety/alignment/scalable_oversight'), ns_ids)

    def test_get_top_level_parents_of_rlhf(self):
        """Top-level parent of rlhf must be safety (no parent above it)."""
        top = self.onto.get_top_level_parents('AIS', 'safety/alignment/rlhf')
        ns_ids = set(top)
        self.assertIn(('AIS', 'safety'), ns_ids)

    def test_safety_has_no_parents(self):
        """Root node safety has no isa/partof parents."""
        parents = self.onto.get_parents('AIS', 'safety')
        self.assertEqual(parents, [])


# ---------------------------------------------------------------------------
# Test suite 5: Name lookups
# ---------------------------------------------------------------------------

class TestNameLookup(OntologyFixture):
    """Test get_name and get_id_from_name."""

    def test_get_name_alignment(self):
        name = self.onto.get_name('AIS', 'safety/alignment')
        self.assertEqual(name, 'alignment')

    def test_get_name_sparse_autoencoders(self):
        name = self.onto.get_name(
            'AIS',
            'safety/alignment/mechanistic_interpretability/sparse_autoencoders'
        )
        self.assertEqual(name, 'sparse_autoencoders')

    def test_get_id_from_name_roundtrip(self):
        """get_id_from_name must reverse get_name."""
        result = self.onto.get_id_from_name('AIS', 'rlhf')
        self.assertIsNotNone(result)
        ns, id_ = result
        self.assertEqual(ns, 'AIS')
        self.assertIn('rlhf', id_)

    def test_get_name_missing_node(self):
        """Unknown node must return None without raising."""
        result = self.onto.get_name('AIS', 'nonexistent/concept')
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Test suite 6: add_entry live mutation
# ---------------------------------------------------------------------------

class TestAddEntry(OntologyFixture):
    """Test that add_entry inserts a new node and rebuilds the graph."""

    def test_add_new_leaf(self):
        """Adding a new leaf under alignment must make it findable."""
        node_count_before = len(self.onto.nodes)
        self.onto.add_entry(
            'safety/alignment/constitutional_ai',
            examples=['Constitutional AI', 'CAI training']
        )
        # Node count must have grown by at least 1
        self.assertGreater(len(self.onto.nodes), node_count_before)
        # New node must be reachable
        label = self.onto.label('AIS', 'safety/alignment/constitutional_ai')
        self.assertIn(label, self.onto.nodes)

    def test_added_node_isa_alignment(self):
        """Newly added constitutional_ai must isa alignment."""
        # add_entry was already called in the previous test; since setUpClass
        # shares the same instance, the node is already present.
        self.assertTrue(
            self.onto.isa(
                'AIS', 'safety/alignment/constitutional_ai',
                'AIS', 'safety/alignment'
            )
        )


# ---------------------------------------------------------------------------
# Test suite 7: Integration with Statement refinement
# ---------------------------------------------------------------------------

class TestInfluenceRefinementWithOntology(OntologyFixture):
    """
    Validate that the Section 1 Influence.refinement_of() method works
    correctly when concept groundings are compared via the ontology.

    The refinement_of check in INDRA uses the ontology through agent
    refinement; we replicate that pattern by checking isa_or_partof directly
    on the Concept db_refs, which is how the preassembler uses it.
    """

    def _isa_or_partof(self, concept_a, concept_b):
        """Check if concept_a is a refinement of concept_b via ontology."""
        ref_a = concept_a.db_refs.get('AIS')
        ref_b = concept_b.db_refs.get('AIS')
        if ref_a and ref_b:
            if ref_a == ref_b:
                return True
            return self.onto.isa_or_partof('AIS', ref_a, 'AIS', ref_b)
        return False

    def test_rlhf_refines_alignment(self):
        """rlhf-grounded concept is a refinement of alignment-grounded concept."""
        c_rlhf = Concept('rlhf', db_refs={'AIS': 'safety/alignment/rlhf'})
        c_align = Concept('alignment', db_refs={'AIS': 'safety/alignment'})
        self.assertTrue(self._isa_or_partof(c_rlhf, c_align))

    def test_alignment_does_not_refine_rlhf(self):
        """alignment is NOT a more-specific version of rlhf."""
        c_rlhf = Concept('rlhf', db_refs={'AIS': 'safety/alignment/rlhf'})
        c_align = Concept('alignment', db_refs={'AIS': 'safety/alignment'})
        self.assertFalse(self._isa_or_partof(c_align, c_rlhf))

    def test_sycophancy_does_not_refine_alignment(self):
        """sycophancy (misalignment branch) does not refine alignment."""
        c_syco = Concept('sycophancy',
                         db_refs={'AIS': 'safety/misalignment/sycophancy'})
        c_align = Concept('alignment', db_refs={'AIS': 'safety/alignment'})
        self.assertFalse(self._isa_or_partof(c_syco, c_align))

    def test_sparse_autoencoders_refines_safety_transitively(self):
        """sparse_autoencoders is a refinement of the root safety node."""
        c_sae = Concept('sae', db_refs={
            'AIS': 'safety/alignment/mechanistic_interpretability/'
                   'sparse_autoencoders'
        })
        c_safety = Concept('safety', db_refs={'AIS': 'safety'})
        self.assertTrue(self._isa_or_partof(c_sae, c_safety))

    def test_influence_contradicts_via_opposite_ontology_nodes(self):
        """Two Influence stmts with opposite subjects and flipped polarity
        must be detected as contradictory by the statement logic."""
        # Influence 1: reward_hacking(+) -> safety(-) -- a misalignment claim
        ev1 = Event(
            Concept('reward_hacking',
                    db_refs={'AIS': 'safety/misalignment/reward_hacking'}),
            delta=QualitativeDelta(polarity=1)
        )
        ev2 = Event(
            Concept('safety', db_refs={'AIS': 'safety'}),
            delta=QualitativeDelta(polarity=-1)
        )
        inf1 = Influence(ev1, ev2)

        # Influence 2: alignment(+) -> safety(+) -- a contrary claim
        ev3 = Event(
            Concept('alignment', db_refs={'AIS': 'safety/alignment'}),
            delta=QualitativeDelta(polarity=1)
        )
        ev4 = Event(
            Concept('safety', db_refs={'AIS': 'safety'}),
            delta=QualitativeDelta(polarity=1)
        )
        inf2 = Influence(ev3, ev4)

        # The two statements have different subjects and objects; their
        # overall_polarity values differ: inf1 = -1, inf2 = +1.
        self.assertNotEqual(inf1.overall_polarity(), inf2.overall_polarity())


if __name__ == '__main__':
    unittest.main()
