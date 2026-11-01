"""daltonx/tests/test_incremental.py
==================================
Unit tests for IncrementalAssembler and Assembly Operations (daltonx.assembly.incremental, operations).

Tests:
1. IncrementalAssembler initialization: builds refinement graph and calculates initial beliefs.
2. IncrementalAssembler add_statements duplicate: appends evidence to existing hash,
   reports it in AssemblyDelta.new_evidences, and boosts belief score.
3. IncrementalAssembler add_statements new refinement: adds statement to AssemblyDelta.new_stmts,
   forms new refinement edge in AssemblyDelta.new_refinements, and updates parent belief.
4. Pipeline operations: get_expanded_events_influences, remove_namespaces, set_positive_polarities,
   filter_out_long_words, merge_deltas, AssemblyPipeline execution.
"""

import unittest
from daltonx.statements import (
    Concept,
    Evidence,
    Event,
    QualitativeDelta,
    Influence,
)
from daltonx.ontology import load_ai_safety_ontology
from daltonx.assembly.incremental import (
    IncrementalAssembler,
    AssemblyDelta,
)
from daltonx.assembly.matches import location_matches_compositional
from daltonx.assembly.operations import (
    AssemblyPipeline,
    get_expanded_events_influences,
    remove_namespaces,
    remove_raw_grounding,
    set_positive_polarities,
    filter_out_long_words,
    merge_deltas,
    make_display_name,
)


def _make_influence(
    subj_name: str,
    subj_path: str,
    obj_name: str,
    obj_path: str,
    evidence=None,
    pol_subj: int = 1,
    pol_obj: int = -1
) -> Influence:
    c_subj = Concept(subj_name, db_refs={'AIS': subj_path, 'TEXT': subj_name})
    c_obj = Concept(obj_name, db_refs={'AIS': obj_path, 'TEXT': obj_name})
    return Influence(
        Event(c_subj, delta=QualitativeDelta(polarity=pol_subj)),
        Event(c_obj, delta=QualitativeDelta(polarity=pol_obj)),
        evidence=evidence if evidence else []
    )


class TestIncrementalAssembler(unittest.TestCase):
    """Test IncrementalAssembler initialization and continual updates."""

    @classmethod
    def setUpClass(cls):
        cls.onto = load_ai_safety_ontology()
        cls.onto.initialize()

    def setUp(self):
        # Initial Statement 1 (General): Alignment reduces Misalignment
        self.ev_gen = Evidence(
            source_api='paperqa',
            pmid='arxiv:2301.gen',
            text='Alignment techniques reduce misalignment.'
        )
        self.stmt_gen = _make_influence(
            'alignment', 'safety/alignment',
            'misalignment', 'safety/misalignment',
            evidence=[self.ev_gen]
        )

        # Initial Statement 2 (Specific): RLHF reduces Sycophancy
        self.ev_spec = Evidence(
            source_api='arxiv_extractor',
            pmid='arxiv:2302.spec',
            text='RLHF mitigates sycophancy.'
        )
        self.stmt_spec = _make_influence(
            'rlhf', 'safety/alignment/rlhf',
            'sycophancy', 'safety/misalignment/sycophancy',
            evidence=[self.ev_spec]
        )

    def test_initialization_and_beliefs(self):
        """IncrementalAssembler builds the graph and calculates initial beliefs."""
        ia = IncrementalAssembler(
            prepared_stmts=[self.stmt_gen, self.stmt_spec],
            ontology=self.onto,
            matches_fun=location_matches_compositional
        )

        # 2 unique statements by hash
        self.assertEqual(len(ia.stmts_by_hash), 2)
        gen_hash = self.stmt_gen.get_hash(matches_fun=location_matches_compositional)
        spec_hash = self.stmt_spec.get_hash(matches_fun=location_matches_compositional)

        self.assertIn(gen_hash, ia.stmts_by_hash)
        self.assertIn(spec_hash, ia.stmts_by_hash)

        # Refinement edge exists: parent (gen) -> child (spec)
        self.assertIn((gen_hash, spec_hash), ia.refinement_edges)
        self.assertTrue(ia.refinements_graph.has_edge(gen_hash, spec_hash))

        # Beliefs: child has 1 arxiv_extractor evidence: 1 - (0.06 + 0.18) = 0.76
        self.assertAlmostEqual(ia.beliefs[spec_hash], 0.76, places=4)

        # Parent accumulated evidence from child: paperqa (0.80) + arxiv_extractor (0.76) => 0.952
        self.assertGreater(ia.beliefs[gen_hash], 0.80)
        self.assertAlmostEqual(ia.beliefs[gen_hash], 0.952, places=4)

    def test_add_duplicate_statement_boosts_belief(self):
        """Adding a duplicate statement appends Evidence and increases belief."""
        ia = IncrementalAssembler(
            prepared_stmts=[self.stmt_gen, self.stmt_spec],
            ontology=self.onto,
            matches_fun=location_matches_compositional
        )
        spec_hash = self.stmt_spec.get_hash(matches_fun=location_matches_compositional)
        gen_hash = self.stmt_gen.get_hash(matches_fun=location_matches_compositional)

        initial_spec_belief = ia.beliefs[spec_hash]
        initial_gen_belief = ia.beliefs[gen_hash]

        # Duplicate statement of stmt_spec with second evidence from llm_reader
        ev_dup = Evidence(
            source_api='llm_reader',
            pmid='arxiv:2303.dup',
            text='Model evaluation shows RLHF decreases sycophantic behavior.'
        )
        dup_stmt = _make_influence(
            'rlhf', 'safety/alignment/rlhf',
            'sycophancy', 'safety/misalignment/sycophancy',
            evidence=[ev_dup]
        )

        delta = ia.add_statements([dup_stmt])

        self.assertIsInstance(delta, AssemblyDelta)
        # Duplicate should NOT create a new statement entry
        self.assertEqual(len(delta.new_stmts), 0)
        # Evidence should be appended to the existing child hash
        self.assertIn(spec_hash, delta.new_evidences)
        self.assertEqual(len(delta.new_evidences[spec_hash]), 1)
        self.assertEqual(delta.new_evidences[spec_hash][0].source_api, 'llm_reader')

        # Child statement belief increased with 2 evidences
        self.assertGreater(delta.beliefs[spec_hash], initial_spec_belief)
        # Parent statement belief increased as well
        self.assertGreater(delta.beliefs[gen_hash], initial_gen_belief)

        # Total evidences for child is now 2
        self.assertEqual(len(ia.evs_by_stmt_hash[spec_hash]), 2)

    def test_add_new_specific_statement_creates_refinement(self):
        """Adding a brand new specific statement forms new refinement edge and boosts parent."""
        ia = IncrementalAssembler(
            prepared_stmts=[self.stmt_gen, self.stmt_spec],
            ontology=self.onto,
            matches_fun=location_matches_compositional
        )
        gen_hash = self.stmt_gen.get_hash(matches_fun=location_matches_compositional)
        initial_gen_belief = ia.beliefs[gen_hash]

        # Brand new claim: Sparse Autoencoders reduce Deceptive Alignment
        # SAE (under mechanistic_interpretability -> alignment)
        # Deceptive Alignment (under misalignment)
        ev_sae = Evidence(
            source_api='manual_curation',
            pmid='arxiv:2304.sae',
            text='Sparse Autoencoders detect features involved in deceptive alignment.'
        )
        stmt_sae = _make_influence(
            'sparse_autoencoders',
            'safety/alignment/mechanistic_interpretability/sparse_autoencoders',
            'deceptive_alignment',
            'safety/misalignment/deceptive_alignment',
            evidence=[ev_sae]
        )
        sae_hash = stmt_sae.get_hash(matches_fun=location_matches_compositional)

        delta = ia.add_statements([stmt_sae])

        # New statement created
        self.assertIn(sae_hash, delta.new_stmts)
        self.assertEqual(len(delta.new_stmts), 1)

        # New refinement edge created: parent (gen) -> sae
        self.assertIn((gen_hash, sae_hash), delta.new_refinements)
        self.assertTrue(ia.refinements_graph.has_edge(gen_hash, sae_hash))

        # Parent belief updated and boosted with new SAE evidence
        self.assertGreater(delta.beliefs[gen_hash], initial_gen_belief)

        # Check JSON serialization of delta
        delta_json = delta.to_json()
        self.assertIn('new_stmts', delta_json)
        self.assertIn('new_evidence', delta_json)
        self.assertIn('new_refinements', delta_json)
        self.assertIn('beliefs', delta_json)
        self.assertIn(str(sae_hash), delta_json['new_stmts'])


class TestAssemblyOperations(unittest.TestCase):
    """Test pipeline operations from daltonx.assembly.operations."""

    def test_get_expanded_events_influences(self):
        """get_expanded_events_influences yields both Event objects and Influence."""
        stmt = _make_influence(
            'alignment', 'safety/alignment',
            'misalignment', 'safety/misalignment',
            evidence=[Evidence(source_api='paperqa')]
        )
        expanded = get_expanded_events_influences([stmt])
        # 2 events (subj, obj) + 1 influence = 3 elements
        self.assertEqual(len(expanded), 3)
        events = [x for x in expanded if isinstance(x, Event)]
        influences = [x for x in expanded if isinstance(x, Influence)]
        self.assertEqual(len(events), 2)
        self.assertEqual(len(influences), 1)

    def test_remove_namespaces(self):
        """remove_namespaces removes specified namespaces from db_refs."""
        c = Concept('test', db_refs={'AIS': 'safety/alignment', 'TMP': 'dummy'})
        stmt = Influence(Event(c), Event(c))
        remove_namespaces([stmt], namespaces=['TMP'])
        self.assertNotIn('TMP', c.db_refs)
        self.assertIn('AIS', c.db_refs)

    def test_remove_raw_grounding(self):
        """remove_raw_grounding removes raw_grounding annotation from evidence."""
        ev = Evidence(
            source_api='test',
            annotations={'agents': {'raw_grounding': {'AIS': 'temp'}, 'raw_text': ['a', 'b']}}
        )
        stmt = _make_influence('a', 'safety/alignment', 'b', 'safety/misalignment', evidence=[ev])
        remove_raw_grounding([stmt])
        self.assertNotIn('raw_grounding', ev.annotations['agents'])
        self.assertIn('raw_text', ev.annotations['agents'])

    def test_set_positive_polarities(self):
        """set_positive_polarities sets polarity=1 for events with None polarity."""
        c1 = Concept('a', db_refs={'AIS': 'safety/alignment'})
        c2 = Concept('b', db_refs={'AIS': 'safety/misalignment'})
        stmt = Influence(Event(c1, delta=QualitativeDelta(polarity=None)), Event(c2, delta=QualitativeDelta(polarity=None)))
        set_positive_polarities([stmt])
        self.assertEqual(stmt.subj.delta.polarity, 1)
        self.assertEqual(stmt.obj.delta.polarity, 1)

    def test_filter_out_long_words(self):
        """filter_out_long_words discards concepts with too many words."""
        stmt_short = _make_influence('RLHF', 'safety/alignment/rlhf', 'sycophancy', 'safety/misalignment/sycophancy')
        stmt_long = _make_influence('This is a very long text exceeding the limit of five words', 'safety/alignment', 'short', 'safety/misalignment')
        filtered = filter_out_long_words([stmt_short, stmt_long], k=5)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].subj.concept.name, 'RLHF')

    def test_make_display_name(self):
        """make_display_name formats ontology IDs into readable English."""
        disp = make_display_name('safety/alignment/mechanistic_interpretability/sparse_autoencoders')
        self.assertEqual(disp, 'sparse autoencoders')

    def test_assembly_pipeline_execution(self):
        """AssemblyPipeline runs sequential operations on statements."""
        pipeline = AssemblyPipeline(steps=[
            {'function': 'set_positive_polarities'},
            {'function': 'filter_out_long_words', 'kwargs': {'k': 8}},
        ])
        stmt = Influence(
            Event(Concept('RLHF', db_refs={'AIS': 'safety/alignment/rlhf'}), delta=QualitativeDelta(polarity=None)),
            Event(Concept('sycophancy', db_refs={'AIS': 'safety/misalignment/sycophancy'}), delta=QualitativeDelta(polarity=-1))
        )
        result = pipeline.run([stmt])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].subj.delta.polarity, 1)


if __name__ == '__main__':
    unittest.main()
