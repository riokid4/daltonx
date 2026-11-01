"""daltonx/tests/test_belief.py
=============================
Unit tests for DaltonX Belief Engine (daltonx.assembly.belief).

Tests:
1. set_prior_probs: Multiple independent evidences produce higher belief than single evidence.
2. Negated evidence: epistemics={'negated': True} decreases belief following pp * (1 - np).
3. set_hierarchy_probs: Specific sub-claims boost general claims via supports hierarchy.
4. build_refinements_graph & extend_refinements_graph: Correct NetworkX DAG construction and extension.
5. BayesianScorer & sample_statements.
"""

import unittest
import networkx

from daltonx.statements import (
    Concept,
    Evidence,
    Event,
    QualitativeDelta,
    Influence,
)
from daltonx.assembly.belief import (
    BeliefEngine,
    SimpleScorer,
    BayesianScorer,
    build_refinements_graph,
    extend_refinements_graph,
    sample_statements,
    assert_no_cycle,
    get_ranked_stmts,
    load_default_probs,
)


def _make_influence(subj_name: str, subj_path: str, obj_name: str, obj_path: str, evidence=None, pol_subj=1, pol_obj=-1) -> Influence:
    c_subj = Concept(subj_name, db_refs={'AIS': subj_path, 'TEXT': subj_name})
    c_obj = Concept(obj_name, db_refs={'AIS': obj_path, 'TEXT': obj_name})
    return Influence(
        Event(c_subj, delta=QualitativeDelta(polarity=pol_subj)),
        Event(c_obj, delta=QualitativeDelta(polarity=pol_obj)),
        evidence=evidence if evidence else []
    )


class TestPriorProbs(unittest.TestCase):
    """Test prior probability computation from Evidence lists."""

    def setUp(self):
        self.engine = BeliefEngine()

    def test_single_evidence_belief(self):
        """Single evidence from paperqa gives expected prior probability."""
        ev = Evidence(source_api='paperqa', pmid='arxiv:2301.00001', text='RLHF reduces sycophancy.')
        stmt = _make_influence('RLHF', 'safety/alignment/rlhf', 'sycophancy', 'safety/misalignment/sycophancy', evidence=[ev])

        self.engine.set_prior_probs([stmt])
        self.assertIsNotNone(stmt.belief)
        self.assertGreater(stmt.belief, 0.0)
        self.assertLess(stmt.belief, 1.0)

        # Expected: syst=0.05, rand=0.15 => neg_prob = 0.05 + 0.15 = 0.20 => belief = 0.80
        self.assertAlmostEqual(stmt.belief, 0.80, places=4)

    def test_multiple_evidences_boost_belief(self):
        """Multiple independent pieces of Evidence produce higher belief than a single piece."""
        ev1 = Evidence(source_api='paperqa', pmid='arxiv:2301.00001', text='Paper 1 shows RLHF reduces sycophancy.')
        ev2 = Evidence(source_api='paperqa', pmid='arxiv:2302.00002', text='Paper 2 confirms RLHF mitigates sycophancy.')
        ev3 = Evidence(source_api='arxiv_extractor', pmid='arxiv:2303.00003', text='Arxiv extraction supporting claim.')

        stmt_single = _make_influence('RLHF', 'safety/alignment/rlhf', 'sycophancy', 'safety/misalignment/sycophancy', evidence=[ev1])
        stmt_double = _make_influence('RLHF', 'safety/alignment/rlhf', 'sycophancy', 'safety/misalignment/sycophancy', evidence=[ev1, ev2])
        stmt_multi_source = _make_influence('RLHF', 'safety/alignment/rlhf', 'sycophancy', 'safety/misalignment/sycophancy', evidence=[ev1, ev2, ev3])

        self.engine.set_prior_probs([stmt_single, stmt_double, stmt_multi_source])

        # 1 piece vs 2 pieces from same source: 0.80 -> 1 - (0.05 + 0.15*0.15) = 1 - 0.0725 = 0.9275
        self.assertGreater(stmt_double.belief, stmt_single.belief)
        self.assertAlmostEqual(stmt_double.belief, 0.9275, places=4)

        # 2 pieces from paperqa + 1 piece from arxiv_extractor gives even higher belief
        self.assertGreater(stmt_multi_source.belief, stmt_double.belief)


class TestNegatedEvidence(unittest.TestCase):
    """Test that negated evidence discounts statement belief following pp * (1 - np)."""

    def setUp(self):
        self.engine = BeliefEngine()

    def test_negated_evidence_decreases_belief(self):
        """Negated evidence (epistemics={'negated': True}) decreases belief."""
        ev_pos = Evidence(
            source_api='paperqa',
            pmid='arxiv:2301.00001',
            text='RLHF reduces sycophancy.'
        )
        ev_neg = Evidence(
            source_api='arxiv_extractor',
            pmid='arxiv:2304.00004',
            text='We found RLHF does not reduce sycophancy in large models.',
            epistemics={'negated': True}
        )

        stmt_positive = _make_influence('RLHF', 'safety/alignment/rlhf', 'sycophancy', 'safety/misalignment/sycophancy', evidence=[ev_pos])
        stmt_with_negation = _make_influence('RLHF', 'safety/alignment/rlhf', 'sycophancy', 'safety/misalignment/sycophancy', evidence=[ev_pos, ev_neg])

        self.engine.set_prior_probs([stmt_positive, stmt_with_negation])

        # Positive score: paperqa alone => pp = 0.80
        # Negative score: arxiv_extractor alone (syst=0.06, rand=0.18) => np = 1 - (0.06 + 0.18) = 0.76
        # Expected combined: pp * (1 - np) = 0.80 * (1 - 0.76) = 0.80 * 0.24 = 0.192
        self.assertLess(stmt_with_negation.belief, stmt_positive.belief)
        self.assertAlmostEqual(stmt_with_negation.belief, 0.192, places=4)

    def test_purely_negated_evidence_yields_zero_belief(self):
        """A statement with only negated evidence has belief 0.0."""
        ev_neg = Evidence(
            source_api='paperqa',
            pmid='arxiv:2301.00001',
            text='RLHF does not reduce sycophancy.',
            epistemics={'negated': True}
        )
        stmt = _make_influence('RLHF', 'safety/alignment/rlhf', 'sycophancy', 'safety/misalignment/sycophancy', evidence=[ev_neg])
        self.engine.set_prior_probs([stmt])
        self.assertEqual(stmt.belief, 0.0)


class TestHierarchyProbs(unittest.TestCase):
    """Test hierarchical evidence propagation from specific sub-claims to general claims."""

    def setUp(self):
        self.engine = BeliefEngine()

    def test_hierarchy_probs_propagates_support_evidence(self):
        """Specific sub-claim (DPO reduces Sycophancy) boosts parent claim (Alignment reduces Misalignment)."""
        # General parent claim: Alignment reduces Misalignment
        p_ev = Evidence(source_api='paperqa', pmid='arxiv:gen01', text='Alignment reduces misalignment.')
        parent_stmt = _make_influence('alignment', 'safety/alignment', 'misalignment', 'safety/misalignment', evidence=[p_ev])

        # Specific child claim: DPO reduces Sycophancy
        c_ev = Evidence(source_api='arxiv_extractor', pmid='arxiv:spec01', text='DPO strongly reduces sycophancy.')
        child_stmt = _make_influence(
            'direct_preference_optimization', 'safety/alignment/rlhf/direct_preference_optimization',
            'sycophancy', 'safety/misalignment/sycophancy',
            evidence=[c_ev]
        )

        # Preassembly links: parent.supports contains child, child.supported_by contains parent
        parent_stmt.supports = [child_stmt]
        child_stmt.supported_by = [parent_stmt]

        # Prior probabilities before hierarchy propagation
        self.engine.set_prior_probs([parent_stmt, child_stmt])
        parent_prior = parent_stmt.belief
        child_prior = child_stmt.belief

        # Parent has 1 paperqa evidence: 0.80
        self.assertAlmostEqual(parent_prior, 0.80, places=4)
        # Child has 1 arxiv_extractor evidence: 1 - (0.06 + 0.18) = 0.76
        self.assertAlmostEqual(child_prior, 0.76, places=4)

        # Run hierarchical probability propagation
        self.engine.set_hierarchy_probs([parent_stmt, child_stmt])

        # Parent accumulated evidence from child_stmt:
        # parent now scored on paperqa (syst=0.05, rand=0.15) AND arxiv_extractor (syst=0.06, rand=0.18)
        # neg_prob = (0.05 + 0.15) * (0.06 + 0.18) = 0.20 * 0.24 = 0.048 => belief = 1 - 0.048 = 0.952
        self.assertGreater(parent_stmt.belief, parent_prior)
        self.assertAlmostEqual(parent_stmt.belief, 0.952, places=4)

        # Child statement belief remains unchanged (evidence flows up, not down)
        self.assertAlmostEqual(child_stmt.belief, child_prior, places=4)


class TestRefinementsGraph(unittest.TestCase):
    """Test NetworkX refinements graph construction and extension."""

    def test_build_refinements_graph(self):
        """build_refinements_graph creates a valid DAG pointing from general to specific."""
        gen_stmt = _make_influence(
            'alignment', 'safety/alignment',
            'misalignment', 'safety/misalignment',
            evidence=[Evidence(source_api='paperqa')]
        )
        spec_stmt = _make_influence(
            'rlhf', 'safety/alignment/rlhf',
            'sycophancy', 'safety/misalignment/sycophancy',
            evidence=[Evidence(source_api='paperqa')]
        )
        gen_stmt.supports = [spec_stmt]
        spec_stmt.supported_by = [gen_stmt]

        g = build_refinements_graph([gen_stmt, spec_stmt])

        self.assertIsInstance(g, networkx.DiGraph)
        self.assertEqual(len(g.nodes), 2)
        self.assertEqual(len(g.edges), 1)

        gen_hash = gen_stmt.get_hash()
        spec_hash = spec_stmt.get_hash()

        self.assertIn(gen_hash, g.nodes)
        self.assertIn(spec_hash, g.nodes)
        # Edge points from general to specific
        self.assertTrue(g.has_edge(gen_hash, spec_hash))

        # Check DAG acyclicity
        assert_no_cycle(g)

    def test_extend_refinements_graph(self):
        """extend_refinements_graph adds a new statement and edges from its general ancestors."""
        gen_stmt = _make_influence(
            'alignment', 'safety/alignment',
            'misalignment', 'safety/misalignment',
            evidence=[Evidence(source_api='paperqa')]
        )
        g = build_refinements_graph([gen_stmt])
        self.assertEqual(len(g.nodes), 1)

        new_spec_stmt = _make_influence(
            'sparse_autoencoders', 'safety/alignment/mechanistic_interpretability/sparse_autoencoders',
            'deceptive_alignment', 'safety/misalignment/deceptive_alignment',
            evidence=[Evidence(source_api='arxiv_extractor')]
        )
        gen_hash = gen_stmt.get_hash()
        new_hash = new_spec_stmt.get_hash()

        extend_refinements_graph(g, new_spec_stmt, less_specifics=[gen_hash])

        self.assertEqual(len(g.nodes), 2)
        self.assertEqual(len(g.edges), 1)
        self.assertTrue(g.has_edge(gen_hash, new_hash))
        assert_no_cycle(g)

        # Ranked statements order: most specific first
        ranked = get_ranked_stmts(g)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0].get_hash(), new_hash)
        self.assertEqual(ranked[1].get_hash(), gen_hash)


class TestBayesianScorerAndSampling(unittest.TestCase):
    """Test BayesianScorer with empirical counts and statement sampling."""

    def test_bayesian_scorer_update_counts(self):
        """BayesianScorer updates error rates dynamically as counts are supplied."""
        scorer = BayesianScorer(
            prior_counts={'paperqa': [90, 10]}
        )
        stmt = _make_influence(
            'alignment', 'safety/alignment',
            'misalignment', 'safety/misalignment',
            evidence=[Evidence(source_api='paperqa')]
        )
        engine = BeliefEngine(scorer=scorer)
        engine.set_prior_probs([stmt])
        initial_belief = stmt.belief
        self.assertGreater(initial_belief, 0.8)

        # Add more negative counts (errors)
        scorer.update_counts({'paperqa': [0, 50]})
        engine.set_prior_probs([stmt])
        # Belief must drop due to higher error rate
        self.assertLess(stmt.belief, initial_belief)

    def test_sample_statements(self):
        """sample_statements filters statements based on their belief probability."""
        st_high = _make_influence(
            'alignment', 'safety/alignment',
            'misalignment', 'safety/misalignment',
            evidence=[Evidence(source_api='manual_curation')]
        )
        st_high.belief = 0.99

        st_low = _make_influence(
            'alignment', 'safety/alignment',
            'misalignment', 'safety/misalignment',
            evidence=[Evidence(source_api='llm_reader')]
        )
        st_low.belief = 0.01

        # With seed=42, st_high should almost certainly be sampled, st_low not
        sampled = sample_statements([st_high, st_low], seed=42)
        self.assertIn(st_high, sampled)
        self.assertNotIn(st_low, sampled)


if __name__ == '__main__':
    unittest.main()
