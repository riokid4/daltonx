"""daltonx/tests/test_grounding.py
===================================
Unit tests for Section 3 – The Grounding Engine.

Tests prove:
  1. String normalization, dash removal, unicode handling, and depluralization
     work identically to Gilda.
  2. ScoredMatch and score() compute exact match vs case/dash/synonym penalties.
  3. Grounder built from AISafetyOntology grounds messy strings
     (e.g., "Reward-Hacking", "reward hacks", "RLHF") to their official 'AIS'
     ontology IDs and sets Concept.db_refs['AIS'] on Section 1 Concept objects.
"""
import unittest

from daltonx.grounding import (
    Term,
    Grounder,
    ScoredMatch,
    Match,
    normalize,
    replace_dashes,
    remove_dashes,
    replace_whitespace,
    depluralize,
    generate_match,
    score,
    score_string_match,
    score_status,
    build_grounder_from_ontology,
    ground_to_concept,
)
from daltonx.ontology import load_ai_safety_ontology
from daltonx.statements import Concept


class TestStringProcessing(unittest.TestCase):
    """Test text normalization and depluralization mechanics."""

    def test_normalize_whitespace_and_casing(self):
        # In Gilda, replace_whitespace collapses multiple spaces into a single space
        self.assertEqual(normalize("   RLHF   "), " rlhf ")
        self.assertEqual(normalize("   RLHF   ".strip()), "rlhf")
        self.assertEqual(normalize("Reinforcement   Learning"), "reinforcement learning")

    def test_normalize_dashes(self):
        # Hyphen, en-dash, em-dash
        self.assertEqual(normalize("reward-hacking"), "rewardhacking")
        self.assertEqual(replace_dashes("sparse–autoencoder"), "sparse-autoencoder")
        self.assertEqual(remove_dashes("sparse-autoencoder"), "sparseautoencoder")

    def test_normalize_unicode(self):
        # Accented characters normalized to ASCII
        self.assertEqual(normalize("Naïve Bayes"), "naive bayes")

    def test_depluralization(self):
        # Basic plural -s
        res = depluralize("autoencoders")
        self.assertEqual(res[0][0], "autoencoder")
        self.assertEqual(res[0][1], "plural_s")

        # Non-plural
        res_np = depluralize("rlhf")
        self.assertEqual(res_np[0][0], "rlhf")
        self.assertEqual(res_np[0][1], "non_plural")

        # Plural -es
        res_es = depluralize("switches")
        self.assertIn(("switch", "plural_es"), res_es)


class TestScoring(unittest.TestCase):
    """Test match generation and scoring functions."""

    def setUp(self):
        self.term_exact = Term(
            norm_text="rlhf",
            text="RLHF",
            db="AIS",
            id="safety/alignment/rlhf",
            entry_name="rlhf",
            status="name",
            source="AIS"
        )
        self.term_synonym = Term(
            norm_text="reinforcement learning from human feedback",
            text="reinforcement learning from human feedback",
            db="AIS",
            id="safety/alignment/rlhf",
            entry_name="rlhf",
            status="synonym",
            source="AIS"
        )

    def test_exact_match_score(self):
        # Query exactly matches reference string
        m = generate_match("RLHF", "RLHF")
        self.assertTrue(m.exact)
        s = score(m, self.term_exact)
        # Exact match with status 'name' gives highest score
        self.assertAlmostEqual(s, (3 * 2 + 1.0) / 9, places=5)

    def test_case_mismatch_penalty(self):
        m_exact = generate_match("RLHF", "RLHF")
        m_lower = generate_match("rlhf", "RLHF")
        s_exact = score(m_exact, self.term_exact)
        s_lower = score(m_lower, self.term_exact)
        # Case variation incurs a small penalty
        self.assertGreater(s_exact, s_lower)

    def test_dash_mismatch_penalty(self):
        m_exact = generate_match("reward hacking", "reward hacking")
        m_dash = generate_match("reward-hacking", "reward hacking")
        s_exact = score(m_exact, self.term_exact)
        s_dash = score(m_dash, self.term_exact)
        self.assertGreater(s_exact, s_dash)

    def test_status_score_prioritizes_name_over_synonym(self):
        # 'name' status gets higher status score than 'synonym'
        self.assertGreater(score_status(self.term_exact), score_status(self.term_synonym))


class TestGrounderWithAISafetyOntology(unittest.TestCase):
    """Test Grounder populated directly from AISafetyOntology."""

    @classmethod
    def setUpClass(cls):
        cls.onto = load_ai_safety_ontology()
        cls.onto.initialize()
        cls.grounder = build_grounder_from_ontology(cls.onto)

    def test_terms_loaded_from_ontology(self):
        # 11 ontology nodes with multiple synonyms each -> >= 50 terms
        self.assertGreaterEqual(len(self.grounder.entries), 20)

    def test_ground_exact_name(self):
        matches = self.grounder.ground("rlhf")
        self.assertGreater(len(matches), 0)
        top = matches[0]
        self.assertEqual(top.term.id, "safety/alignment/rlhf")
        self.assertEqual(top.term.db, "AIS")

    def test_ground_messy_dash_and_case(self):
        matches = self.grounder.ground("Reward-Hacking")
        self.assertGreater(len(matches), 0)
        top = matches[0]
        self.assertEqual(top.term.id, "safety/misalignment/reward_hacking")
        self.assertEqual(top.term.db, "AIS")

    def test_ground_plural(self):
        # "sparse autoencoders" -> depluralizes to "sparse autoencoder"
        matches = self.grounder.ground("sparse autoencoders")
        self.assertGreater(len(matches), 0)
        top = matches[0]
        self.assertEqual(
            top.term.id,
            "safety/alignment/mechanistic_interpretability/sparse_autoencoders"
        )

    def test_ground_synonym_phrase(self):
        # "mesa-optimizer deception" is an example synonym for deceptive_alignment
        matches = self.grounder.ground("mesa-optimizer deception")
        self.assertGreater(len(matches), 0)
        top = matches[0]
        self.assertEqual(top.term.id, "safety/misalignment/deceptive_alignment")

    def test_ground_to_concept_sets_db_refs(self):
        # Verify seamless integration with Section 1 Concept class
        concept = ground_to_concept("Reward-Hacking", self.grounder)
        self.assertIsInstance(concept, Concept)
        self.assertIn("AIS", concept.db_refs)
        self.assertEqual(concept.db_refs["AIS"], "safety/misalignment/reward_hacking")
        self.assertEqual(concept.name, "reward_hacking")

        # Unmatched string preserves name with empty db_refs
        unmatched = ground_to_concept("completely unknown concept xyz123", self.grounder)
        self.assertIsInstance(unmatched, Concept)
        self.assertEqual(unmatched.db_refs, {})


if __name__ == "__main__":
    unittest.main()
