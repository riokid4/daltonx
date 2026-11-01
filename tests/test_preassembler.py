"""daltonx/tests/test_preassembler.py
======================================
Unit tests for Section 4 – The Preassembler & Refinement Filters.

Tests prove:
  1. combine_duplicate_stmts merges duplicate AI Safety claims from multiple
     sources and merges their Evidence lists without losing provenance.
  2. combine_related (default OntologyRefinementFilter and CompositionalRefinementFilter)
     links general claims (e.g., "Alignment reduces Misalignment") and specific
     claims (e.g., "RLHF reduces Sycophancy", "Sparse Autoencoders reduce Deceptive Alignment")
     via supports and supported_by.
  3. find_contradicts detects contradicting claims (both opposite polarity and
     opposite ontology concepts).
  4. flatten_stmts and flatten_evidence recover the hierarchical graph contents.
"""
import unittest

from daltonx.ontology import load_ai_safety_ontology
from daltonx.statements import (
    Concept,
    Event,
    Influence,
    QualitativeDelta,
    Evidence,
    AISafetyContext,
    RefContext,
)
from daltonx.assembly import (
    Preassembler,
    OntologyRefinementFilter,
    CompositionalRefinementFilter,
    flatten_stmts,
    flatten_evidence,
    matches_compositional,
)


def _make_influence(subj_id, subj_name, obj_id, obj_name, subj_pol, obj_pol, text=None, source_api="test"):
    """Helper to build grounded Influence statements for tests."""
    c_subj = Concept(subj_name, db_refs={'AIS': subj_id, 'TEXT': subj_name})
    c_obj = Concept(obj_name, db_refs={'AIS': obj_id, 'TEXT': obj_name})
    ev = Evidence(source_api=source_api, text=text if text else f"{subj_name} influences {obj_name}")
    stmt = Influence(
        Event(c_subj, delta=QualitativeDelta(polarity=subj_pol)),
        Event(c_obj, delta=QualitativeDelta(polarity=obj_pol)),
        evidence=[ev]
    )
    return stmt


class TestPreassemblerDuplicates(unittest.TestCase):
    """Test duplicate claim deduplication and evidence merging."""

    @classmethod
    def setUpClass(cls):
        cls.onto = load_ai_safety_ontology()
        cls.onto.initialize()

    def test_combine_duplicates_merges_evidence(self):
        # Two papers extracting the exact same claim: RLHF(+) -> sycophancy(-)
        stmt1 = _make_influence(
            'safety/alignment/rlhf', 'rlhf',
            'safety/misalignment/sycophancy', 'sycophancy',
            1, -1,
            text="Paper 1: RLHF training significantly mitigated sycophantic behaviour.",
            source_api="arxiv:paper1"
        )
        stmt2 = _make_influence(
            'safety/alignment/rlhf', 'rlhf',
            'safety/misalignment/sycophancy', 'sycophancy',
            1, -1,
            text="Paper 2: Experiments show RLHF reduces model flattery and sycophancy.",
            source_api="arxiv:paper2"
        )
        uuid1 = stmt1.uuid
        uuid2 = stmt2.uuid

        pa = Preassembler(self.onto, [stmt1, stmt2])
        unique_stmts = pa.combine_duplicates()

        # Should be deduplicated into 1 statement
        self.assertEqual(len(unique_stmts), 1)
        merged = unique_stmts[0]

        # Both evidence items must be preserved
        self.assertEqual(len(merged.evidence), 2)
        source_apis = {ev.source_api for ev in merged.evidence}
        self.assertEqual(source_apis, {"arxiv:paper1", "arxiv:paper2"})

        # Provenance: prior_uuids must track original statement IDs
        prior_uuids = set()
        for ev in merged.evidence:
            prior_uuids.update(ev.annotations.get('prior_uuids', []))
        self.assertIn(uuid1, prior_uuids)
        self.assertIn(uuid2, prior_uuids)

    def test_distinct_claims_not_combined(self):
        # Claim A: RLHF -> sycophancy
        # Claim B: Sparse Autoencoders -> deceptive alignment
        stmtA = _make_influence(
            'safety/alignment/rlhf', 'rlhf',
            'safety/misalignment/sycophancy', 'sycophancy',
            1, -1
        )
        stmtB = _make_influence(
            'safety/alignment/mechanistic_interpretability/sparse_autoencoders', 'sae',
            'safety/misalignment/deceptive_alignment', 'deceptive_alignment',
            1, -1
        )

        pa = Preassembler(self.onto, [stmtA, stmtB])
        unique_stmts = pa.combine_duplicates()
        self.assertEqual(len(unique_stmts), 2)


class TestPreassemblerRefinement(unittest.TestCase):
    """Test hierarchical refinement graph linking via supports and supported_by."""

    @classmethod
    def setUpClass(cls):
        cls.onto = load_ai_safety_ontology()
        cls.onto.initialize()

    def test_combine_related_general_and_specific(self):
        # General claim: Alignment(+) -> Misalignment(-)
        general_stmt = _make_influence(
            'safety/alignment', 'alignment',
            'safety/misalignment', 'misalignment',
            1, -1,
            text="General: Alignment techniques decrease misalignment risks."
        )

        # Specific claim 1: RLHF(+) -> Sycophancy(-)
        specific_stmt1 = _make_influence(
            'safety/alignment/rlhf', 'rlhf',
            'safety/misalignment/sycophancy', 'sycophancy',
            1, -1,
            text="Specific 1: RLHF decreases sycophancy."
        )

        # Specific claim 2: Sparse Autoencoders(+) -> Deceptive Alignment(-)
        specific_stmt2 = _make_influence(
            'safety/alignment/mechanistic_interpretability/sparse_autoencoders', 'sparse_autoencoders',
            'safety/misalignment/deceptive_alignment', 'deceptive_alignment',
            1, -1,
            text="Specific 2: Sparse Autoencoders reduce deceptive alignment."
        )

        pa = Preassembler(self.onto, [general_stmt, specific_stmt1, specific_stmt2])
        top_stmts = pa.combine_related(return_toplevel=True)

        # Top-level statements are the most specific ones (no supporting statements above them)
        self.assertEqual(len(top_stmts), 2)
        top_hashes = {st.get_hash() for st in top_stmts}
        self.assertIn(specific_stmt1.get_hash(), top_hashes)
        self.assertIn(specific_stmt2.get_hash(), top_hashes)

        # Each specific statement must be supported by the general statement
        for top_stmt in top_stmts:
            self.assertEqual(len(top_stmt.supported_by), 1)
            supported_by_stmt = top_stmt.supported_by[0]
            self.assertEqual(supported_by_stmt.get_hash(), general_stmt.get_hash())
            self.assertIn(top_stmt, supported_by_stmt.supports)

        # flatten_stmts must recover all 3 statements
        all_stmts = flatten_stmts(top_stmts)
        self.assertEqual(len(all_stmts), 3)

        # flatten_evidence on top statements should collect evidence from supporting statements
        flattened_ev_stmts = flatten_evidence(top_stmts)
        for f_stmt in flattened_ev_stmts:
            # 1 direct evidence + 1 supported_by evidence from general_stmt
            self.assertEqual(len(f_stmt.evidence), 2)
            support_types = {ev.annotations.get('support_type') for ev in f_stmt.evidence}
            self.assertIn('direct', support_types)
            self.assertIn('supported_by', support_types)

    def test_combine_related_with_compositional_filter(self):
        # General claim: Alignment -> Misalignment
        general_stmt = _make_influence(
            'safety/alignment', 'alignment',
            'safety/misalignment', 'misalignment',
            1, -1
        )
        # Specific claim: RLHF -> Sycophancy
        specific_stmt = _make_influence(
            'safety/alignment/rlhf', 'rlhf',
            'safety/misalignment/sycophancy', 'sycophancy',
            1, -1
        )

        filt = CompositionalRefinementFilter(self.onto)
        pa = Preassembler(self.onto, [general_stmt, specific_stmt])
        top_stmts = pa.combine_related(return_toplevel=True, filters=[filt])

        self.assertEqual(len(top_stmts), 1)
        self.assertEqual(top_stmts[0].get_hash(), specific_stmt.get_hash())
        self.assertEqual(len(top_stmts[0].supported_by), 1)
        self.assertEqual(top_stmts[0].supported_by[0].get_hash(), general_stmt.get_hash())


class TestPreassemblerContradictions(unittest.TestCase):
    """Test contradiction detection between opposing claims."""

    @classmethod
    def setUpClass(cls):
        cls.onto = load_ai_safety_ontology()
        cls.onto.initialize()

    def test_find_contradicts_opposite_polarity(self):
        # Claim 1: RLHF increases Alignment (overall pol = +1)
        stmt1 = _make_influence(
            'safety/alignment/rlhf', 'rlhf',
            'safety/alignment', 'alignment',
            1, 1,
            text="Paper A: RLHF increases alignment."
        )

        # Claim 2: RLHF decreases Alignment (overall pol = -1)
        stmt2 = _make_influence(
            'safety/alignment/rlhf', 'rlhf',
            'safety/alignment', 'alignment',
            1, -1,
            text="Paper B: RLHF decreases alignment."
        )

        pa = Preassembler(self.onto, [stmt1, stmt2])
        contradictions = pa.find_contradicts()
        self.assertEqual(len(contradictions), 1)
        c1, c2 = contradictions[0]
        self.assertIn(c1.get_hash(), {stmt1.get_hash(), stmt2.get_hash()})
        self.assertIn(c2.get_hash(), {stmt1.get_hash(), stmt2.get_hash()})

    def test_find_contradicts_opposite_ontology_concept(self):
        # In our ontology: alignment and misalignment are opposites.
        # Claim 1: RLHF(+) increases Alignment(+) -> overall pol = +1
        stmt1 = _make_influence(
            'safety/alignment/rlhf', 'rlhf',
            'safety/alignment', 'alignment',
            1, 1
        )

        # Claim 2: RLHF(+) increases Misalignment(+) -> overall pol = +1
        # Because alignment is_opposite misalignment, claiming RLHF increases alignment
        # contradicts claiming RLHF increases misalignment!
        stmt2 = _make_influence(
            'safety/alignment/rlhf', 'rlhf',
            'safety/misalignment', 'misalignment',
            1, 1
        )

        pa = Preassembler(self.onto, [stmt1, stmt2])
        contradictions = pa.find_contradicts()
        self.assertEqual(len(contradictions), 1)

    def test_compatible_claims_not_contradictory(self):
        # Claim 1: RLHF(+) increases Alignment(+) -> overall pol = +1
        stmt1 = _make_influence(
            'safety/alignment/rlhf', 'rlhf',
            'safety/alignment', 'alignment',
            1, 1
        )

        # Claim 2: RLHF(+) decreases Misalignment(-) -> overall pol = -1
        # Increasing alignment and decreasing misalignment are COMPATIBLE!
        stmt2 = _make_influence(
            'safety/alignment/rlhf', 'rlhf',
            'safety/misalignment', 'misalignment',
            1, -1
        )

        pa = Preassembler(self.onto, [stmt1, stmt2])
        contradictions = pa.find_contradicts()
        self.assertEqual(len(contradictions), 0)


if __name__ == '__main__':
    unittest.main()
