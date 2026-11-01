# -*- coding: utf-8 -*-
"""Unit tests for Section 8: AI Safety Literature Reader & End-to-End Pipeline.

Tests:
1. AISafetyProcessor converting raw extracted dicts (with noisy text, AISafetyContext,
   negation, hedging, model scales, training paradigms) into grounded Influence statements.
2. Passage chunking with metadata preservation.
3. LLMExtractor pluggable backends (custom callable client and deterministic offline rules).
4. Full End-to-End Pipeline test:
   Passages -> LLMExtractor -> AISafetyProcessor -> AssemblyPipeline ->
   IncrementalAssembler -> CAGAssembler -> CytoscapeJS export, verifying:
   - concept grounding
   - statement deduplication
   - hierarchical refinement (supports / supported_by)
   - belief propagation
   - incremental delta updates
   - CytoscapeJS graph export structure
"""

import json
import os
import sys
import unittest

# Ensure daltonx root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from daltonx.assembly.incremental import IncrementalAssembler
from daltonx.assembly.operations import AssemblyPipeline
from daltonx.exporters.cag import CAGAssembler
from daltonx.ontology import load_ai_safety_ontology
from daltonx.reader import (
    AISafetyProcessor,
    LLMExtractor,
    Passage,
    chunk_text,
    process_text_to_graph,
)
from daltonx.statements import (
    AISafetyContext,
    Concept,
    Event,
    Evidence,
    Influence,
    QualitativeDelta,
    RefContext,
)


class TestAISafetyProcessor(unittest.TestCase):
    """Tests for AISafetyProcessor statement construction and concept grounding."""

    @classmethod
    def setUpClass(cls):
        cls.ontology = load_ai_safety_ontology()
        cls.ontology.initialize()
        cls.processor = AISafetyProcessor(ontology=cls.ontology)

    def test_processor_grounding_rlhf_and_sycophancy(self):
        """Noisy strings 'RLHF training' and 'sycophantic responses' ground correctly."""
        raw_extraction = {
            'cause_text': 'RLHF training',
            'cause_polarity': 1,
            'cause_adjectives': ['iterative'],
            'effect_text': 'sycophantic responses',
            'effect_polarity': 1,
            'effect_adjectives': ['severe'],
            'evidence_text': 'Iterative RLHF training increases severe sycophantic responses.',
            'paper_id': 'arXiv:2310.13548',
            'citation': 'Sharma et al., 2023',
            'section': 'Results',
        }
        stmt = self.processor.get_causal_relation(raw_extraction)
        self.assertIsNotNone(stmt)
        self.assertIsInstance(stmt, Influence)

        # Subject: RLHF
        self.assertEqual(stmt.subj.concept.name, 'rlhf')
        self.assertIn('AIS', stmt.subj.concept.db_refs)
        top_subj_id = stmt.subj.concept.db_refs['AIS'][0][0]
        self.assertEqual(top_subj_id, 'safety/alignment/rlhf')
        self.assertEqual(stmt.subj.delta.polarity, 1)
        self.assertIn('iterative', stmt.subj.delta.adjectives)

        # Object: Sycophancy
        self.assertEqual(stmt.obj.concept.name, 'sycophancy')
        self.assertIn('AIS', stmt.obj.concept.db_refs)
        top_obj_id = stmt.obj.concept.db_refs['AIS'][0][0]
        self.assertEqual(top_obj_id, 'safety/misalignment/sycophancy')
        self.assertEqual(stmt.obj.delta.polarity, 1)
        self.assertIn('severe', stmt.obj.delta.adjectives)

        # Evidence
        self.assertEqual(len(stmt.evidence), 1)
        ev = stmt.evidence[0]
        self.assertEqual(ev.source_api, 'ais')
        self.assertEqual(ev.pmid, 'arXiv:2310.13548')
        self.assertEqual(ev.annotations['citation'], 'Sharma et al., 2023')
        self.assertEqual(ev.annotations['subj_text'], 'RLHF training')
        self.assertEqual(ev.annotations['obj_text'], 'sycophantic responses')

    def test_processor_grounding_sparse_autoencoders_and_deceptive_alignment(self):
        """Sparse autoencoders and deceptive alignment ground to ontology nodes."""
        raw_extraction = {
            'cause_text': 'sparse autoencoders',
            'cause_polarity': 1,
            'effect_text': 'deceptive alignment',
            'effect_polarity': -1,
            'evidence_text': 'Sparse autoencoders mitigate deceptive alignment in frontier models.',
            'paper_id': 'arXiv:2401.00001',
        }
        stmt = self.processor.get_causal_relation(raw_extraction)
        self.assertIsNotNone(stmt)

        # Subj
        self.assertEqual(stmt.subj.concept.name, 'sparse_autoencoders')
        self.assertEqual(
            stmt.subj.concept.db_refs['AIS'][0][0],
            'safety/alignment/mechanistic_interpretability/sparse_autoencoders',
        )
        self.assertEqual(stmt.subj.delta.polarity, 1)

        # Obj
        self.assertEqual(stmt.obj.concept.name, 'deceptive_alignment')
        self.assertEqual(
            stmt.obj.concept.db_refs['AIS'][0][0],
            'safety/misalignment/deceptive_alignment',
        )
        self.assertEqual(stmt.obj.delta.polarity, -1)

    def test_processor_context_extraction(self):
        """Context fields (model, scale, training paradigm, eval setting) are attached to AISafetyContext."""
        raw_extraction = {
            'cause_text': 'constitutional AI',
            'effect_text': 'model safety',
            'model': 'Claude 3',
            'model_scale': '175B',
            'architecture': 'Transformer',
            'training_paradigm': 'RLHF',
            'evaluation_setting': 'Anthropic Safety Benchmark',
            'empirical_vs_theoretical': 'empirical',
            'evidence_text': 'Constitutional AI increases model safety in Claude 3 evaluations.',
        }
        stmt = self.processor.get_causal_relation(raw_extraction)
        self.assertIsNotNone(stmt)

        ev = stmt.evidence[0]
        self.assertIsNotNone(ev.context)
        self.assertIsInstance(ev.context, AISafetyContext)

        ctx = ev.context
        # Model RefContext
        self.assertIsNotNone(ctx.model)
        self.assertEqual(ctx.model.name, 'Claude 3')
        self.assertEqual(ctx.model.db_refs.get('scale'), '175B')
        self.assertEqual(ctx.model.db_refs.get('architecture'), 'Transformer')

        # Environment RefContext
        self.assertIsNotNone(ctx.environment)
        self.assertEqual(ctx.environment.name, 'Anthropic Safety Benchmark')
        self.assertEqual(ctx.environment.db_refs.get('training_paradigm'), 'RLHF')
        self.assertEqual(ctx.environment.db_refs.get('empirical_vs_theoretical'), 'empirical')

    def test_processor_negated_and_hedged_epistemics(self):
        """Negated and hedged claims set appropriate epistemic flags on Evidence."""
        raw_extraction = {
            'cause_text': 'unsupervised pretraining',
            'effect_text': 'model alignment',
            'negated': True,
            'hedged': 'preliminary observation',
            'evidence_text': 'Unsupervised pretraining does not lead to model alignment.',
        }
        stmt = self.processor.get_causal_relation(raw_extraction)
        self.assertIsNotNone(stmt)
        ev = stmt.evidence[0]
        self.assertTrue(ev.epistemics.get('negated'))
        self.assertIn('preliminary observation', ev.epistemics.get('hedgings', []))

    def test_processor_skips_invalid_extractions(self):
        """Extractions missing cause or effect return None without error."""
        self.assertIsNone(self.processor.get_causal_relation({}))
        self.assertIsNone(self.processor.get_causal_relation({'cause_text': 'RLHF'}))
        self.assertIsNone(self.processor.get_causal_relation({'effect_text': 'alignment'}))


class TestPassageChunker(unittest.TestCase):
    """Tests for text chunking with metadata."""

    def test_chunk_short_text(self):
        """Text shorter than chunk_size produces a single passage with metadata."""
        meta = {'paper_id': 'P01', 'title': 'Test Paper'}
        passages = chunk_text("A short paper abstract on AI safety.", chunk_size=500, metadata=meta)
        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0].text, "A short paper abstract on AI safety.")
        self.assertEqual(passages[0].metadata['paper_id'], 'P01')
        self.assertEqual(passages[0].chunk_index, 0)

    def test_chunk_long_paragraphs(self):
        """Multi-paragraph long text is split into multiple overlapping passages."""
        p1 = "Paragraph 1: " + "Deep learning models require safety evaluations. " * 8
        p2 = "Paragraph 2: " + "Reinforcement learning from human feedback aligns outputs. " * 8
        p3 = "Paragraph 3: " + "Mechanistic interpretability provides visibility into features. " * 8
        full_text = f"{p1}\n\n{p2}\n\n{p3}"

        passages = chunk_text(full_text, chunk_size=400, overlap=50, metadata={'paper_id': 'P02'})
        self.assertGreater(len(passages), 1)
        for p in passages:
            self.assertEqual(p.metadata['paper_id'], 'P02')
            self.assertGreater(len(p.text), 0)


class TestLLMExtractor(unittest.TestCase):
    """Tests for LLMExtractor pluggable backends and offline extraction."""

    def test_offline_extractor_increases_pattern(self):
        """Offline rules detect 'RLHF increases sycophancy'."""
        extractor = LLMExtractor(client='offline')
        text = "Empirical results show that RLHF training increases sycophantic responses in GPT-4."
        claims = extractor.extract_from_passage(text)
        self.assertGreaterEqual(len(claims), 1)
        claim = claims[0]
        self.assertIn('rlhf', claim['cause_text'].lower())
        self.assertIn('sycophant', claim['effect_text'].lower())
        self.assertEqual(claim['cause_polarity'], 1)
        self.assertEqual(claim['effect_polarity'], 1)
        self.assertEqual(claim['model'], 'GPT-4')

    def test_offline_extractor_mitigates_pattern(self):
        """Offline rules detect 'sparse autoencoders reduce deceptive alignment'."""
        extractor = LLMExtractor(client='offline')
        text = "Our experiments demonstrate that sparse autoencoders significantly mitigate deceptive alignment."
        claims = extractor.extract_from_passage(text)
        self.assertGreaterEqual(len(claims), 1)
        claim = claims[0]
        self.assertEqual(claim['effect_polarity'], -1)
        self.assertIn('significant', claim['cause_adjectives'])

    def test_offline_extractor_negation_and_hedging(self):
        """Offline rules detect negation ('does not') and hedging ('may')."""
        extractor = LLMExtractor(client='offline')
        text = "RLHF training does not increase safety. Scalable oversight may enhance alignment."
        claims = extractor.extract_from_passage(text)
        self.assertGreaterEqual(len(claims), 2)
        # First claim negated
        self.assertTrue(claims[0]['negated'])
        # Second claim hedged
        self.assertTrue(claims[1]['hedged'])

    def test_custom_callable_client(self):
        """A custom callable backend returns structured JSON parsed into claims."""
        def fake_llm(prompt, text):
            return {
                'causal_claims': [
                    {
                        'cause_text': 'scalable oversight',
                        'cause_polarity': 1,
                        'effect_text': 'model alignment',
                        'effect_polarity': 1,
                        'evidence_text': text,
                        'model': 'Claude 3',
                        'confidence': 0.95,
                    }
                ]
            }

        extractor = LLMExtractor(client=fake_llm)
        stmts = extractor.extract_statements("Passage text about scalable oversight and alignment.")
        self.assertEqual(len(stmts), 1)
        stmt = stmts[0]
        self.assertEqual(stmt.subj.concept.name, 'scalable_oversight')
        self.assertEqual(stmt.obj.concept.name, 'alignment')
        self.assertEqual(stmt.belief, 0.95)


class TestEndToEndPipeline(unittest.TestCase):
    """Full End-to-End Pipeline test:
    Passages -> LLMExtractor -> AISafetyProcessor -> AssemblyPipeline ->
    IncrementalAssembler -> CAGAssembler -> CytoscapeJS export.
    """

    def test_full_pipeline_with_deduplication_refinement_and_belief(self):
        """Verify complete flow:
        - 4 simulated paper claims with general, specific, duplicate, and negative influence.
        - Deduplication combines duplicates and boosts belief.
        - Hierarchical refinement creates parent-child supports edges.
        - CAGAssembler produces CytoscapeJS graph with signed edges and belief weights.
        """
        ontology = load_ai_safety_ontology()
        ontology.initialize()

        # Custom deterministic extractor simulating extraction from 4 scientific papers
        def mock_paper_extractor(prompt, text):
            if "Paper 1" in text:
                return [{
                    'cause_text': 'alignment',
                    'cause_polarity': 1,
                    'effect_text': 'safety',
                    'effect_polarity': 1,
                    'evidence_text': 'Alignment increases safety across models.',
                    'paper_id': 'Paper1_General',
                    'citation': 'Anthropic, 2023',
                }]
            elif "Paper 2" in text:
                return [{
                    # Specific refinement of Paper 1: rlhf isa alignment
                    'cause_text': 'rlhf',
                    'cause_polarity': 1,
                    'effect_text': 'safety',
                    'effect_polarity': 1,
                    'evidence_text': 'RLHF training enhances safety in production.',
                    'paper_id': 'Paper2_Specific',
                    'citation': 'OpenAI, 2023',
                }]
            elif "Paper 3" in text:
                return [{
                    # Opposite polarity side effect: rlhf increases sycophancy
                    'cause_text': 'rlhf',
                    'cause_polarity': 1,
                    'effect_text': 'sycophancy',
                    'effect_polarity': 1,
                    'evidence_text': 'RLHF exacerbates sycophancy in dialogue models.',
                    'paper_id': 'Paper3_SideEffect',
                    'citation': 'Sharma et al., 2023',
                }]
            elif "Paper 4" in text:
                return [{
                    # Duplicate of Paper 1 to test deduplication and belief boosting
                    'cause_text': 'alignment',
                    'cause_polarity': 1,
                    'effect_text': 'safety',
                    'effect_polarity': 1,
                    'evidence_text': 'Our study confirms alignment increases safety.',
                    'paper_id': 'Paper4_Duplicate',
                    'citation': 'DeepMind, 2024',
                }]
            return []

        # 4 Passages
        passages = [
            Passage("Paper 1: Alignment research promotes overall AI safety.", metadata={'paper_id': 'P1'}),
            Passage("Paper 2: Iterative RLHF training provides verifiable safety.", metadata={'paper_id': 'P2'}),
            Passage("Paper 3: RLHF training inadvertently drives sycophantic behavior.", metadata={'paper_id': 'P3'}),
            Passage("Paper 4: Independent replication: Alignment research yields safety.", metadata={'paper_id': 'P4'}),
        ]

        extractor = LLMExtractor(client=mock_paper_extractor, processor=AISafetyProcessor(ontology=ontology))
        pipeline = AssemblyPipeline()  # Preprocessing pipeline
        assembler = IncrementalAssembler(ontology=ontology)

        # Run high-level end-to-end pipeline
        result = process_text_to_graph(
            text_or_passages=passages,
            extractor=extractor,
            pipeline=pipeline,
            assembler=assembler,
            grounding_ontology='AIS',
        )

        # 1. Verify extraction count: 4 raw statements extracted
        self.assertEqual(len(result['raw_statements']), 4)

        # 2. Verify assembly: 4 raw statements collapse to 3 unique statements
        #    (Paper 1 and Paper 4 are duplicates)
        assembled_stmts = result['statements']
        self.assertEqual(len(assembled_stmts), 3)

        # Find the deduplicated statement (alignment -> safety)
        dup_stmt = None
        for s in assembled_stmts:
            if s.subj.concept.name == 'alignment' and s.obj.concept.name == 'safety':
                dup_stmt = s
                break
        self.assertIsNotNone(dup_stmt)

        # 3. Deduplication verified: merged evidence from Paper 1 and Paper 4
        self.assertEqual(len(dup_stmt.evidence), 2)
        pmids = {ev.pmid for ev in dup_stmt.evidence}
        self.assertIn('Paper1_General', pmids)
        self.assertIn('Paper4_Duplicate', pmids)

        # 4. Refinement verified: rlhf -> safety is a refinement of alignment -> safety
        #    Therefore dup_stmt (general) is supported_by the specific rlhf statement
        rlhf_safety_stmt = None
        for s in assembled_stmts:
            if s.subj.concept.name == 'rlhf' and s.obj.concept.name == 'safety':
                rlhf_safety_stmt = s
                break
        self.assertIsNotNone(rlhf_safety_stmt)

        # Check refinement graph link
        self.assertIn(rlhf_safety_stmt, dup_stmt.supported_by)
        self.assertIn(dup_stmt, rlhf_safety_stmt.supports)

        # 5. Belief propagation verified:
        # Belief of dup_stmt includes evidence from itself AND supported_by children!
        self.assertGreater(dup_stmt.belief, 0.85)

        # 6. Incremental delta verified
        delta = result['delta']
        self.assertIsNotNone(delta)
        self.assertEqual(len(delta.new_stmts), 3)
        self.assertGreaterEqual(len(delta.new_refinements), 1)

        # 7. CAG & CytoscapeJS export verified
        cag = result['cag']
        cyjs = result['cytoscapejs']

        # Nodes and edges in networkx MultiDiGraph
        self.assertGreaterEqual(cag.number_of_nodes(), 3)
        self.assertEqual(cag.number_of_edges(), 3)

        # CytoscapeJS structure
        self.assertIn('nodes', cyjs)
        self.assertIn('edges', cyjs)
        self.assertEqual(len(cyjs['edges']), 3)

        # Verify edge data attributes (linecolor, arrow shape, belief)
        for edge in cyjs['edges']:
            edata = edge['data']
            self.assertIn('source', edata)
            self.assertIn('target', edata)
            self.assertIn('linecolor', edata)
            self.assertIn('targetArrowShape', edata)
            self.assertIn('belief', edata)
            self.assertGreater(edata['belief'], 0.0)

        # Verify JSON serialisability of CytoscapeJS payload
        cyjs_str = json.dumps(cyjs)
        self.assertIsInstance(cyjs_str, str)


if __name__ == '__main__':
    unittest.main(verbosity=2)
