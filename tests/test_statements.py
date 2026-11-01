import json
import unittest

from daltonx.statements import (
    Concept,
    QualitativeDelta,
    QuantitativeState,
    RefContext,
    TimeContext,
    AISafetyContext,
    Evidence,
    Event,
    Influence,
    Statement
)


class TestStatements(unittest.TestCase):
    def setUp(self):
        self.cause_concept = Concept('reward_hacking', db_refs={'AI_ONT': 'ai/alignment/outer/reward_hacking'})
        self.cause_delta = QualitativeDelta(polarity=1, adjectives=['excessive'])
        self.cause_context = AISafetyContext(
            environment=RefContext('RLHF_Sandbox', db_refs={'ENV': 'simulation/rlhf'}),
            model=RefContext('Llama-3-70B', db_refs={'MODEL': 'meta/llama3-70b'})
        )
        self.cause_ev = Evidence(
            source_api='llm_extractor',
            source_id='paper_arxiv_2405',
            text='Excessive reward hacking significantly increases deceptive alignment in frontier LLMs.',
            annotations={'confidence': 0.95}
        )
        self.cause_event = Event(self.cause_concept, delta=self.cause_delta,
                                 context=self.cause_context, evidence=[self.cause_ev])

        self.effect_concept = Concept('deceptive_alignment', db_refs={'AI_ONT': 'ai/alignment/inner/deceptive_alignment'})
        self.effect_delta = QualitativeDelta(polarity=1, adjectives=['emergent'])
        self.effect_context = AISafetyContext(
            environment=RefContext('RLHF_Sandbox', db_refs={'ENV': 'simulation/rlhf'}),
            model=RefContext('Llama-3-70B', db_refs={'MODEL': 'meta/llama3-70b'})
        )
        self.effect_event = Event(self.effect_concept, delta=self.effect_delta, context=self.effect_context)

        self.influence = Influence(self.cause_event, self.effect_event, evidence=[self.cause_ev])

    def test_influence_creation_and_attributes(self):
        self.assertEqual(self.influence.overall_polarity(), 1)
        self.assertEqual(self.influence.polarity_count(), 2)
        self.assertEqual(self.influence.subj.concept.name, 'reward_hacking')
        self.assertEqual(self.influence.obj.concept.name, 'deceptive_alignment')
        self.assertEqual(len(self.influence.evidence), 1)

    def test_hashing(self):
        shallow_hash = self.influence.get_hash(shallow=True)
        full_hash = self.influence.get_hash(shallow=False)
        self.assertIsInstance(shallow_hash, int)
        self.assertIsInstance(full_hash, int)
        self.assertNotEqual(shallow_hash, full_hash)
        # Idempotent
        self.assertEqual(shallow_hash, self.influence.get_hash(shallow=True))

    def test_json_roundtrip(self):
        json_dict = self.influence.to_json()
        self.assertEqual(json_dict['type'], 'Influence')
        self.assertIn('subj', json_dict)
        self.assertIn('obj', json_dict)
        self.assertIn('matches_hash', json_dict)

        # Deserialize back
        recovered_stmt = Statement._from_json(json_dict)
        self.assertIsInstance(recovered_stmt, Influence)
        self.assertEqual(recovered_stmt.subj.concept.name, 'reward_hacking')
        self.assertEqual(recovered_stmt.obj.concept.name, 'deceptive_alignment')
        self.assertEqual(recovered_stmt.overall_polarity(), 1)
        self.assertEqual(recovered_stmt.get_hash(shallow=True), self.influence.get_hash(shallow=True))
        self.assertEqual(len(recovered_stmt.evidence), 1)
        self.assertIsInstance(recovered_stmt.subj.context, AISafetyContext)

    def test_polarity_flip(self):
        inf_copy = self.influence.make_generic_copy()
        self.assertEqual(inf_copy.overall_polarity(), 1)
        inf_copy.flip_polarity(0) # flip cause from 1 to -1
        self.assertEqual(inf_copy.overall_polarity(), -1)
        inf_copy.flip_polarity(1) # flip effect from 1 to -1 -> (-1 * -1) = 1
        self.assertEqual(inf_copy.overall_polarity(), 1)

    def test_quantitative_state(self):
        qs1 = QuantitativeState(entity='alignment_score', value=10, unit='second')
        qs2 = QuantitativeState(entity='alignment_score', value=5, unit='second')
        diff, unit = qs1 - qs2
        self.assertEqual(diff, 5)
        self.assertEqual(unit, 'second')
        self.assertTrue(qs1 > qs2)


if __name__ == '__main__':
    unittest.main()
