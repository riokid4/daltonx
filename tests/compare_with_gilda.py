"""daltonx/tests/compare_with_gilda.py
=======================================
Comparative benchmark: Grounds 20 messy text strings using:
  1) Original gilda-master code (gilda.grounder.Grounder)
  2) Our standalone port (daltonx.grounding.Grounder)

Prints a side-by-side comparison table of matched concept IDs and scores,
and asserts 100% equivalence between both implementations.
"""
import sys
import os

# Add workspace and gilda-master to python path
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
GILDA_DIR = os.path.join(WORKSPACE_DIR, 'gilda-master')
if WORKSPACE_DIR not in sys.path:
    sys.path.insert(0, WORKSPACE_DIR)
if GILDA_DIR not in sys.path:
    sys.path.insert(0, GILDA_DIR)

from gilda.grounder import Grounder as OrigGrounder
from gilda.term import Term as OrigTerm

from daltonx.ontology import load_ai_safety_ontology
from daltonx.grounding import (
    Grounder as DaltonxGrounder,
    Term as DaltonxTerm,
    ontology_to_terms
)

# 20 diverse, messy test strings covering real-world noise:
# - casing differences (ALL CAPS, lower, mixed)
# - hyphens vs spaces
# - leading/trailing whitespace
# - pluralization (-s, -es)
# - partial/synonymous phrases from AI Safety literature
TEST_STRINGS = [
    "  RLHF  ",
    "rlhf",
    "reinforcement-learning-from-human-feedback",
    "Reinforcement Learning from Human Feedback",
    "sparse autoencoders",
    "sparse-autoencoder",
    "SAE features",
    "dictionary learning on activations",
    "mechanistic-interpretability",
    "reverse engineering transformer circuits",
    "sycophantic ai",
    "model flattery",
    "telling users what they want to hear",
    "reward-hacking",
    "gaming the reward signal",
    "specification gaming",
    "deceptive-alignment",
    "mesa-optimizer deception",
    "supervising models smarter than us",
    "AI-safety",
]


def run_comparison():
    print("=" * 88)
    print("GILDA VS. DALTONX GROUNDING COMPARISON (20 MESSY STRINGS)")
    print("=" * 88)

    # 1. Load ontology and convert to Term instances
    onto = load_ai_safety_ontology()
    onto.initialize()
    daltonx_terms = ontology_to_terms(onto)

    # Replicate terms as original gilda Term objects
    orig_terms = [
        OrigTerm(
            t.norm_text, t.text, t.db, t.id, t.entry_name, t.status, t.source,
            t.organism, t.source_db, t.source_id
        ) for t in daltonx_terms
    ]

    print(f"Loaded {len(daltonx_terms)} terms from AISafetyOntology into both grounders.\n")

    # 2. Instantiate both Grounders
    orig_grounder = OrigGrounder(orig_terms)
    daltonx_grounder = DaltonxGrounder(daltonx_terms)

    # 3. Ground all 20 strings with both and compare
    header = f"{'#':<3} | {'Query String':<35} | {'Orig ID':<20} | {'Orig Score':<10} | {'Daltonx ID':<20} | {'Daltonx Score':<12} | {'Match?'}"
    print(header)
    print("-" * len(header))

    differences = []

    for idx, query in enumerate(TEST_STRINGS, 1):
        # Ground with original gilda
        orig_matches = orig_grounder.ground(query)
        orig_best = orig_matches[0] if orig_matches else None
        orig_id = orig_best.term.id if orig_best else "None"
        orig_score = round(orig_best.score, 6) if orig_best else 0.0

        # Ground with daltonx port
        daltonx_matches = daltonx_grounder.ground(query)
        daltonx_best = daltonx_matches[0] if daltonx_matches else None
        daltonx_id = daltonx_best.term.id if daltonx_best else "None"
        daltonx_score = round(daltonx_best.score, 6) if daltonx_best else 0.0

        # Compare IDs and scores
        id_match = (orig_id == daltonx_id)
        score_diff = abs(orig_score - daltonx_score)
        score_match = score_diff < 1e-6
        is_identical = id_match and score_match

        status_str = "OK" if is_identical else "DIFF"
        if not is_identical:
            differences.append({
                "query": query,
                "orig_id": orig_id,
                "daltonx_id": daltonx_id,
                "orig_score": orig_score,
                "daltonx_score": daltonx_score
            })

        print(f"{idx:<3} | {query:<35} | {orig_id:<20} | {orig_score:<10.6f} | {daltonx_id:<20} | {daltonx_score:<12.6f} | {status_str}")

    print("-" * len(header))
    print(f"\nTotal Queries: {len(TEST_STRINGS)}")
    print(f"Identical Matches: {len(TEST_STRINGS) - len(differences)}")
    print(f"Differences: {len(differences)}")

    if differences:
        print("\nDIFFERENCES DETECTED:")
        for d in differences:
            print(f"  Query: '{d['query']}'")
            print(f"    Orig:    ID={d['orig_id']}, score={d['orig_score']}")
            print(f"    Daltonx: ID={d['daltonx_id']}, score={d['daltonx_score']}")
        sys.exit(1)
    else:
        print("\nSUCCESS: 100% agreement between original Gilda and daltonx port across all 20 messy strings!")


if __name__ == '__main__':
    run_comparison()
