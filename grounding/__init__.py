"""daltonx.grounding
==================
Grounding engine for mapping noisy text to standardized ontology concepts.
Ported 1-to-1 from gilda.
"""

from .term import Term, get_curie, filter_out_duplicates
from .process import (
    normalize,
    replace_dashes,
    remove_dashes,
    replace_whitespace,
    depluralize,
    get_capitalization_pattern,
    replace_greek_uni,
    replace_greek_latin,
    replace_greek_spelled_out,
    replace_unicode,
    replace_roman_arabic,
)
from .scorer import Match, generate_match, score_string_match, score_status, score
from .grounder import (
    Grounder,
    ScoredMatch,
    Annotation,
    ontology_to_terms,
    build_grounder_from_ontology,
    ground_to_concept,
)

__all__ = [
    "Term",
    "Grounder",
    "ScoredMatch",
    "Annotation",
    "ontology_to_terms",
    "build_grounder_from_ontology",
    "ground_to_concept",
    "normalize",
    "replace_dashes",
    "remove_dashes",
    "replace_whitespace",
    "depluralize",
    "get_capitalization_pattern",
    "replace_greek_uni",
    "replace_greek_latin",
    "replace_greek_spelled_out",
    "replace_unicode",
    "replace_roman_arabic",
    "Match",
    "generate_match",
    "score_string_match",
    "score_status",
    "score",
]
