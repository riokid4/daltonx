"""daltonx.grounding.grounder
==============================
Main Grounder class and ScoredMatch data structure.
Ported 1-to-1 from gilda-master/gilda/grounder.py.
"""
import os
import csv
import json
import gzip
import logging
import itertools
import collections.abc
from pathlib import Path
from collections import defaultdict, Counter
from typing import Iterator, List, Mapping, Optional, Set, Tuple, Union, Iterable

from .term import Term, get_curie, get_bioregistry_url
from .process import (
    normalize, replace_dashes, replace_greek_uni,
    replace_greek_latin, replace_greek_spelled_out, depluralize,
    replace_roman_arabic
)
from .scorer import Match, generate_match, score

__all__ = [
    "Grounder",
    "ScoredMatch",
    "Annotation",
    "load_terms_file",
    "load_entries_from_terms_file",
    "filter_for_organism",
    "ontology_to_terms",
    "build_grounder_from_ontology",
    "ground_to_concept",
]

logger = logging.getLogger(__name__)

DEFAULT_NAMESPACE_PRIORITY = [
    'AIS', 'FPLX', 'HGNC', 'UP', 'CHEBI', 'GO', 'MESH', 'DOID', 'HP', 'EFO'
]


class Grounder(object):
    """Class to look up and ground query texts into standard ontology concept IDs.

    Parameters
    ----------
    terms : Optional[Union[Iterable[Term], Mapping[str, List[Term]]]]
        The collection of Term objects to index.
    namespace_priority : Optional[List[str]]
        Priority order of namespaces.
    """

    entries: Mapping[str, List[Term]]
    namespace_priority: List[str]

    def __init__(
        self,
        terms: Optional[Union[Iterable[Term], Mapping[str, List[Term]]]] = None,
        *,
        namespace_priority: Optional[List[str]] = None,
    ):
        if terms is None:
            self.entries = {}
        elif isinstance(terms, dict):
            self.entries = terms
        elif isinstance(terms, collections.abc.Iterable):
            self.entries = defaultdict(list)
            for term in terms:
                self.entries[term.norm_text].append(term)
            self.entries = dict(self.entries)
        else:
            raise TypeError('terms is neither a list/iterable of terms nor a dict')

        self.prefix_index = {}
        self._build_prefix_index()

        self.namespace_priority = (
            DEFAULT_NAMESPACE_PRIORITY
            if namespace_priority is None else
            namespace_priority
        )

    def _build_prefix_index(self):
        prefix_index = defaultdict(set)
        for norm_term in self.entries:
            if not norm_term:
                continue
            parts = norm_term.split()
            if not parts:
                continue
            prefix_index[parts[0]].add(len(parts))
        self.prefix_index = dict(prefix_index)

    def lookup(self, raw_str: str) -> List[Term]:
        """Return matching Terms for a given raw string."""
        lookups = self._generate_lookups(raw_str)
        entries = []
        for lookup in lookups:
            entries += self.entries.get(lookup, [])
        return entries

    def _generate_lookups(self, raw_str: str) -> Set[str]:
        norm = normalize(raw_str)
        lookups = {norm}
        norm_spacedash = normalize(replace_dashes(raw_str, ' '))
        lookups.add(norm_spacedash)
        greek_replaced = normalize(replace_greek_uni(raw_str))
        lookups.add(greek_replaced)
        greek_replaced = normalize(replace_greek_latin(raw_str))
        lookups.add(greek_replaced)
        greek_replaced = normalize(replace_greek_spelled_out(raw_str))
        lookups.add(greek_replaced)
        roman_arabic = normalize(replace_roman_arabic(raw_str))
        lookups.add(roman_arabic)
        for singular, _ in depluralize(raw_str):
            lookups.add(normalize(singular))

        logger.debug('Looking up the following strings: %s' % ', '.join(lookups))
        return lookups

    def _score_namespace(self, term) -> int:
        try:
            return len(self.namespace_priority) - self.namespace_priority.index(term.db)
        except ValueError:
            return 0

    def ground_best(
        self,
        raw_str: str,
        context: Optional[str] = None,
        organisms: Optional[List[str]] = None,
        namespaces: Optional[List[str]] = None,
    ) -> Optional["ScoredMatch"]:
        """Return the best scored grounding for a given raw string."""
        scored_matches = self.ground(
            raw_str=raw_str,
            context=context,
            organisms=organisms,
            namespaces=namespaces,
        )
        if scored_matches:
            return scored_matches[0]
        return None

    def ground(self, raw_str: str, context=None, organisms=None,
               namespaces=None) -> List["ScoredMatch"]:
        """Return scored groundings for a given raw string."""
        if not organisms:
            organisms = ['9606']
        raw_str = raw_str.strip()
        entries = self.lookup(raw_str)
        entries = filter_for_organism(entries, organisms)

        scored_matches = []
        for term in entries:
            match = generate_match(raw_str, term.text)
            sc = score(match, term)
            scored_match = ScoredMatch(term, sc, match)
            scored_matches.append(scored_match)

        if not scored_matches:
            return scored_matches

        unique_scores = self._merge_equivalent_matches(scored_matches)

        rank_fun = lambda x: (x.score, self._score_namespace(x.term))
        unique_scores = sorted(unique_scores, key=rank_fun, reverse=True)

        if namespaces:
            unique_scores = [
                sm for sm in unique_scores
                if sm.get_namespaces() & set(namespaces)
            ]

        return unique_scores

    @staticmethod
    def _merge_equivalent_matches(scored_matches):
        unique_entries = []
        term_dbid = lambda x: (x.term.db, x.term.id)
        scored_matches.sort(key=term_dbid)
        entry_groups = itertools.groupby(scored_matches, key=term_dbid)
        for _, entry_group in entry_groups:
            entries = sorted(list(entry_group), key=lambda x: x.score,
                             reverse=True)
            entries[0].subsumed_terms = [e.term for e in entries[1:]]
            unique_entries.append(entries[0])
        return unique_entries

    @classmethod
    def from_ontology(cls, ontology):
        """Construct a Grounder directly from an AISafetyOntology instance."""
        terms = ontology_to_terms(ontology)
        return cls(terms)


class ScoredMatch(object):
    """Class representing a scored match to a grounding term."""

    def __init__(self, term: Term, score, match: Match, disambiguation=None,
                 subsumed_terms=None):
        self.term = term
        self.url = term.get_bioregistry_url()
        self.score = score
        self.match = match
        self.disambiguation = disambiguation
        self.subsumed_terms = subsumed_terms if subsumed_terms else None

    def __str__(self):
        disamb_str = '' if self.disambiguation is None else \
            (',disambiguation=' + json.dumps(self.disambiguation))
        return 'ScoredMatch(%s,%s,%s%s)' % \
            (self.term, self.score, self.match, disamb_str)

    def __repr__(self):
        return str(self)

    def to_json(self):
        js = {
            'term': self.term.to_json(),
            'url': self.url,
            'score': self.score,
            'match': self.match.to_json()
        }
        if self.disambiguation is not None:
            js['disambiguation'] = self.disambiguation
        if self.subsumed_terms:
            js['subsumed_terms'] = [t.to_json() for t in self.subsumed_terms]
        return js

    def multiply(self, value):
        self.score = self.score * value

    def get_namespaces(self) -> Set[str]:
        return {ns for ns, _ in self.get_groundings()}

    def get_groundings(self) -> Set[Tuple[str, str]]:
        term_groundings = self.term.get_groundings()
        if self.subsumed_terms:
            for sub_term in self.subsumed_terms:
                term_groundings |= sub_term.get_groundings()
        return term_groundings


class Annotation:
    """A named entity annotation in a given text."""

    def __init__(self, text: str, matches: List[ScoredMatch], start: int,
                 end: int):
        self.text = text
        self.matches = matches
        self.start = start
        self.end = end

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"Annotation({self.text}, {self.matches}, {self.start}, {self.end})"

    def to_json(self):
        return {
            'text': self.text,
            'matches': [m.to_json() for m in self.matches],
            'start': self.start,
            'end': self.end
        }


def filter_for_organism(terms, organisms):
    terms_by_organism = defaultdict(list)
    for term in terms:
        if term.organism is not None and term.organism not in organisms:
            continue
        terms_by_organism[term.organism].append(term)
    all_terms = terms_by_organism[None]
    if set(terms_by_organism) != {None}:
        top_organism = min(set(terms_by_organism) - {None},
                           key=lambda x: organisms.index(x))
        all_terms += terms_by_organism[top_organism]
    return all_terms


def load_entries_from_terms_file(terms_file: Union[str, Path]) -> Iterator[Term]:
    with gzip.open(terms_file, 'rt', encoding='utf-8') as fh:
        reader = csv.reader(fh, delimiter='\t')
        next(reader)
        for row in reader:
            row_nones = [r if r else None for r in row]
            yield Term(*row_nones)


def load_terms_file(terms_file: Union[str, Path]) -> Mapping[str, List[Term]]:
    entries = {}
    for term in load_entries_from_terms_file(terms_file):
        if term.norm_text in entries:
            entries[term.norm_text].append(term)
        else:
            entries[term.norm_text] = [term]
    return entries


def ontology_to_terms(ontology) -> List[Term]:
    """Convert an ontology (like AISafetyOntology) into a list of Gilda Term objects.

    For each node in the ontology graph:
      - node name -> Term with status='name'
      - each example text -> Term with status='synonym'
    """
    if not ontology._initialized:
        ontology.initialize()

    terms = []
    for node, data in ontology.nodes(data=True):
        ns, node_id = ontology.get_ns_id(node)
        name = data.get('name', node_id.split('/')[-1])

        # Primary name term
        terms.append(
            Term(
                norm_text=normalize(name),
                text=name,
                db=ns,
                id=node_id,
                entry_name=name,
                status='name',
                source=ns
            )
        )

        # Full path as synonym if multi-part
        if '/' in node_id:
            terms.append(
                Term(
                    norm_text=normalize(node_id),
                    text=node_id,
                    db=ns,
                    id=node_id,
                    entry_name=name,
                    status='synonym',
                    source=ns
                )
            )

        # Examples from YAML as synonyms
        for ex in data.get('examples', []):
            terms.append(
                Term(
                    norm_text=normalize(ex),
                    text=ex,
                    db=ns,
                    id=node_id,
                    entry_name=name,
                    status='synonym',
                    source=ns
                )
            )

    return terms


def build_grounder_from_ontology(ontology=None) -> Grounder:
    """Build and return a Grounder instance from an ontology (default: bundled AISafetyOntology).

    Parameters
    ----------
    ontology : Optional[AISafetyOntology]
        The ontology instance to extract terms from. If None, loads the default
        bundled AI Safety ontology.

    Returns
    -------
    Grounder
        A Grounder populated with terms from the ontology.
    """
    if ontology is None:
        from daltonx.ontology import load_ai_safety_ontology
        ontology = load_ai_safety_ontology()
    terms = ontology_to_terms(ontology)
    return Grounder(terms)


def ground_to_concept(raw_str: str, grounder: Grounder):
    """Ground a raw string to an official AIS concept, returning a daltonx Concept with db_refs['AIS'].

    Parameters
    ----------
    raw_str : str
        The messy input string.
    grounder : Grounder
        The grounder instance.

    Returns
    -------
    Concept
        A Concept instance from daltonx.statements with db_refs populated.
    """
    from daltonx.statements import Concept
    best_match = grounder.ground_best(raw_str)
    if best_match:
        return Concept(
            name=best_match.term.entry_name,
            db_refs={best_match.term.db: best_match.term.id}
        )
    return Concept(name=raw_str, db_refs={})

