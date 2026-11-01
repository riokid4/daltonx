"""daltonx.assembly.preassembler
==============================
De-duplicates statements, builds specificity/refinement hierarchies,
merges evidence with full provenance tracking, and finds contradictions.
Ported 1-to-1 from indra-master/indra/preassembler/__init__.py.
"""

__all__ = [
    'Preassembler',
    'find_refinements_for_statement',
    'render_stmt_graph',
    'flatten_stmts',
    'flatten_evidence',
    'default_matches_fun',
]

import copy
import time
import logging
import itertools
import functools
import collections
import networkx as nx

from daltonx.statements import (
    Statement,
    Influence,
    Event,
    stmt_type as daltonx_stmt_type,
)
from .refinement import (
    RefinementFilter,
    OntologyRefinementFilter,
    RefinementConfirmationFilter,
    SplitGroupFilter,
    default_refinement_fun,
)

logger = logging.getLogger(__name__)


def fast_deepcopy(obj):
    """Deepcopy statement or list of statements."""
    return copy.deepcopy(obj)


def default_matches_fun(st):
    """Default matches key function."""
    return st.matches_key()


class Preassembler(object):
    """De-duplicates statements and arranges them in a specificity hierarchy.

    Parameters
    ----------
    ontology : AISafetyOntology
        An ontology graph instance.
    stmts : Optional[list[Statement]]
        A list of statements to pre-assemble.
    matches_fun : Optional[callable]
        Custom function returning a grouping key for duplicate identification.
    refinement_fun : Optional[callable]
        Custom function returning True if statement 1 refines statement 2.
    """

    def __init__(self, ontology, stmts=None, matches_fun=None, refinement_fun=None):
        self.ontology = ontology
        if stmts:
            self.stmts = fast_deepcopy(stmts)
        else:
            self.stmts = []
        self.unique_stmts = None
        self.related_stmts = None
        self.matches_fun = matches_fun if matches_fun else default_matches_fun
        self.refinement_fun = refinement_fun if refinement_fun else default_refinement_fun
        self._comparison_counter = 0

    def add_statements(self, stmts):
        """Add to the current list of statements."""
        self.stmts += fast_deepcopy(stmts)

    def combine_duplicates(self):
        """Combine duplicates among `stmts` and save result in `unique_stmts`."""
        if self.unique_stmts is None:
            self.unique_stmts = self.combine_duplicate_stmts(self.stmts)
        return self.unique_stmts

    def _get_stmt_matching_groups(self, stmts):
        """Use the matches_fun method to get sets of matching statements."""
        st = list(set(stmts))
        st.sort(key=self.matches_fun)
        return itertools.groupby(st, key=self.matches_fun)

    def combine_duplicate_stmts(self, stmts):
        """Combine evidence from duplicate Statements."""
        def _ev_keys(sts):
            ev_keys = []
            for stmt in sts:
                for ev in stmt.evidence:
                    ev_keys.append(ev.matches_key())
            return ev_keys

        unique_stmts = []
        for _, duplicates in self._get_stmt_matching_groups(stmts):
            ev_keys = set()
            duplicates = list(duplicates)
            start_ev_keys = _ev_keys(duplicates)
            new_stmt = None

            for stmt_ix, stmt in enumerate(duplicates):
                if stmt_ix == 0:
                    new_stmt = stmt.make_generic_copy()
                if len(duplicates) == 1:
                    new_stmt.uuid = stmt.uuid

                raw_text = [
                    None if ag is None else ag.db_refs.get('TEXT')
                    for ag in stmt.agent_list(deep_sorted=True)
                ]
                raw_grounding = [
                    None if ag is None else ag.db_refs
                    for ag in stmt.agent_list(deep_sorted=True)
                ]

                for ev in stmt.evidence:
                    ev_key = ev.matches_key() + str(raw_text) + str(raw_grounding)
                    if ev_key not in ev_keys:
                        if 'agents' in ev.annotations:
                            ev.annotations['agents']['raw_text'] = raw_text
                            ev.annotations['agents']['raw_grounding'] = raw_grounding
                        else:
                            ev.annotations['agents'] = {
                                'raw_text': raw_text,
                                'raw_grounding': raw_grounding,
                            }
                        if 'prior_uuids' not in ev.annotations:
                            ev.annotations['prior_uuids'] = []
                        ev.annotations['prior_uuids'].append(stmt.uuid)
                        new_stmt.evidence.append(ev)
                        ev_keys.add(ev_key)

            end_ev_keys = _ev_keys([new_stmt])
            if len(end_ev_keys) != len(start_ev_keys):
                logger.debug('%d redundant evidences eliminated.' % (len(start_ev_keys) - len(end_ev_keys)))

            assert isinstance(new_stmt, Statement)
            unique_stmts.append(new_stmt)

        for stmt in unique_stmts:
            for shallow in (True, False):
                stmt.get_hash(shallow=shallow, refresh=True, matches_fun=self.matches_fun)

        return unique_stmts

    def combine_related(self, return_toplevel=True, filters=None, **kwargs):
        """Connect related statements based on their refinement relationships."""
        if self.related_stmts is not None:
            if return_toplevel:
                return self.related_stmts
            else:
                assert self.unique_stmts is not None
                return self.unique_stmts

        unique_stmts = self.combine_duplicates()
        idx_map = self._generate_id_maps(unique_stmts, filters=filters)

        for ix1, ix2 in idx_map:
            unique_stmts[ix1].supported_by.append(unique_stmts[ix2])
            unique_stmts[ix2].supports.append(unique_stmts[ix1])

        self.related_stmts = [st for st in unique_stmts if not st.supports]
        logger.debug('%d top level' % len(self.related_stmts))
        if return_toplevel:
            return self.related_stmts
        else:
            return unique_stmts

    def _generate_relation_tuples(self, unique_stmts, split_idx=None, filters=None):
        """Return refinement relations as a set of statement hash tuples."""
        relations = self._generate_relations(
            unique_stmts=unique_stmts,
            split_idx=split_idx,
            filters=filters
        )
        relation_tuples = set()
        for refiner, refineds in relations.items():
            relation_tuples |= {(refiner, refined) for refined in refineds}
        return relation_tuples

    def _generate_relations(self, unique_stmts, split_idx=None, filters=None):
        """Return refinement relations as a dict using statement hashes."""
        ts = time.time()
        stmts_by_hash = {
            stmt.get_hash(matches_fun=self.matches_fun): stmt
            for stmt in unique_stmts
        }

        if not filters:
            filters = [OntologyRefinementFilter(ontology=self.ontology)]
        else:
            filters = list(filters)

        if split_idx:
            stmt_to_idx = {
                stmt.get_hash(matches_fun=self.matches_fun): idx
                for idx, stmt in enumerate(unique_stmts)
            }
            split_groups = {sh: (idx <= split_idx) for sh, idx in stmt_to_idx.items()}
            sgf = SplitGroupFilter(split_groups=split_groups)
            filters.append(sgf)

        confirm_filter = RefinementConfirmationFilter(
            ontology=self.ontology,
            refinement_fun=self.refinement_fun
        )
        filters.append(confirm_filter)

        for filt in filters:
            filt.initialize(stmts_by_hash=stmts_by_hash)

        relations = {}
        for stmt_hash, stmt in stmts_by_hash.items():
            rels = find_refinements_for_statement(stmt, filters)
            if rels:
                relations[stmt_hash] = rels

        te = time.time()
        logger.info('Found %d refinements in %.2fs' % (sum([len(v) for v in relations.values()]), te - ts))
        self._comparison_counter = confirm_filter.comparison_counter
        return relations

    def _generate_id_maps(self, unique_stmts, split_idx=None, filters=None, **kwargs):
        """Return pairs of statement indices representing refinement relations."""
        stmt_to_idx = {
            stmt.get_hash(matches_fun=self.matches_fun): idx
            for idx, stmt in enumerate(unique_stmts)
        }
        if len(unique_stmts) != len(stmt_to_idx):
            raise ValueError(
                'The unique statements used as an input for finding refinements '
                'do not all have distinct matches key hashes.'
            )
        relation_tuples = self._generate_relation_tuples(
            unique_stmts,
            split_idx=split_idx,
            filters=filters
        )
        idx_maps = [
            (stmt_to_idx[refiner], stmt_to_idx[refined])
            for refiner, refined in relation_tuples
        ]
        return idx_maps

    def find_contradicts(self):
        """Return pairs of contradicting Statements."""
        stmts_by_type = collections.defaultdict(list)
        for stmt in self.stmts:
            stmts_by_type[daltonx_stmt_type(stmt)].append(stmt)
        stmts_by_type = dict(stmts_by_type)

        contradicts = []
        neu_stmts = [Influence]
        for stt in neu_stmts:
            stmts = stmts_by_type.get(stt, [])
            for st1, st2 in itertools.combinations(stmts, 2):
                if st1.contradicts(st2, self.ontology):
                    contradicts.append((st1, st2))

        return contradicts

    def _normalize_relations(self, ns, rank_key, rel_fun, flip_polarity):
        def _replace_grounding(ns, entry, rank_key, rel_fun):
            rel_ents = list(rel_fun(ns, entry))
            if rel_ents:
                rel_ents = [
                    (ns, e[1].split('#')[1] if '#' in e[1] else e[1])
                    for e in rel_ents
                ]
                sorted_entries = sorted([(ns, entry)] + rel_ents, key=rank_key)
                _, chosen = sorted_entries[0]
                return chosen, chosen != entry
            else:
                return entry, False

        if rank_key is None:
            def polarity_rank_key(args):
                ns, entry = args
                pol = self.ontology.get_polarity(ns, entry)
                pol_rank = -1 if pol is None else -pol
                return pol_rank, entry
            rank_key = polarity_rank_key

        for stmt in self.stmts:
            for agent_idx, agent in enumerate(stmt.agent_list()):
                if agent is not None and ns in agent.db_refs:
                    grounding = agent.db_refs[ns]
                    if isinstance(grounding, list):
                        new_grounding = []
                        for idx, (entry, sc) in enumerate(grounding):
                            chosen, changed = _replace_grounding(ns, entry, rank_key, rel_fun)
                            new_grounding.append((chosen, sc))
                            if idx == 0 and changed and flip_polarity:
                                stmt.flip_polarity(agent_idx=agent_idx)
                        agent.db_refs[ns] = new_grounding
                    else:
                        chosen, changed = _replace_grounding(ns, grounding, rank_key, rel_fun)
                        agent.db_refs[ns] = chosen
                        if changed and flip_polarity:
                            stmt.flip_polarity(agent_idx=agent_idx)

    def normalize_opposites(self, ns='AIS', rank_key=None):
        """Normalize to one of a pair of opposite concepts across statements."""
        rel_fun = functools.partial(self.ontology.child_rel, rel_types={'is_opposite'})
        self._normalize_relations(ns, rank_key, rel_fun, True)


def find_refinements_for_statement(stmt, filters):
    """Return refinements for a single statement given initialized filters."""
    first_filter = True
    relations = set()
    for filt in filters:
        possibly_related = None if first_filter else relations
        relations = filt.get_less_specifics(stmt, possibly_related=possibly_related)
        first_filter = False
    return relations


def render_stmt_graph(statements, reduce=True):
    """Build a networkx directed graph representing the statement hierarchy."""
    edges = set()
    stmt_dict = {}

    def process_stmt(stmt):
        key = stmt.matches_key()
        stmt_dict[key] = stmt
        for sby in stmt.supported_by:
            edges.add((key, sby.matches_key()))
            process_stmt(sby)

    for stmt in statements:
        process_stmt(stmt)

    g = nx.DiGraph()
    g.add_edges_from(edges)
    if reduce and len(g) > 0:
        g = nx.algorithms.dag.transitive_reduction(g)
    return g


def flatten_stmts(stmts):
    """Return the full set of unique statements in a pre-assembled statement hierarchy."""
    total_stmts = set(stmts)
    for stmt in stmts:
        if stmt.supported_by:
            children = flatten_stmts(stmt.supported_by)
            total_stmts = total_stmts.union(children)
    return list(total_stmts)


def flatten_evidence(stmts, collect_from='supported_by'):
    """Collect evidence from supporting or supported statements."""
    if collect_from not in ('supports', 'supported_by'):
        raise ValueError('collect_from must be "supports" or "supported_by"')

    stmts = fast_deepcopy(stmts)
    for stmt in stmts:
        orig_ev_keys = [ev.matches_key() for ev in stmt.evidence]
        total_evidence = _flatten_evidence_for_stmt(stmt, collect_from)
        new_evidence = []
        for ev in total_evidence:
            ev_key = ev.matches_key()
            if ev_key in orig_ev_keys:
                ev.annotations['support_type'] = 'direct'
                new_evidence.append(ev)
            else:
                ev_copy = fast_deepcopy(ev)
                ev_copy.annotations['support_type'] = collect_from
                new_evidence.append(ev_copy)
        stmt.evidence = new_evidence
    return stmts


def _flatten_evidence_for_stmt(stmt, collect_from):
    supp_stmts = (stmt.supports if collect_from == 'supports' else stmt.supported_by)
    total_evidence = set(stmt.evidence)
    for supp_stmt in supp_stmts:
        child_evidence = _flatten_evidence_for_stmt(supp_stmt, collect_from)
        total_evidence = total_evidence.union(child_evidence)
    return list(total_evidence)
