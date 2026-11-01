"""daltonx.assembly.refinement
============================
Refinement filters and algorithms for finding hierarchical relationships
between statements.
Ported 1-to-1 from:
  - indra-master/indra/preassembler/refinement.py
  - indra_world-master/indra_world/assembly/refinement.py
"""

__all__ = [
    'get_agent_key',
    'get_relevant_keys',
    'RefinementFilter',
    'RefinementConfirmationFilter',
    'OntologyRefinementFilter',
    'SplitGroupFilter',
    'default_refinement_fun',
    'compositional_refinement',
    'event_compositional_refinement',
    'location_refinement',
    'location_refinement_compositional',
    'location_time_refinement',
    'location_time_delta_refinement',
    'CompositionalRefinementFilter',
    'get_relevants_for_stmt',
]

import collections
import logging
from daltonx.statements import Event, Influence, stmt_type as daltonx_stmt_type
from .matches import has_location, has_time, get_location, concept_matches_compositional

logger = logging.getLogger(__name__)


def get_agent_key(agent, comp_idx=None):
    """Return a key for an Agent/Concept for use in refinement finding."""
    if isinstance(agent, Event):
        agent = agent.concept
    if agent is None:
        return None

    gr = agent.get_grounding(ns_order=['AIS'])
    if gr[0] is None:
        if comp_idx is None or comp_idx == 0:
            agent_key = ('NAME', agent.name)
        else:
            agent_key = None
    else:
        if comp_idx is not None and isinstance(gr[1], (list, tuple)):
            comp_gr = gr[1][comp_idx] if comp_idx < len(gr[1]) else None
            agent_key = ('AIS', comp_gr) if comp_gr else None
        else:
            val = gr[1]
            if isinstance(val, (list, tuple)) and len(val) > 0:
                val = val[0]
            agent_key = (gr[0], str(val))
    return agent_key


def get_relevant_keys(agent_key, all_keys_for_role, ontology, direction):
    """Return relevant agent keys for an agent key for refinement finding."""
    rel_fun = ontology.get_parents if direction == 'less_specific' else \
        ontology.get_children
    relevant_keys = {None, agent_key}
    if agent_key is not None:
        relevant_keys |= set(rel_fun(*agent_key))
    relevant_keys &= all_keys_for_role
    return relevant_keys


class RefinementFilter:
    """Base class for pre-filters applied to candidate refinements."""

    def __init__(self):
        self.shared_data = {}

    def initialize(self, stmts_by_hash):
        self.shared_data['stmts_by_hash'] = stmts_by_hash

    def get_related(self, stmt, possibly_related=None, direction='less_specific'):
        raise NotImplementedError('Subclasses must implement get_related.')

    def get_more_specifics(self, stmt, possibly_related=None):
        return self.get_related(stmt, possibly_related=possibly_related, direction='more_specific')

    def get_less_specifics(self, stmt, possibly_related=None):
        return self.get_related(stmt, possibly_related=possibly_related, direction='less_specific')

    def extend(self, stmts_by_hash):
        self.shared_data['stmts_by_hash'].update(stmts_by_hash)


class OntologyRefinementFilter(RefinementFilter):
    """Uses the ontology to filter candidate refinements based on paths in the ontology graph."""

    def __init__(self, ontology):
        super().__init__()
        self.ontology = ontology

    def initialize(self, stmts_by_hash):
        self.shared_data['stmts_by_hash'] = {}
        self.extend(stmts_by_hash)

    def extend(self, stmts_by_hash):
        self.shared_data['stmts_by_hash'].update(stmts_by_hash)
        stmts_by_type = collections.defaultdict(set)
        for stmt_hash, stmt in stmts_by_hash.items():
            stmts_by_type[daltonx_stmt_type(stmt)].add(stmt_hash)
        stmts_by_type = dict(stmts_by_type)

        for stmt_type_str, stmts_this_type in stmts_by_type.items():
            first_stmt = stmts_by_hash[next(iter(stmts_this_type))]
            roles = getattr(first_stmt, '_agent_order', ('subj', 'obj') if isinstance(first_stmt, Influence) else ('concept',))
            if stmt_type_str not in self.shared_data:
                self.shared_data[stmt_type_str] = {}
                self.shared_data[stmt_type_str]['agent_key_to_hash'] = \
                    {role: collections.defaultdict(set) for role in roles}
                self.shared_data[stmt_type_str]['hash_to_agent_key'] = \
                    {role: collections.defaultdict(set) for role in roles}
                self.shared_data[stmt_type_str]['all_keys_by_role'] = {}

            for sh in stmts_this_type:
                for role in roles:
                    agent_keys = self._agent_keys_for_stmt_role(stmts_by_hash[sh], role)
                    for agent_key in agent_keys:
                        self.shared_data[stmt_type_str]['agent_key_to_hash'][role][agent_key].add(sh)
                        self.shared_data[stmt_type_str]['hash_to_agent_key'][role][sh].add(agent_key)

            for role in roles:
                self.shared_data[stmt_type_str]['all_keys_by_role'][role] = \
                    set(self.shared_data[stmt_type_str]['agent_key_to_hash'][role])

    @staticmethod
    def _agent_keys_for_stmt_role(stmt, role):
        agents = getattr(stmt, role)
        if isinstance(agents, list) and not agents:
            return {None}
        else:
            return {get_agent_key(ag) for ag in (agents if isinstance(agents, list) else [agents])}

    def get_related(self, stmt, possibly_related=None, direction='less_specific'):
        stmt_type_str = daltonx_stmt_type(stmt)
        if stmt_type_str not in self.shared_data:
            return set()

        hash_to_agent_key = self.shared_data[stmt_type_str]['hash_to_agent_key']
        agent_key_to_hash = self.shared_data[stmt_type_str]['agent_key_to_hash']
        all_keys_by_role = self.shared_data[stmt_type_str]['all_keys_by_role']

        stmt_hash = stmt.get_hash()
        relevants = possibly_related

        for role, hash_to_agent_key_for_role in hash_to_agent_key.items():
            if stmt_hash in hash_to_agent_key_for_role:
                agent_keys = hash_to_agent_key_for_role[stmt_hash]
            else:
                agent_keys = self._agent_keys_for_stmt_role(stmt, role)

            for agent_key in agent_keys:
                relevant_keys = get_relevant_keys(
                    agent_key,
                    all_keys_by_role[role],
                    self.ontology,
                    direction=direction
                )
                role_relevant_stmt_hashes = set.union(
                    *[agent_key_to_hash[role][rel] for rel in relevant_keys]
                ) - {stmt_hash}

                if relevants is None:
                    relevants = role_relevant_stmt_hashes
                else:
                    relevants &= role_relevant_stmt_hashes

        return relevants if relevants is not None else set()


class CompositionalRefinementFilter(RefinementFilter):
    """Refinement filter supporting compositional concept grounding."""

    def __init__(self, ontology, nproc=None):
        super().__init__()
        self.ontology = ontology
        self.nproc = nproc

    def initialize(self, stmts_by_hash):
        super().initialize(stmts_by_hash)
        agent_key_to_hash = {}
        hash_to_agent_key = {}
        all_keys_by_role = {}
        comp_idxes = list(range(4))

        if stmts_by_hash:
            first = stmts_by_hash[next(iter(stmts_by_hash))]
            roles = getattr(first, '_agent_order', ('subj', 'obj') if isinstance(first, Influence) else ('concept',))
        else:
            roles = ('subj', 'obj')

        for role in roles:
            agent_key_to_hash[role] = {}
            hash_to_agent_key[role] = {}
            for comp_idx in comp_idxes:
                agent_key_to_hash[role][comp_idx] = collections.defaultdict(set)
                hash_to_agent_key[role][comp_idx] = collections.defaultdict(set)

        self._extend_maps(roles, stmts_by_hash, agent_key_to_hash, hash_to_agent_key, all_keys_by_role)
        self.shared_data['agent_key_to_hash'] = agent_key_to_hash
        self.shared_data['hash_to_agent_key'] = hash_to_agent_key
        self.shared_data['all_keys_by_role'] = all_keys_by_role
        self.shared_data['roles'] = roles

    @staticmethod
    def _extend_maps(roles, stmts_by_hash, agent_key_to_hash, hash_to_agent_key, all_keys_by_role):
        comp_idxes = list(range(4))
        for sh, stmt in stmts_by_hash.items():
            for role in roles:
                agents = getattr(stmt, role)
                for comp_idx in comp_idxes:
                    agent_keys = {get_agent_key(agent, comp_idx) for agent in
                                  (agents if isinstance(agents, list) else [agents])}
                    for agent_key in agent_keys:
                        agent_key_to_hash[role][comp_idx][agent_key].add(sh)
                        hash_to_agent_key[role][comp_idx][sh].add(agent_key)
        for role in roles:
            all_keys_by_role[role] = {}
            for comp_idx in comp_idxes:
                all_keys_by_role[role][comp_idx] = set(agent_key_to_hash[role][comp_idx].keys())

    def extend(self, stmts_by_hash):
        if not stmts_by_hash:
            return
        first = stmts_by_hash[next(iter(stmts_by_hash))]
        roles = getattr(first, '_agent_order', ('subj', 'obj') if isinstance(first, Influence) else ('concept',))
        self._extend_maps(roles, stmts_by_hash,
                          self.shared_data['agent_key_to_hash'],
                          self.shared_data['hash_to_agent_key'],
                          self.shared_data['all_keys_by_role'])
        self.shared_data['stmts_by_hash'].update(stmts_by_hash)

    def get_related(self, stmt, possibly_related=None, direction='less_specific'):
        sh = stmt.get_hash()
        all_keys_by_role = self.shared_data['all_keys_by_role']
        agent_key_to_hash = self.shared_data['agent_key_to_hash']
        hash_to_agent_key = self.shared_data['hash_to_agent_key']
        relevants = get_relevants_for_stmt(
            sh,
            all_keys_by_role=all_keys_by_role,
            agent_key_to_hash=agent_key_to_hash,
            hash_to_agent_key=hash_to_agent_key,
            ontology=self.ontology,
            direction=direction
        )
        return relevants if relevants is not None else set()


def get_relevants_for_stmt(sh, all_keys_by_role, agent_key_to_hash, hash_to_agent_key, ontology, direction):
    relevants = None
    for role in hash_to_agent_key:
        for comp_idx in hash_to_agent_key[role]:
            keys_for_stmt = hash_to_agent_key[role][comp_idx].get(sh, set())
            for agent_key in keys_for_stmt:
                relevant_keys = get_relevant_keys(
                    agent_key,
                    all_keys_by_role[role][comp_idx],
                    ontology=ontology,
                    direction=direction
                )
                if relevants is None:
                    relevants = set.union(
                        *[agent_key_to_hash[role][comp_idx][rel] for rel in relevant_keys]
                    ) - {sh}
                elif not relevants:
                    break
                else:
                    role_relevant_stmt_hashes = set()
                    for rel in relevant_keys:
                        role_relevant_stmt_hashes |= (agent_key_to_hash[role][comp_idx][rel] & relevants)
                    relevants = role_relevant_stmt_hashes - {sh}
    return relevants if relevants is not None else set()


class RefinementConfirmationFilter(RefinementFilter):
    """Executes the exact statement refinement_of() method on pre-filtered candidate pairs."""

    def __init__(self, ontology, refinement_fun=None):
        super().__init__()
        self.ontology = ontology
        self.refinement_fun = refinement_fun if refinement_fun else default_refinement_fun
        self.shared_data = {}
        self.comparison_counter = 0

    def get_related(self, stmt, possibly_related=None, direction='less_specific'):
        stmts_by_hash = self.shared_data['stmts_by_hash']
        relateds = set()
        if possibly_related is None:
            possibly_related = set(stmts_by_hash.keys()) - {stmt.get_hash()}

        for possible_related_hash in possibly_related:
            more_spec_stmt, less_spec_stmt = (
                (stmt, stmts_by_hash[possible_related_hash])
                if direction == 'less_specific'
                else (stmts_by_hash[possible_related_hash], stmt)
            )
            ref = self.refinement_fun(
                more_spec_stmt,
                less_spec_stmt,
                ontology=self.ontology,
                entities_refined=False
            )
            self.comparison_counter += 1
            if ref:
                relateds.add(possible_related_hash)
        return relateds


class SplitGroupFilter(RefinementFilter):
    """Filters candidate refinements across split groups."""

    def __init__(self, split_groups):
        super().__init__()
        self.split_groups = split_groups

    def get_related(self, stmt, possibly_related=None, direction='less_specific'):
        sh = stmt.get_hash()
        group = self.split_groups.get(sh)
        possibly_related_prior = possibly_related if possibly_related is not None \
            else set(self.shared_data['stmts_by_hash'].keys())
        related = {stmt_hash for stmt_hash in possibly_related_prior
                   if self.split_groups.get(stmt_hash) != group}
        return related


def default_refinement_fun(st1, st2, ontology, entities_refined=False):
    return st1.refinement_of(st2, ontology, entities_refined=entities_refined)


def compositional_refinement(st1, st2, ontology, entities_refined=False):
    if type(st1) != type(st2):
        return False
    if isinstance(st1, Event):
        return event_compositional_refinement(st1, st2, ontology, entities_refined)
    elif isinstance(st1, Influence):
        subj_ref = event_compositional_refinement(st1.subj, st2.subj, ontology, entities_refined, ignore_polarity=True)
        if not subj_ref:
            return False
        obj_ref = event_compositional_refinement(st1.obj, st2.obj, ontology, entities_refined, ignore_polarity=True)
        if not obj_ref:
            return False
        return st1.delta_refinement_of(st2)
    return False


def event_compositional_refinement(st1, st2, ontology, entities_refined=False, ignore_polarity=False):
    gr1 = concept_matches_compositional(st1.concept)
    gr2 = concept_matches_compositional(st2.concept)
    refinement = True
    if not isinstance(gr1, tuple) and not isinstance(gr2, tuple):
        if gr1 != gr2:
            refinement = ontology.isa_or_partof('AIS', str(gr1), 'AIS', str(gr2))
    else:
        tuple1 = gr1 if isinstance(gr1, tuple) else (gr1,)
        tuple2 = gr2 if isinstance(gr2, tuple) else (gr2,)
        for entry1, entry2 in zip(tuple1, tuple2):
            if entry2 is None:
                continue
            elif entry1 is None:
                refinement = False
                break
            elif entry1 == entry2:
                continue
            else:
                if not ontology.isa_or_partof('AIS', str(entry1), 'AIS', str(entry2)):
                    refinement = False
                    break
    if not refinement:
        return False
    if ignore_polarity:
        return True
    pol_ref = (st1.delta.polarity and not st2.delta.polarity) or (st1.delta.polarity == st2.delta.polarity)
    return pol_ref


def location_refinement(st1, st2, ontology, entities_refined=False):
    if type(st1) != type(st2):
        return False
    if isinstance(st1, Event):
        return st1.refinement_of(st2, ontology, entities_refined)
    elif isinstance(st1, Influence):
        subj_ref = st1.subj.refinement_of(st2.subj, ontology, entities_refined, ignore_polarity=True)
        obj_ref = st1.obj.refinement_of(st2.obj, ontology, entities_refined, ignore_polarity=True)
        delta_ref = st1.delta_refinement_of(st2)
        return delta_ref and subj_ref and obj_ref
    return st1.refinement_of(st2, ontology, entities_refined)


location_refinement_compositional = location_refinement


def location_time_refinement(st1, st2, ontology, entities_refined=False):
    ref = location_refinement(st1, st2, ontology, entities_refined)
    if not ref:
        return False
    if not has_time(st2):
        return True
    elif not has_time(st1):
        return False
    return st1.context.time.refinement_of(st2.context.time)


def location_time_delta_refinement(st1, st2, ontology, entities_refined=False):
    loc_time_ref = location_time_refinement(st1, st2, ontology, entities_refined)
    if not loc_time_ref:
        return False
    if not st2.delta:
        return True
    elif not st1.delta:
        return False
    return st1.delta.refinement_of(st2.delta)
