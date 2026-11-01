"""daltonx.assembly.incremental
=============================
Incremental Assembler and Assembly Delta for continual statement integration.
Ported 1-to-1 from indra_world-master/indra_world/assembly/incremental_assembler.py,
adapted for daltonx statements, AISafetyOntology, and BeliefEngine.
"""

__all__ = [
    'IncrementalAssembler',
    'AssemblyDelta',
    'parse_factor_polarity_curation',
    'parse_factor_grounding_curation',
]

import copy
import logging
from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Optional, Set, Any, Tuple
import networkx

from daltonx.ontology import load_ai_safety_ontology
from daltonx.assembly.belief import (
    BeliefScorer,
    default_scorer,
    extend_refinements_graph,
)
from daltonx.assembly.refinement import (
    CompositionalRefinementFilter,
    RefinementConfirmationFilter,
    location_refinement,
)
from daltonx.assembly.matches import location_matches_compositional
from daltonx.assembly.operations import (
    AssemblyPipeline,
    add_flattened_grounding_compositional,
    standardize_names_compositional,
)

location_refinement_compositional = location_refinement

logger = logging.getLogger(__name__)


class IncrementalAssembler:
    """Assemble a set of prepared statements and allow incremental extensions.

    Parameters
    ----------
    prepared_stmts :
        A list of prepared Statements.
    refinement_filters :
        A list of RefinementFilter instances. Defaults to
        [CompositionalRefinementFilter, RefinementConfirmationFilter].
    matches_fun :
        A custom matches function for determining duplicates and calculating hashes.
        Default: location_matches_compositional.
    curations :
        A dict of user curations keyed by statement hash.
    post_processing_steps :
        Pipeline steps to run post-processing on statements.
    ontology :
        An initialized AISafetyOntology instance.
    belief_scorer :
        A BeliefScorer instance. Defaults to default_scorer.
    """

    def __init__(
        self,
        prepared_stmts: Optional[List[Any]] = None,
        refinement_filters: Optional[List[Any]] = None,
        matches_fun: Optional[Any] = None,
        curations: Optional[Dict[int, Dict[str, Any]]] = None,
        post_processing_steps: Optional[List[Dict[str, Any]]] = None,
        ontology: Optional[Any] = None,
        belief_scorer: Optional[BeliefScorer] = None,
    ):
        self.matches_fun = (
            matches_fun if matches_fun is not None else location_matches_compositional
        )
        self.stmts_by_hash: Dict[int, Any] = {}
        self.evs_by_stmt_hash: Dict[int, List[Any]] = {}
        self.refinement_edges: Set[Tuple[int, int]] = set()
        self.prepared_stmts = prepared_stmts or []
        self.known_corrects: Set[int] = set()

        if ontology is None:
            self.ontology = load_ai_safety_ontology()
            self.ontology.initialize()
        else:
            self.ontology = ontology
            if not getattr(self.ontology, '_initialized', False):
                self.ontology.initialize()

        if not refinement_filters:
            crf = CompositionalRefinementFilter(ontology=self.ontology)
            rcf = RefinementConfirmationFilter(
                ontology=self.ontology,
                refinement_fun=location_refinement_compositional,
            )
            self.refinement_filters = [crf, rcf]
        else:
            self.refinement_filters = refinement_filters

        self.curations = curations if curations else {}
        self.post_processing_steps = (
            [
                {'function': 'add_flattened_grounding_compositional'},
                {'function': 'standardize_names_compositional'},
            ]
            if post_processing_steps is None
            else post_processing_steps
        )

        self.deduplicate()
        self.apply_curations()
        self.get_refinements()
        self.refinements_graph = self.build_refinements_graph(
            self.stmts_by_hash, self.refinement_edges
        )
        self.belief_scorer = (
            belief_scorer if belief_scorer is not None else default_scorer
        )
        self.beliefs = self.get_beliefs()

    def deduplicate(self) -> None:
        """Group prepared statements by hash and combine their evidences."""
        for stmt in self.prepared_stmts:
            self.annotate_evidences(stmt)
            stmt_hash = stmt.get_hash(matches_fun=self.matches_fun)
            evs = stmt.evidence
            if stmt_hash not in self.stmts_by_hash:
                self.stmts_by_hash[stmt_hash] = stmt
            if stmt_hash not in self.evs_by_stmt_hash:
                self.evs_by_stmt_hash[stmt_hash] = []
            self.evs_by_stmt_hash[stmt_hash] += evs

    def get_refinements(self) -> None:
        """Find refinement edges (less_specific, more_specific) between unique statements."""
        for filter in self.refinement_filters:
            filter.initialize(self.stmts_by_hash)
        for sh, stmt in self.stmts_by_hash.items():
            refinements = None
            for filter in self.refinement_filters:
                refinements = filter.get_related(stmt, refinements)
            # Add less specific first and more specific second (parent -> child)
            refinement_edges = {(ref, sh) for ref in refinements}
            self.refinement_edges |= refinement_edges

    @staticmethod
    def build_refinements_graph(
        stmts_by_hash: Dict[int, Any], refinement_edges: Set[Tuple[int, int]]
    ) -> networkx.DiGraph:
        """Construct NetworkX DiGraph where edges point from general to specific."""
        g = networkx.DiGraph()
        nodes = [(sh, {'stmt': stmt}) for sh, stmt in stmts_by_hash.items()]
        g.add_nodes_from(nodes)
        g.add_edges_from(refinement_edges)
        return g

    def add_statements(self, stmts: List[Any]) -> 'AssemblyDelta':
        """Add new statements for incremental assembly.

        Parameters
        ----------
        stmts :
            A list of new Statements to incrementally integrate.

        Returns
        -------
        AssemblyDelta :
            Summary of new statements, appended evidences, new refinement edges,
            and updated beliefs.
        """
        stmts_by_hash = defaultdict(list)
        for stmt in stmts:
            self.annotate_evidences(stmt)
            sh = stmt.get_hash(matches_fun=self.matches_fun)
            stmts_by_hash[sh].append(stmt)
        stmts_by_hash = dict(stmts_by_hash)

        new_stmts = {}
        new_evidences = defaultdict(list)
        for sh, stmts_for_hash in stmts_by_hash.items():
            if sh not in self.stmts_by_hash:
                new_stmts[sh] = stmts_for_hash[0]
                self.stmts_by_hash[sh] = stmts_for_hash[0]
                self.evs_by_stmt_hash[sh] = []
            for stmt in stmts_for_hash:
                for ev in stmt.evidence:
                    new_evidences[sh].append(ev)
                    self.evs_by_stmt_hash[sh].append(ev)
        new_evidences = dict(new_evidences)

        # Run post-processing on new statements
        ap = AssemblyPipeline(steps=self.post_processing_steps)
        ap.run(list(new_stmts.values()))

        # Extend refinement filters
        for filter in self.refinement_filters:
            filter.extend(new_stmts)

        new_refinements = set()
        for sh, stmt in new_stmts.items():
            refinements = None
            for filter in self.refinement_filters:
                refinements = filter.get_related(stmt, refinements)
            # Edge points from less specific (ref) to more specific (sh)
            new_refinements |= {(ref, sh) for ref in refinements}
            extend_refinements_graph(
                self.refinements_graph,
                stmt,
                list(refinements),
                matches_fun=self.matches_fun,
            )

        self.refinement_edges |= new_refinements
        beliefs = self.get_beliefs()
        return AssemblyDelta(
            new_stmts,
            new_evidences,
            new_refinements,
            beliefs,
            matches_fun=self.matches_fun,
        )

    def get_all_supporting_evidence(self, sh: int) -> Set[Any]:
        """Return direct and indirect evidence for a statement hash."""
        all_evs = set(self.evs_by_stmt_hash.get(sh, []))
        if sh in self.refinements_graph:
            for supp in networkx.descendants(self.refinements_graph, sh):
                all_evs |= set(self.evs_by_stmt_hash.get(supp, []))
        return all_evs

    def get_beliefs(self) -> Dict[int, float]:
        """Calculate and return belief scores for all statements."""
        self.beliefs = {}
        for sh in self.evs_by_stmt_hash.keys():
            if sh in self.known_corrects:
                self.beliefs[sh] = 1.0
            else:
                evs = list(self.get_all_supporting_evidence(sh))
                self.beliefs[sh] = self.belief_scorer.score_evidence_list(evs)
        return self.beliefs

    def get_statements(self) -> List[Any]:
        """Return flat list of all assembled Statements with evidence and beliefs."""
        stmts_dict = {}
        for sh, stmt in deepcopy(self.stmts_by_hash).items():
            stmt.evidence = self.evs_by_stmt_hash.get(sh, [])
            stmt.belief = self.beliefs.get(sh)
            stmts_dict[sh] = stmt

        # Link supports / supported_by across the assembled statement instances
        for sh, stmt in stmts_dict.items():
            child_hashes = [v for u, v in self.refinement_edges if u == sh]
            parent_hashes = [u for u, v in self.refinement_edges if v == sh]
            stmt.supported_by = [stmts_dict[h] for h in child_hashes if h in stmts_dict]
            stmt.supports = [stmts_dict[h] for h in parent_hashes if h in stmts_dict]

        ap = AssemblyPipeline(steps=self.post_processing_steps)
        return ap.run(list(stmts_dict.values()))

    @staticmethod
    def annotate_evidences(stmt: Any) -> None:
        """Add agent raw text annotations to statement evidences."""
        for ev in stmt.evidence:
            raw_text = [
                None if ag is None else ag.db_refs.get('TEXT', ag.name)
                for ag in stmt.agent_list(deep_sorted=True)
            ]
            if 'agents' in ev.annotations:
                ev.annotations['agents']['raw_text'] = raw_text
            else:
                ev.annotations['agents'] = {'raw_text': raw_text}

    def apply_curations(self) -> None:
        """Apply user curations to de-duplicated statements."""
        for stmt_hash, curation in self.curations.items():
            if stmt_hash not in self.stmts_by_hash:
                continue
            stmt = self.stmts_by_hash[stmt_hash]
            update_type = curation.get('update_type')
            if update_type == 'discard_statement':
                self.stmts_by_hash.pop(stmt_hash, None)
                self.evs_by_stmt_hash.pop(stmt_hash, None)
            elif update_type == 'vet_statement':
                self.known_corrects.add(stmt_hash)
            elif update_type == 'factor_polarity':
                self.apply_polarity_curation(stmt, curation)
            elif update_type == 'reverse_relation':
                self.apply_reverse_curation(stmt, curation)
            elif update_type == 'factor_grounding':
                self.apply_grounding_curation(stmt, curation)

            if update_type in {'factor_polarity', 'reverse_relation', 'factor_grounding'}:
                new_hash = stmt.get_hash(matches_fun=self.matches_fun, refresh=True)
                if new_hash not in self.stmts_by_hash:
                    self.stmts_by_hash[new_hash] = self.stmts_by_hash.pop(stmt_hash)
                    self.evs_by_stmt_hash[new_hash] = self.evs_by_stmt_hash.pop(stmt_hash)
                else:
                    self.evs_by_stmt_hash[new_hash] += self.evs_by_stmt_hash.pop(stmt_hash)

    @staticmethod
    def apply_polarity_curation(stmt: Any, curation: Dict[str, Any]) -> None:
        role, new_pol = parse_factor_polarity_curation(curation)
        if role == 'subj' and hasattr(stmt, 'subj') and stmt.subj.delta:
            stmt.subj.delta.polarity = new_pol
        elif role == 'obj' and hasattr(stmt, 'obj') and stmt.obj.delta:
            stmt.obj.delta.polarity = new_pol

    @staticmethod
    def apply_reverse_curation(stmt: Any, curation: Dict[str, Any]) -> None:
        if hasattr(stmt, 'subj') and hasattr(stmt, 'obj'):
            stmt.subj, stmt.obj = stmt.obj, stmt.subj

    @staticmethod
    def apply_grounding_curation(stmt: Any, curation: Dict[str, Any]) -> None:
        role, txt, grounding = parse_factor_grounding_curation(curation)
        target = stmt.subj if role == 'subj' else (stmt.obj if role == 'obj' else None)
        if target and target.concept:
            target.concept.db_refs['AIS'] = grounding


class AssemblyDelta:
    """Represents incremental changes from newly added statements.

    Attributes
    ----------
    new_stmts :
        Dict mapping statement hash to new Statement object.
    new_evidences :
        Dict mapping statement hash to newly appended Evidence objects.
    new_refinements :
        Set of (parent_hash, child_hash) tuples representing new refinement edges.
    beliefs :
        Dict of updated belief scores for all statements.
    matches_fun :
        Matches key function used for hashing.
    """

    def __init__(
        self,
        new_stmts: Dict[int, Any],
        new_evidences: Dict[int, List[Any]],
        new_refinements: Set[Tuple[int, int]],
        beliefs: Dict[int, float],
        matches_fun: Optional[Any] = None,
    ):
        self.new_stmts = new_stmts
        self.new_evidences = new_evidences
        self.new_refinements = new_refinements
        self.beliefs = beliefs
        self.matches_fun = matches_fun

    def to_json(self) -> Dict[str, Any]:
        """Return JSON representation of the assembly delta."""
        new_stmts_json = {
            str(sh): stmt.to_json(matches_fun=self.matches_fun)
            for sh, stmt in self.new_stmts.items()
        }
        for stmtj in new_stmts_json.values():
            stmtj.pop('evidence', None)

        new_evs_json = {
            str(sh): [ev.to_json() for ev in evs]
            for sh, evs in self.new_evidences.items()
        }
        return {
            'new_stmts': new_stmts_json,
            'new_evidence': new_evs_json,
            'new_refinements': [list(edge) for edge in self.new_refinements],
            'beliefs': {str(k): v for k, v in self.beliefs.items()},
        }


def parse_factor_polarity_curation(cur: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
    """Parse details from a curation that changes an event's polarity."""
    bef_subj = cur.get('before', {}).get('subj', {})
    bef_obj = cur.get('before', {}).get('obj', {})
    aft_subj = cur.get('after', {}).get('subj', {})
    aft_obj = cur.get('after', {}).get('obj', {})

    if bef_subj.get('polarity') != aft_subj.get('polarity'):
        return 'subj', aft_subj.get('polarity')
    elif bef_obj.get('polarity') != aft_obj.get('polarity'):
        return 'obj', aft_obj.get('polarity')
    return None, None


def parse_factor_grounding_curation(cur: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Any]:
    """Parse details from a curation that changes a concept's grounding."""
    bef_subj = cur.get('before', {}).get('subj', {})
    bef_obj = cur.get('before', {}).get('obj', {})
    aft_subj = cur.get('after', {}).get('subj', {})
    aft_obj = cur.get('after', {}).get('obj', {})

    if bef_subj.get('concept') != aft_subj.get('concept'):
        return 'subj', aft_subj.get('factor'), aft_subj.get('concept')
    elif bef_obj.get('concept') != aft_obj.get('concept'):
        return 'obj', aft_obj.get('factor'), aft_obj.get('concept')
    return None, None, None
