"""daltonx.assembly.belief
======================
Belief Engine for calculating Statement prior probabilities,
hierarchical evidence propagation, and belief scores.

Ported from indra.belief and indra_world.belief, adapted for AI Safety
domain knowledge sources and standalone execution.
"""

import copy
import logging
from typing import List, Optional, Dict, Callable, Tuple, Sequence, Iterable, Any
import numpy
import networkx

from daltonx.statements import Evidence, Statement

logger = logging.getLogger(__name__)

# Default prior error probabilities for AI Safety extraction and reading sources:
# 'rand': probability of random error/noise
# 'syst': probability of systematic error
DEFAULT_BELIEF_PROBS: Dict[str, Dict[str, float]] = {
    'syst': {
        # AI Safety specific sources & readers
        'paperqa': 0.05,
        'llm_reader': 0.08,
        'arxiv_extractor': 0.06,
        'manual_curation': 0.01,
        'expert_assertion': 0.0,
        # General extraction sources
        'eidos': 0.05,
        'sofia': 0.05,
        'hume': 0.05,
        'cwms': 0.05,
        'assertion': 0.0,
        'default': 0.05,
    },
    'rand': {
        # AI Safety specific sources & readers
        'paperqa': 0.15,
        'llm_reader': 0.20,
        'arxiv_extractor': 0.18,
        'manual_curation': 0.01,
        'expert_assertion': 0.0,
        # General extraction sources
        'eidos': 0.30,
        'sofia': 0.30,
        'hume': 0.30,
        'cwms': 0.30,
        'assertion': 0.0,
        'default': 0.30,
    }
}


def load_default_probs() -> Dict[str, Dict[str, float]]:
    """Return a deep copy of the default prior belief probabilities."""
    return copy.deepcopy(DEFAULT_BELIEF_PROBS)


class BeliefScorer(object):
    """Base class for a belief engine scorer, which computes the
    prior probability of a statement given its type and evidence.
    """

    def score_statements(
        self,
        statements: Sequence[Statement],
        extra_evidence: Optional[List[List[Evidence]]] = None,
    ) -> Sequence[float]:
        """Computes belief probabilities for a list of Statements."""
        raise NotImplementedError('Need to subclass BeliefScorer and implement methods.')

    def score_statement(
        self,
        statement: Statement,
        extra_evidence: Optional[List[Evidence]] = None,
    ) -> float:
        """Score a single statement by passing arguments to `score_statements`."""
        extra_evidence_wrap = None if extra_evidence is None else [extra_evidence]
        return self.score_statements([statement], extra_evidence_wrap)[0]

    def check_prior_probs(
        self,
        statements: Sequence[Statement],
    ) -> None:
        """Verify that the scorer has prior error rates for all sources."""
        raise NotImplementedError('Need to subclass BeliefScorer and implement methods.')


class SimpleScorer(BeliefScorer):
    """Computes the prior probability of a statement given its evidence.

    Parameters
    ----------
    prior_probs :
        A dictionary of prior probabilities used to override/extend default ones.
        Structure: {'rand': {source: rate}, 'syst': {source: rate}}
    subtype_probs :
        A dictionary of rule-specific error rates per source:
        {source: {rule_name: error_rate}}
    """

    def __init__(
        self,
        prior_probs: Optional[Dict[str, Dict[str, float]]] = None,
        subtype_probs: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.prior_probs = load_default_probs()
        self.subtype_probs: Optional[Dict[str, Dict[str, float]]] = {}
        self.update_probs(prior_probs, subtype_probs)

    def update_probs(
        self,
        prior_probs: Optional[Dict[str, Dict[str, float]]] = None,
        subtype_probs: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """Update Scorer's prior probabilities with the given dictionaries."""
        if prior_probs:
            for key in ('rand', 'syst'):
                if key in prior_probs:
                    self.prior_probs[key].update(prior_probs[key])
        if subtype_probs:
            self.subtype_probs = copy.deepcopy(subtype_probs)

    def score_evidence_list(
        self,
        evidences: List[Evidence],
    ) -> float:
        """Return belief score given a list of supporting evidences.

        Evaluates positive evidences and negative (negated) evidences
        separately and combines them via: `pp * (1 - np)`.
        """
        def _score(ev_list: List[Evidence]) -> float:
            if not ev_list:
                return 0.0
            sources = [ev.source_api for ev in ev_list]
            uniq_sources = numpy.unique(sources)

            # Systematic error factors per unique source
            syst_factors = {
                s: self.prior_probs['syst'].get(s, self.prior_probs['syst'].get('default', 0.05))
                for s in uniq_sources
            }

            # Random error factors per evidence piece
            rand_factors: Dict[str, List[float]] = {k: [] for k in uniq_sources}
            for ev in ev_list:
                rand_factors[ev.source_api].append(
                    evidence_random_noise_prior(
                        ev,
                        self.prior_probs['rand'],
                        self.subtype_probs
                    )
                )

            # Probability of incorrectness across sources
            neg_prob_prior = 1.0
            for s in uniq_sources:
                neg_prob_prior *= (syst_factors[s] + float(numpy.prod(rand_factors[s])))

            prob_prior = 1.0 - neg_prob_prior
            return float(prob_prior)

        # Split evidence into positive and negated
        pos_evidence = [ev for ev in evidences if not ev.epistemics.get('negated')]
        neg_evidence = [ev for ev in evidences if ev.epistemics.get('negated')]

        pp = _score(pos_evidence)
        np = _score(neg_evidence)

        # Formula: pp * (1 - np)
        score = pp * (1.0 - np)
        return float(score)

    def score_statements(
        self,
        statements: Sequence[Statement],
        extra_evidence: Optional[List[List[Evidence]]] = None,
    ) -> List[float]:
        """Computes belief probabilities for a list of Statements."""
        check_extra_evidence(extra_evidence, len(statements))
        beliefs = []
        for ix, stmt in enumerate(statements):
            all_evidence = get_stmt_evidence(stmt, ix, extra_evidence)
            beliefs.append(self.score_evidence_list(all_evidence))
        return beliefs

    def check_prior_probs(
        self,
        statements: Sequence[Statement],
    ) -> None:
        """Make sure all sources have entries in prior_probs or default exists."""
        sources = set()
        for stmt in statements:
            sources |= set([ev.source_api for ev in stmt.evidence])
        return self._check_sources(sources)

    def _check_sources(
        self,
        sources: Iterable[str],
    ) -> None:
        """Verify each source exists or default fallback is configured."""
        for err_type in ('rand', 'syst'):
            for source in sources:
                if source not in self.prior_probs[err_type]:
                    if 'default' not in self.prior_probs[err_type]:
                        msg = ('BeliefEngine missing probability parameter '
                               f'for source: {source}')
                        raise Exception(msg)


class BayesianScorer(SimpleScorer):
    """Belief scorer assuming Beta priors updated from empirical counts."""

    def __init__(
        self,
        prior_counts: Optional[Dict[str, List[int]]] = None,
        subtype_counts: Optional[Dict[str, Dict[str, List[int]]]] = None,
    ):
        self.prior_probs = load_default_probs()
        self.subtype_probs = {}
        self.prior_counts = copy.deepcopy(prior_counts) if prior_counts else {}
        self.subtype_counts = copy.deepcopy(subtype_counts) if subtype_counts else {}
        self.update_probs()

    def update_probs(self):
        """Update probability estimates based on counts."""
        syst_error = 0.05
        prior_probs: Dict[str, Dict[str, float]] = {'syst': {}, 'rand': {}}
        for source, (p, n) in self.prior_counts.items():
            if n + p == 0:
                continue
            prior_probs['syst'][source] = syst_error
            prior_probs['rand'][source] = (
                1.0 - min((float(p) / (n + p), 1.0 - syst_error)) - syst_error
            )
        subtype_probs: Dict[str, Dict[str, float]] = {}
        for source, entry in self.subtype_counts.items():
            for rule, (p, n) in entry.items():
                if n + p == 0:
                    continue
                if source not in subtype_probs:
                    subtype_probs[source] = {}
                subtype_probs[source][rule] = (
                    1.0 - min((float(p) / (n + p), 1.0 - syst_error)) - syst_error
                )
        super(BayesianScorer, self).update_probs(prior_probs, subtype_probs)

    def update_counts(
        self,
        prior_counts: Dict[str, List[int]],
        subtype_counts: Optional[Dict[str, Dict[str, List[int]]]] = None,
    ) -> None:
        """Update internal counts and recompute probabilities."""
        for source, (pos, neg) in prior_counts.items():
            if source not in self.prior_counts:
                self.prior_counts[source] = [0, 0]
            self.prior_counts[source][0] += pos
            self.prior_counts[source][1] += neg
        if subtype_counts:
            for source, subtype_dict in subtype_counts.items():
                if source not in self.subtype_counts:
                    self.subtype_counts[source] = {}
                for subtype, (pos, neg) in subtype_dict.items():
                    if subtype not in self.subtype_counts[source]:
                        self.subtype_counts[source][subtype] = [0, 0]
                    self.subtype_counts[source][subtype][0] += pos
                    self.subtype_counts[source][subtype][1] += neg
        self.update_probs()


default_scorer = SimpleScorer()


def get_ai_safety_scorer() -> SimpleScorer:
    """Return a SimpleScorer tailored for AI Safety reading pipelines."""
    return SimpleScorer()


def get_eidos_scorer() -> SimpleScorer:
    """Alias for backwards compatibility with indra_world assembly pipelines."""
    return SimpleScorer()


class BeliefEngine(object):
    """Assigns beliefs to Statements based on evidence and hierarchical support.

    Parameters
    ----------
    scorer :
        A BeliefScorer instance. Defaults to default_scorer.
    matches_fun :
        Function returning the matches key of a statement.
    refinements_graph :
        A DiGraph whose nodes are statement hashes and edges point from
        less specific (general) to more specific (refined) statements.
    """

    def __init__(
        self,
        scorer: Optional[BeliefScorer] = None,
        matches_fun: Optional[Callable[[Statement], str]] = None,
        refinements_graph: Optional[networkx.DiGraph] = None,
    ):
        if scorer is None:
            scorer = default_scorer
        assert isinstance(scorer, BeliefScorer)
        self.scorer = scorer
        self.matches_fun = (
            matches_fun if matches_fun is not None else (lambda stmt: stmt.matches_key())
        )
        self.refinements_graph = refinements_graph

    def set_prior_probs(
        self,
        statements: Sequence[Statement],
    ) -> None:
        """Sets prior belief probabilities for a list of Statements."""
        self.scorer.check_prior_probs(statements)
        prior_probs = self.scorer.score_statements(statements)
        assert len(prior_probs) == len(statements), (
            "prior_probs and statements should have the same length"
        )
        for ix, st in enumerate(statements):
            st.belief = prior_probs[ix]

    def set_hierarchy_probs(
        self,
        statements: Sequence[Statement],
    ) -> None:
        """Sets hierarchical belief probabilities for Statements.

        Evidence from more specific statements (in the `supports` field)
        is accumulated into more general statements, boosting the parent belief.
        """
        beliefs = self.get_hierarchy_probs(statements)
        for stmt in statements:
            sh = stmt.get_hash(matches_fun=self.matches_fun)
            stmt.belief = beliefs[sh]

    def get_hierarchy_probs(
        self,
        statements: Sequence[Statement],
    ) -> Dict[int, float]:
        """Compute hierarchical belief probabilities for Statements."""
        if self.refinements_graph is None:
            self.refinements_graph = build_refinements_graph(
                statements, matches_fun=self.matches_fun
            )
        all_extra_evs = get_ev_for_stmts_from_supports(
            statements, self.refinements_graph, matches_fun=self.matches_fun
        )
        return self._hierarchy_probs_from_evidences(statements, all_extra_evs)

    def get_hierarchy_probs_from_hashes(
        self,
        statements: Sequence[Statement],
        refiners_list: List[List[int]],
    ) -> Dict[int, float]:
        """Compute belief of statements given explicitly specified refiners."""
        if self.refinements_graph is None:
            raise ValueError("refinements_graph not initialized.")
        all_extra_evs = get_ev_for_stmts_from_hashes(
            statements, refiners_list, self.refinements_graph
        )
        return self._hierarchy_probs_from_evidences(statements, all_extra_evs)

    def _hierarchy_probs_from_evidences(
        self,
        statements: Sequence[Statement],
        extra_evidence: List[List[Evidence]],
    ) -> Dict[int, float]:
        beliefs = self.scorer.score_statements(statements, extra_evidence)
        hashes = [s.get_hash(matches_fun=self.matches_fun) for s in statements]
        return dict(zip(hashes, beliefs))

    def set_linked_probs(
        self,
        linked_statements: Sequence[Any],
    ) -> None:
        """Sets the belief probabilities for linked Statements."""
        for st in linked_statements:
            source_probs = [
                s.belief for s in st.source_stmts if getattr(s, 'belief', None) is not None
            ]
            if source_probs:
                st.inferred_stmt.belief = float(numpy.prod(source_probs))


def build_refinements_graph(
    statements: Sequence[Statement],
    matches_fun: Optional[Callable[[Statement], str]] = None,
) -> networkx.DiGraph:
    """Return a DiGraph based on statement matches hashes and refinements.

    Edges point from less detailed (general) to more detailed (specific) statements.
    In Preassembly, a general statement lists its specific refinements in `supports`.
    """
    g = networkx.DiGraph()
    for st1 in statements:
        sh1 = st1.get_hash(matches_fun=matches_fun)
        g.add_node(sh1, stmt=st1)
        for st2 in st1.supports:
            sh2 = st2.get_hash(matches_fun=matches_fun)
            g.add_node(sh2, stmt=st2)
            g.add_edge(sh1, sh2)
    return g


def extend_refinements_graph(
    g: networkx.DiGraph,
    stmt: Statement,
    less_specifics: List[int],
    matches_fun: Optional[Callable[[Statement], str]] = None,
) -> networkx.DiGraph:
    """Extend refinements graph with a new statement and its general ancestors.

    Parameters
    ----------
    g :
        Refinements graph.
    stmt :
        More specific statement to add.
    less_specifics :
        List of hashes of statements that are less specific (general) versions.
    matches_fun :
        Function to compute statement matches key.
    """
    sh = stmt.get_hash(matches_fun=matches_fun)
    g.add_node(sh, stmt=stmt)
    for less_spec in less_specifics:
        g.add_edge(less_spec, sh)
    return g


def get_ev_for_stmts_from_supports(
    statements: Sequence[Statement],
    refinements_graph: Optional[networkx.DiGraph] = None,
    matches_fun: Optional[Callable[[Statement], str]] = None,
) -> List[List[Evidence]]:
    """Collect evidence from the more specific statements for each statement."""
    if refinements_graph is None:
        refinements_graph = build_refinements_graph(statements, matches_fun=matches_fun)
    assert_no_cycle(refinements_graph)

    refiners_list = []
    for stmt in statements:
        stmt_hash = stmt.get_hash(matches_fun=matches_fun)
        # In the refinements graph, edges point from general to specific.
        # Descendants are all transitive specific refinements of this statement.
        if stmt_hash in refinements_graph:
            refiners = list(networkx.descendants(refinements_graph, stmt_hash))
        else:
            refiners = []
        refiners_list.append(refiners)

    return get_ev_for_stmts_from_hashes(statements, refiners_list, refinements_graph)


def get_ev_for_stmts_from_hashes(
    statements: Sequence[Statement],
    refiners_list: List[List[int]],
    refinements_graph: networkx.DiGraph,
) -> List[List[Evidence]]:
    """Collect non-negated evidence from refiners given as hashes."""
    all_extra_evs = []
    for stmt, refiners in zip(statements, refiners_list):
        extra_ev_for_stmt = list(set(
            ev
            for supp in refiners
            if supp in refinements_graph.nodes and 'stmt' in refinements_graph.nodes[supp]
            for ev in refinements_graph.nodes[supp]['stmt'].evidence
            if not ev.epistemics.get('negated')
        ))
        all_extra_evs.append(extra_ev_for_stmt)
    return all_extra_evs


def sample_statements(
    stmts: Sequence[Statement],
    seed: Optional[int] = None,
) -> List[Statement]:
    """Sample statements independently according to their belief scores."""
    if seed is not None:
        numpy.random.seed(seed)
    r = numpy.random.rand(len(stmts))
    return [stmt for i, stmt in enumerate(stmts) if r[i] < (stmt.belief or 0.0)]


def evidence_random_noise_prior(
    evidence: Evidence,
    type_probs: Dict[str, float],
    subtype_probs: Optional[Dict[str, Dict[str, float]]],
) -> float:
    """Return the random-noise prior probability for an evidence piece."""
    stype, subtype = tag_evidence_subtype(evidence)
    if subtype_probs is not None and stype in subtype_probs:
        if subtype is not None and subtype in subtype_probs[stype]:
            return subtype_probs[stype][subtype]
    if stype in type_probs:
        return type_probs[stype]
    return type_probs.get('default', 0.30)


def tag_evidence_subtype(
    evidence: Evidence,
) -> Tuple[str, Optional[str]]:
    """Return (source_api, subtype) extracted from evidence annotations."""
    source_api = evidence.source_api
    annotations = evidence.annotations
    subtype = annotations.get('rule') or annotations.get('found_by') or annotations.get('source_sub_id')
    return (source_api, subtype)


def assert_no_cycle(g: networkx.DiGraph) -> None:
    """Raise AssertionError if the graph contains cycles."""
    try:
        cyc = networkx.algorithms.cycles.find_cycle(g)
    except networkx.exception.NetworkXNoCycle:
        return
    raise AssertionError(f"Cycle found in refinements graph: {cyc}")


def get_ranked_stmts(g: networkx.DiGraph) -> List[Statement]:
    """Return statements ordered by reverse topological sort (most specific first)."""
    node_ranks = reversed(list(networkx.algorithms.dag.topological_sort(g)))
    return [g.nodes[n]['stmt'] for n in node_ranks]


def check_extra_evidence(
    extra_evidence: Optional[List[List[Evidence]]],
    num_stmts: int,
) -> None:
    """Validate format and length of extra_evidence list."""
    if extra_evidence is not None:
        if len(extra_evidence) != num_stmts:
            raise ValueError("extra_evidence must be a list of the same length as stmts.")
        for entry in extra_evidence:
            if not isinstance(entry, list):
                raise ValueError("extra_evidence must be a list of lists.")
            if entry and not isinstance(entry[0], Evidence):
                raise ValueError("extra_evidence list entries must contain Evidence objects.")


def get_stmt_evidence(
    stmt: Statement,
    ix: int,
    extra_evidence: Optional[List[List[Evidence]]],
) -> List[Evidence]:
    """Combine a statement's own evidence with any extra evidence provided."""
    stmt_ev = set(stmt.evidence)
    if extra_evidence is not None:
        stmt_ev.update(extra_evidence[ix])
    return list(stmt_ev)
