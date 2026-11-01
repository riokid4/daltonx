"""daltonx.assembly.operations
===========================
Pipeline operations and preprocessing functions for statement assembly.
Ported 1-to-1 from indra_world-master/indra_world/assembly/operations.py,
adapted for daltonx statements, AISafetyOntology, and namespace 'AIS'.
"""

__all__ = [
    'AssemblyPipeline',
    'register_pipeline',
    'PIPELINE_FUNCTIONS',
    'get_expanded_events_influences',
    'remove_namespaces',
    'remove_raw_grounding',
    'check_event_context',
    'filter_context_date',
    'filter_groundings',
    'deduplicate_groundings',
    'compositional_grounding_filter_stmt',
    'compositional_grounding_filter',
    'standardize_names_compositional',
    'add_flattened_grounding_compositional',
    'validate_grounding_format',
    'make_display_name',
    'make_display_name_linear',
    'set_positive_polarities',
    'filter_out_long_words',
    'sort_compositional_groundings',
    'merge_deltas',
    'listify',
    # Re-exports from matches & refinement
    'concept_matches_compositional',
    'matches_compositional',
    'location_matches_compositional',
    'location_matches',
    'event_compositional_refinement',
    'compositional_refinement',
    'location_refinement',
    'location_refinement_compositional',
    'CompositionalRefinementFilter',
    'RefinementConfirmationFilter',
]

import copy
import logging
import itertools
import statistics
from collections import defaultdict
from typing import List, Optional, Any, Callable, Dict

from daltonx.statements import Influence, Event, Concept
from .matches import (
    concept_matches_compositional,
    matches_compositional,
    location_matches_compositional,
    location_matches,
)
from .refinement import (
    event_compositional_refinement,
    compositional_refinement,
    location_refinement,
    CompositionalRefinementFilter,
    RefinementConfirmationFilter,
)

location_refinement_compositional = location_refinement

logger = logging.getLogger(__name__)

PIPELINE_FUNCTIONS: Dict[str, Callable] = {}


def register_pipeline(func: Any) -> Any:
    """Decorator to register a function for use in AssemblyPipeline."""
    if callable(func):
        PIPELINE_FUNCTIONS[func.__name__] = func
    return func


class AssemblyPipeline:
    """An assembly pipeline that runs specified operations on statements."""

    def __init__(self, steps: Optional[List[Dict[str, Any]]] = None):
        self.steps = steps if steps is not None else []

    def run(self, statements: List[Any], **kwargs) -> List[Any]:
        """Run all steps of the pipeline on the input statements."""
        stmts = statements
        for step in self.steps:
            func = step.get('function')
            if isinstance(func, str):
                if func in PIPELINE_FUNCTIONS:
                    func = PIPELINE_FUNCTIONS[func]
                elif func in globals():
                    func = globals()[func]
                else:
                    logger.warning(f"Pipeline function '{func}' not found; skipping.")
                    continue
            args = step.get('args', [])
            step_kwargs = copy.copy(step.get('kwargs', {}))
            step_kwargs.update(kwargs)
            res = func(stmts, *args, **step_kwargs)
            if res is not None:
                stmts = res
        return stmts

    def append(self, func: Any, *args, **kwargs):
        """Append a step to the pipeline."""
        func_name = func.__name__ if callable(func) else str(func)
        if callable(func):
            register_pipeline(func)
        self.steps.append({'function': func_name, 'args': list(args), 'kwargs': kwargs})


@register_pipeline
def get_expanded_events_influences(stmts: List[Any]) -> List[Any]:
    """Return a list of standalone Events and parent Influence statements."""
    events_influences = []
    for stmt_orig in stmts:
        stmt = copy.deepcopy(stmt_orig)
        if isinstance(stmt, Influence):
            for member in [stmt.subj, stmt.obj]:
                member.evidence = copy.deepcopy(stmt.evidence)
                for ev in member.evidence:
                    ev.context = None
                events_influences.append(member)
            events_influences.append(stmt_orig)
        elif isinstance(stmt, Event):
            events_influences.append(stmt)
        else:
            events_influences.append(stmt_orig)
    return events_influences


@register_pipeline
def remove_namespaces(stmts: List[Any], namespaces: List[str]) -> List[Any]:
    """Remove unnecessary namespaces from Concept grounding."""
    for stmt in stmts:
        for agent in stmt.agent_list():
            if agent and hasattr(agent, 'db_refs'):
                for ns in namespaces:
                    agent.db_refs.pop(ns, None)
    return stmts


@register_pipeline
def remove_raw_grounding(stmts: List[Any]) -> List[Any]:
    """Remove raw_grounding annotations to decrease serialized size."""
    for stmt in stmts:
        for ev in stmt.evidence:
            if ev.annotations and 'agents' in ev.annotations:
                ev.annotations['agents'].pop('raw_grounding', None)
    return stmts


@register_pipeline
def check_event_context(events: List[Event]) -> None:
    """Validate event context consistency with evidence context."""
    for event in events:
        if not event.context and event.evidence and event.evidence[0].context:
            raise AssertionError(f"Event context mismatch: {event}, {event.evidence}")


@register_pipeline
def filter_context_date(stmts: List[Any], from_date: Optional[str] = None, to_date: Optional[str] = None) -> List[Any]:
    """Filter statements by time context dates."""
    if not from_date and not to_date:
        return stmts
    new_stmts = []
    for stmt in stmts:
        events = [stmt.subj, stmt.obj] if isinstance(stmt, Influence) else ([stmt] if isinstance(stmt, Event) else [])
        for event in events:
            if event.context and event.context.time:
                t = event.context.time
                if from_date and getattr(t, 'start', None) and (t.start < from_date):
                    event.context.time = None
                if to_date and getattr(t, 'end', None) and (t.end > to_date):
                    event.context.time = None
        new_stmts.append(stmt)
    return new_stmts


@register_pipeline
def filter_groundings(stmts: List[Any], groundings_to_exclude: Optional[List[str]] = None, namespace: str = 'AIS') -> List[Any]:
    """Filter out statements that ground to excluded ontology IDs."""
    if not groundings_to_exclude:
        return stmts
    excl = set(groundings_to_exclude)
    new_stmts = []
    for stmt in stmts:
        exclude = False
        for agent in stmt.agent_list():
            if not agent:
                continue
            gr = agent.db_refs.get(namespace)
            if isinstance(gr, str) and gr in excl:
                exclude = True
                break
            elif isinstance(gr, list):
                for entry in gr:
                    if isinstance(entry, (tuple, list)) and entry[0] in excl:
                        exclude = True
                        break
        if not exclude:
            new_stmts.append(stmt)
    return new_stmts


def deduplicate_groundings(groundings: List[Any]) -> List[Any]:
    """Deduplicate grounding list while preserving order."""
    seen = set()
    result = []
    for gr in groundings:
        key = str(gr)
        if key not in seen:
            seen.add(key)
            result.append(gr)
    return result


def compositional_grounding_filter_stmt(
    stmt: Any,
    score_threshold: float,
    groundings_to_exclude: Optional[List[str]] = None,
    remove_self_loops: bool = False,
    namespace: str = 'AIS',
) -> Optional[Any]:
    """Filter and sanitize compositional groundings on a single statement."""
    groundings_to_exclude = set(groundings_to_exclude) if groundings_to_exclude else set()
    stmt = copy.deepcopy(stmt)
    for concept in stmt.agent_list():
        if concept is not None and namespace in concept.db_refs:
            grs = concept.db_refs[namespace]
            if isinstance(grs, str):
                if grs in groundings_to_exclude:
                    return None
                continue
            new_groundings = []
            for gr in grs:
                if isinstance(gr, (list, tuple)):
                    entry_list = list(gr)
                    for idx, entry in enumerate(entry_list):
                        if entry is not None:
                            val = entry[0] if isinstance(entry, (tuple, list)) else entry
                            score = entry[1] if isinstance(entry, (tuple, list)) and len(entry) > 1 else 1.0
                            if val in groundings_to_exclude or score < score_threshold:
                                entry_list[idx] = None
                    if not all(e is None for e in entry_list):
                        new_groundings.append(tuple(entry_list))
            if not new_groundings:
                return None
            concept.db_refs[namespace] = deduplicate_groundings(new_groundings)
        else:
            return None
    return stmt


@register_pipeline
def compositional_grounding_filter(
    stmts: List[Any],
    score_threshold: float,
    groundings_to_exclude: Optional[List[str]] = None,
    remove_self_loops: bool = False,
    namespace: str = 'AIS',
) -> List[Any]:
    """Filter statements by compositional grounding score and exclusions."""
    stmts_out = []
    for stmt in stmts:
        filtered = compositional_grounding_filter_stmt(
            stmt, score_threshold, groundings_to_exclude, remove_self_loops, namespace
        )
        if filtered:
            stmts_out.append(filtered)
    return stmts_out


def make_display_name(comp_grounding: Any) -> str:
    """Return display name with 'of' linkers from grounding."""
    if isinstance(comp_grounding, str):
        return comp_grounding.split('/')[-1].replace('_', ' ')
    if isinstance(comp_grounding, (tuple, list)):
        if len(comp_grounding) == 2 and isinstance(comp_grounding[0], str) and isinstance(comp_grounding[1], (int, float)):
            return comp_grounding[0].split('/')[-1].replace('_', ' ')
        entries = []
        for entry in comp_grounding:
            if not entry:
                continue
            val = entry[0] if isinstance(entry, (tuple, list)) else entry
            if val:
                entries.append(str(val).split('/')[-1].replace('_', ' '))
        entries_reversed = entries[::-1]
        return ' of '.join(entries_reversed) if entries_reversed else ''
    return str(comp_grounding)


def make_display_name_linear(comp_grounding: Any) -> str:
    """Return display name with space linkers from grounding."""
    if isinstance(comp_grounding, str):
        return comp_grounding.split('/')[-1].replace('_', ' ')
    if isinstance(comp_grounding, (tuple, list)):
        if len(comp_grounding) == 2 and isinstance(comp_grounding[0], str) and isinstance(comp_grounding[1], (int, float)):
            return comp_grounding[0].split('/')[-1].replace('_', ' ')
        entries = []
        for entry in comp_grounding:
            if not entry:
                continue
            val = entry[0] if isinstance(entry, (tuple, list)) else entry
            if val:
                entries.append(str(val).split('/')[-1].replace('_', ' '))
        return ' '.join(entries) if entries else ''
    return str(comp_grounding)


@register_pipeline
def standardize_names_compositional(stmts: List[Any], namespace: str = 'AIS') -> List[Any]:
    """Update Concept names to display names based on their primary grounding."""
    for stmt in stmts:
        for concept in stmt.agent_list():
            if not concept:
                continue
            gr = concept.db_refs.get(namespace) or concept.db_refs.get('WM')
            if gr:
                first = gr[0] if isinstance(gr, list) and gr else gr
                disp_name = make_display_name(first)
                if disp_name:
                    concept.name = disp_name
    return stmts


@register_pipeline
def add_flattened_grounding_compositional(stmts: List[Any], namespace: str = 'AIS') -> List[Any]:
    """Add flattened grounding annotations to db_refs for downstream consumers."""
    for stmt in stmts:
        for concept in stmt.agent_list():
            if not concept:
                continue
            grs = concept.db_refs.get(namespace) or concept.db_refs.get('WM')
            if grs:
                flat_list = []
                gr_entries = grs if isinstance(grs, list) else [grs]
                for comp_grounding in gr_entries:
                    disp_name = make_display_name(comp_grounding)
                    if (
                        isinstance(comp_grounding, (list, tuple))
                        and len(comp_grounding) == 2
                        and isinstance(comp_grounding[0], str)
                        and isinstance(comp_grounding[1], (int, float))
                    ):
                        flat_id = comp_grounding[0]
                        score = float(comp_grounding[1])
                    elif isinstance(comp_grounding, (list, tuple)):
                        theme = comp_grounding[0][0] if comp_grounding[0] and isinstance(comp_grounding[0], (list, tuple)) else (str(comp_grounding[0]) if comp_grounding[0] else '')
                        others = [
                            entry[0].split('/')[-1]
                            for entry in comp_grounding[1:]
                            if entry and isinstance(entry, (list, tuple))
                        ]
                        flat_id = '_'.join([theme] + others)
                        scores = [
                            entry[1]
                            for entry in comp_grounding
                            if entry and isinstance(entry, (list, tuple)) and len(entry) > 1 and isinstance(entry[1], (int, float))
                        ]
                        score = statistics.mean(scores) if scores else 1.0
                    else:
                        flat_id = str(comp_grounding)
                        score = 1.0
                    flat_list.append({'grounding': flat_id, 'name': disp_name, 'score': score})
                concept.db_refs[f'{namespace}_FLAT'] = flat_list
    return stmts


@register_pipeline
def validate_grounding_format(stmts: List[Any], namespace: str = 'AIS') -> List[Any]:
    """Validate grounding list structure on all statements."""
    for stmt in stmts:
        for concept in stmt.agent_list():
            if not concept:
                continue
            grs = concept.db_refs.get(namespace)
            if grs is not None and isinstance(grs, list):
                for gr in grs:
                    if isinstance(gr, tuple):
                        assert len(gr) >= 1
    return stmts


@register_pipeline
def set_positive_polarities(stmts: List[Any]) -> List[Any]:
    """Set default positive polarity (+1) for any event delta with None polarity."""
    for stmt in stmts:
        if isinstance(stmt, Influence):
            for event in [stmt.subj, stmt.obj]:
                if event and event.delta and event.delta.polarity is None:
                    event.delta.polarity = 1
    return stmts


@register_pipeline
def filter_out_long_words(stmts: List[Any], k: int = 10) -> List[Any]:
    """Filter out statements where any concept raw text exceeds k words."""
    def text_too_long(concept: Any, limit: int) -> bool:
        if not concept:
            return False
        txt = concept.db_refs.get('TEXT', concept.name)
        return len(txt.split()) > limit if txt else False

    return [
        stmt for stmt in stmts
        if not any(text_too_long(c, k) for c in stmt.agent_list())
    ]


@register_pipeline
def sort_compositional_groundings(statements: List[Any], namespace: str = 'AIS') -> List[Any]:
    """Sort compositional groundings by theme score."""
    for stmt in statements:
        for concept in stmt.agent_list():
            if not concept:
                continue
            grs = concept.db_refs.get(namespace)
            if isinstance(grs, list) and grs and isinstance(grs[0], (tuple, list)):
                concept.db_refs[namespace] = sorted(
                    grs,
                    key=lambda x: (x[0][1] if x[0] and len(x[0]) > 1 else 0.0),
                    reverse=True
                )
    return statements


@register_pipeline
def merge_deltas(stmts_in: List[Any]) -> List[Any]:
    """Gather and merge Influence delta polarities across all evidences."""
    stmts_out = []
    for stmt in stmts_in:
        if not isinstance(stmt, Influence):
            stmts_out.append(stmt)
            continue
        deltas: Dict[Any, List[Any]] = {}
        for role in ('subj', 'obj'):
            for info in ('polarity', 'adjectives'):
                key = (role, info)
                deltas[key] = []
                for ev in stmt.evidence:
                    entry = ev.annotations.get(f'{role}_{info}')
                    deltas[key].append(entry if entry else None)

        polarity_pairs = list(zip(deltas[('subj', 'polarity')], deltas[('obj', 'polarity')]))
        both_pols = [p for p in polarity_pairs if p[0] is not None and p[1] is not None]
        if both_pols:
            subj_pol, obj_pol = max(set(both_pols), key=both_pols.count)
            stmt.subj.delta.polarity = subj_pol
            stmt.obj.delta.polarity = obj_pol
        else:
            one_pol = [p for p in polarity_pairs if p[0] is not None or p[1] is not None]
            if one_pol:
                subj_pol, obj_pol = max(set(one_pol), key=one_pol.count)
                stmt.subj.delta.polarity = subj_pol
                stmt.obj.delta.polarity = obj_pol

        stmts_out.append(stmt)
    return stmts_out


@register_pipeline
def listify(obj: Any) -> List[Any]:
    """Wrap a single object in a list if not already a list."""
    return [obj] if not isinstance(obj, list) else obj
