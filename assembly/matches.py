"""daltonx.assembly.matches
=========================
Custom statement matching functions for duplicate detection.
Ported 1-to-1 from indra_world-master/indra_world/assembly/matches.py,
adapted to AISafetyContext and namespace 'AIS'.
"""

__all__ = [
    'has_location',
    'has_time',
    'get_location_from_object',
    'get_location',
    'get_time',
    'has_delta',
    'get_delta',
    'concept_matches_compositional',
    'matches_compositional',
    'location_matches_compositional',
    'location_matches',
    'event_location_time_matches',
    'event_location_time_delta_matches',
    'location_time_matches',
    'location_time_delta_matches',
]

from daltonx.statements import Influence, Event, QuantitativeState, QualitativeDelta


def has_location(stmt):
    """Return True if a Statement has grounded location context."""
    if not stmt.context:
        return False
    # Check geo_location or environment or model
    geo = getattr(stmt.context, 'geo_location', None)
    if geo and (geo.db_refs.get('GEOID') or geo.name):
        return True
    env = getattr(stmt.context, 'environment', None)
    if env and (env.db_refs.get('ENV') or env.name):
        return True
    return False


def has_time(stmt):
    """Return True if a Statement has time context."""
    if not stmt.context or not stmt.context.time:
        return False
    return True


def get_location_from_object(loc_obj):
    """Return location identifier or name from a RefContext location object."""
    if not loc_obj:
        return None
    if loc_obj.db_refs.get('GEOID'):
        return loc_obj.db_refs['GEOID']
    elif loc_obj.db_refs.get('ENV'):
        return loc_obj.db_refs['ENV']
    elif loc_obj.name:
        return loc_obj.name
    return None


def get_location(stmt):
    """Return the grounded location context associated with a Statement."""
    if not has_location(stmt):
        return None
    if getattr(stmt.context, 'geo_location', None):
        return get_location_from_object(stmt.context.geo_location)
    elif getattr(stmt.context, 'environment', None):
        return get_location_from_object(stmt.context.environment)
    return None


def get_time(stmt):
    """Return the time context associated with a Statement."""
    if not has_time(stmt):
        return None
    return stmt.context.time


def has_delta(stmt):
    return stmt.delta is not None


def get_delta(stmt):
    delta = stmt.delta
    if isinstance(delta, QualitativeDelta):
        return delta.polarity
    elif isinstance(delta, QuantitativeState):
        return (delta.entity, delta.value, delta.unit, delta.polarity)
    return None


def concept_matches_compositional(concept):
    """Return a tuple key or string id for grounding matching."""
    ais = concept.db_refs.get('AIS')
    if not ais:
        return concept.name
    if isinstance(ais, str):
        return ais
    if isinstance(ais, (list, tuple)):
        if len(ais) > 0:
            first_entry = ais[0]
            # Case 1: Flat scored grounding list: [(id_str, score), ...]
            if (
                isinstance(first_entry, (list, tuple))
                and len(first_entry) == 2
                and not isinstance(first_entry[0], (list, tuple))
            ):
                top_entry = sorted(
                    ais,
                    key=lambda x: x[1] if len(x) > 1 and isinstance(x[1], (int, float)) else 0,
                    reverse=True,
                )[0][0]
                return top_entry
            # Case 2: Compositional 4-tuple grounding: [((theme, score), ...)]
            elif isinstance(first_entry, (list, tuple)):
                return tuple(
                    entry[0]
                    if isinstance(entry, (list, tuple)) and len(entry) > 0
                    else (entry if isinstance(entry, str) else None)
                    for entry in first_entry
                )
    return str(ais)


def matches_compositional(stmt):
    """Return a matches_key using compositional grounding for duplicate grouping."""
    if isinstance(stmt, Influence):
        key = (
            stmt.__class__.__name__,
            concept_matches_compositional(stmt.subj.concept),
            concept_matches_compositional(stmt.obj.concept),
            stmt.polarity_count(),
            stmt.overall_polarity()
        )
    elif isinstance(stmt, Event):
        pol = stmt.delta.polarity if stmt.delta else None
        key = (
            stmt.__class__.__name__,
            concept_matches_compositional(stmt.concept),
            pol
        )
    else:
        key = stmt.matches_key()
    return str(key)


def location_matches_compositional(stmt):
    """Return a matches_key which takes location into account with compositional grounding."""
    if isinstance(stmt, Event):
        context_key = get_location(stmt)
        matches_key = str((matches_compositional(stmt), context_key))
    elif isinstance(stmt, Influence):
        subj_context_key = get_location(stmt.subj)
        obj_context_key = get_location(stmt.obj)
        matches_key = str((matches_compositional(stmt), subj_context_key, obj_context_key))
    else:
        matches_key = matches_compositional(stmt)
    return matches_key


def location_matches(stmt):
    """Return a matches_key which takes location context into account."""
    if isinstance(stmt, Event):
        context_key = get_location(stmt)
        matches_key = str((stmt.concept.matches_key(), context_key))
    elif isinstance(stmt, Influence):
        subj_context_key = get_location(stmt.subj)
        obj_context_key = get_location(stmt.obj)
        matches_key = str((stmt.matches_key(), subj_context_key, obj_context_key))
    else:
        matches_key = stmt.matches_key()
    return matches_key


def event_location_time_matches(event):
    """Return Event matches key which takes location and time into account."""
    mk = location_matches(event)
    if not has_time(event):
        return mk
    time = get_time(event)
    matches_key = str((mk, time.start, time.end, time.duration))
    return matches_key


def location_time_matches(stmt):
    """Return matches key which takes location and time into account."""
    if isinstance(stmt, Event):
        return event_location_time_matches(stmt)
    elif isinstance(stmt, Influence):
        subj_mk = event_location_time_matches(stmt.subj)
        obj_mk = event_location_time_matches(stmt.obj)
        return str((stmt.matches_key(), subj_mk, obj_mk))
    else:
        return stmt.matches_key()


def event_location_time_delta_matches(event):
    mk = event_location_time_matches(event)
    if not has_delta(event):
        return mk
    delta = get_delta(event)
    matches_key = str((mk, delta))
    return matches_key


def location_time_delta_matches(stmt):
    if isinstance(stmt, Event):
        return event_location_time_delta_matches(stmt)
    elif isinstance(stmt, Influence):
        subj_mk = event_location_time_delta_matches(stmt.subj)
        obj_mk = event_location_time_delta_matches(stmt.obj)
        return str((stmt.matches_key(), subj_mk, obj_mk))
    else:
        return stmt.matches_key()
