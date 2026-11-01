import sys
import uuid
import logging
from copy import copy, deepcopy
from collections import OrderedDict as _o

from .util import make_hash, mk_str, stmt_type
from .concept import Concept
from .delta import Delta, QualitativeDelta, QuantitativeState
from .evidence import Evidence
from .context import Context


__all__ = ['Statement', 'Event', 'Influence', 'stmt_type', 'mk_str']

logger = logging.getLogger(__name__)


class Statement(object):
    """The parent class of all statements.

    Parameters
    ----------
    evidence : None or Evidence or list of Evidence
        If a list of Evidence objects is passed to the constructor, the value
        is set to this list. If a bare Evidence object is passed, it is
        enclosed in a list. If no evidence is passed (the default), the value
        is set to an empty list.
    supports : list of Statement
        Statements that this Statement supports.
    supported_by : list of Statement
        Statements supported by this statement.
    """
    _agent_order = NotImplemented

    def __init__(self, evidence=None, supports=None, supported_by=None):
        if evidence is None:
            self.evidence = []
        elif isinstance(evidence, Evidence):
            self.evidence = [evidence]
        elif isinstance(evidence, list):
            self.evidence = evidence
        else:
            raise ValueError('evidence must be an Evidence object, a list '
                             '(of Evidence objects), or None.')

        self.supports = supports if supports else []
        self.supported_by = supported_by if supported_by else []
        self.belief = 1.0
        self.uuid = '%s' % uuid.uuid4()
        self._full_hash = None
        self._shallow_hash = None

    def matches_key(self):
        raise NotImplementedError("Method must be implemented in child class.")

    def matches(self, other):
        return self.matches_key() == other.matches_key()

    def get_hash(self, shallow=True, refresh=False, matches_fun=None):
        """Get a hash for this Statement.

        There are two types of hash, "shallow" and "full". A shallow hash is
        as unique as the information carried by the statement, i.e. it is a hash
        of the `matches_key` (14 nibbles).

        A full hash includes, in addition to the matches key, information from
        the evidence of the statement (16 nibbles).
        """
        if shallow:
            if not hasattr(self, '_shallow_hash') or self._shallow_hash is None \
                    or refresh or matches_fun:
                matches_key = matches_fun(self) if matches_fun else self.matches_key()
                self._shallow_hash = make_hash(matches_key, 14)
            ret = self._shallow_hash
        else:
            if not hasattr(self, '_full_hash') or self._full_hash is None \
                    or refresh or matches_fun:
                ev_mk_list = sorted([ev.matches_key() for ev in self.evidence])
                matches_key = matches_fun(self) if matches_fun else self.matches_key()
                self._full_hash = make_hash(matches_key + str(ev_mk_list), 16)
            ret = self._full_hash
        return ret

    def _tag_evidence(self):
        """Set all the Evidence stmt_tag to my deep matches-key hash."""
        h = self.get_hash(shallow=False)
        for ev in self.evidence:
            ev.stmt_tag = h

    def agent_list(self, deep_sorted=False):
        """Get the canonicalized agent list."""
        ag_list = []
        for ag_name in self._agent_order:
            ag_attr = getattr(self, ag_name)
            if isinstance(ag_attr, Concept) or ag_attr is None:
                ag_list.append(ag_attr)
            elif isinstance(ag_attr, list):
                if not all([isinstance(ag, Concept) for ag in ag_attr]):
                    raise TypeError("Expected all elements of list to be Concept, but got: %s"
                                    % {type(ag) for ag in ag_attr})
                ag_list.extend(ag_attr)
            else:
                raise TypeError("Expected type Concept or list, got type %s." % type(ag_attr))
        return ag_list

    def real_agent_list(self):
        """Return all agents in the statement that are not None."""
        return [a for a in self.agent_list() if a is not None]

    def entities_match(self, other):
        self_key = self.entities_match_key()
        other_key = other.entities_match_key()
        if len(self_key) != len(other_key):
            return False
        for self_agent, other_agent in zip(self_key, other_key):
            if self_agent is None or other_agent is None:
                continue
            if self_agent != other_agent:
                return False
        return True

    def entities_match_key(self):
        key = tuple(a.entity_matches_key() if a is not None else None
                    for a in self.agent_list())
        return key

    def equals(self, other):
        return self.types_equals(other) \
            and self.agents_equal(other) \
            and self.evidence_equals(other)

    def types_equals(self, other):
        if stmt_type(self) != stmt_type(other):
            return False
        return True

    def agents_equal(self, other):
        if len(self.agent_list()) == len(other.agent_list()):
            for s, o in zip(self.agent_list(), other.agent_list()):
                if (s is None and o is not None) or \
                        (s is not None and o is None):
                    return False
                if s is not None and o is not None and not s.equals(o):
                    return False
        else:
            return False
        return True

    def evidence_equals(self, other):
        if len(self.evidence) == len(other.evidence):
            for s, o in zip(self.evidence, other.evidence):
                if not s.equals(o):
                    return False
        else:
            return False
        return True

    def contradicts(self, other, ontology):
        return False

    def to_json(self, use_sbo=False, matches_fun=None):
        """Return serialized Statement as a JSON dict."""
        stmt_type_name = type(self).__name__
        all_stmts = [self] + self.supports + self.supported_by
        for st in all_stmts:
            if not hasattr(st, 'uuid') or not st.uuid:
                st.uuid = '%s' % uuid.uuid4()

        json_dict = _o(type=stmt_type_name)
        json_dict['belief'] = self.belief
        if self.evidence:
            evidence = [ev.to_json() for ev in self.evidence]
            json_dict['evidence'] = evidence
        json_dict['id'] = '%s' % self.uuid
        json_dict['matches_hash'] = \
            '%s' % self.get_hash(shallow=True, refresh=True,
                                 matches_fun=matches_fun)
        if self.supports:
            json_dict['supports'] = ['%s' % st.uuid for st in self.supports]
        if self.supported_by:
            json_dict['supported_by'] = ['%s' % st.uuid for st in self.supported_by]
        return json_dict

    @classmethod
    def _from_json(cls, json_dict):
        stmt_type_name = json_dict.get('type')
        stmt_cls = getattr(sys.modules[__name__], stmt_type_name, None)
        if not stmt_cls:
            raise ValueError('Unknown statement type %s' % stmt_type_name)
        stmt = stmt_cls._from_json(json_dict)
        evidence = json_dict.get('evidence', [])
        stmt.evidence = [Evidence._from_json(ev) for ev in evidence]
        stmt.supports = json_dict.get('supports', [])[:]
        stmt.supported_by = json_dict.get('supported_by', [])[:]
        stmt.belief = json_dict.get('belief', 1.0)
        stmt_id = json_dict.get('id')
        if not stmt_id:
            stmt_id = '%s' % uuid.uuid4()
        stmt.uuid = stmt_id
        json_matches_hash = json_dict.get('matches_hash')
        if json_matches_hash:
            try:
                stmt._shallow_hash = int(json_matches_hash)
            except ValueError:
                pass
        return stmt

    def make_generic_copy(self, deeply=False):
        """Make a new matching Statement with no provenance."""
        if deeply:
            kwargs = deepcopy(self.__dict__)
        else:
            kwargs = self.__dict__.copy()
        for attr in ['evidence', 'uuid', 'supports', 'supported_by']:
            kwargs.pop(attr, None)
        my_belief = kwargs.pop('belief', 1.0)
        my_hash = kwargs.pop('_full_hash', None)
        my_shallow_hash = kwargs.pop('_shallow_hash', None)
        new_instance = self.__class__(**kwargs)
        new_instance._full_hash = my_hash
        new_instance._shallow_hash = my_shallow_hash
        new_instance.belief = my_belief
        return new_instance

    def __repr__(self):
        return str(self)


class Event(Statement):
    """An event representing the change of a Concept.

    Attributes
    ----------
    concept : Concept
        The concept over which the event is defined.
    delta : Delta
        Represents a change in the concept, with a polarity
        and an adjectives entry.
    context : Context
        The context associated with the event.
    """
    _agent_order = ['concept']

    def __init__(self, concept, delta=None, context=None, evidence=None,
                 supports=None, supported_by=None):
        super().__init__(evidence, supports, supported_by)
        self.concept = concept
        self.delta = delta if delta else QualitativeDelta(polarity=None, adjectives=None)
        self.context = context

    def matches_key(self):
        mk = (self.concept.matches_key(),)
        return str(mk)

    def refinement_of(self, other, ontology, entities_refined=False,
                      ignore_polarity=False):
        concept_ref = self.concept.refinement_of(
            other.concept, ontology, entities_refined=entities_refined)
        if ignore_polarity:
            pol_ref = True
        else:
            pol_ref = (self.delta.polarity and not other.delta.polarity) or \
                self.delta.polarity == other.delta.polarity
        return concept_ref and pol_ref

    def equals(self, other):
        return self.concept.equals(other.concept) and \
            self.delta.equals(other.delta)

    def to_json(self, with_evidence=True, use_sbo=False, matches_fun=None):
        generic = super(Event, self).to_json(use_sbo=use_sbo, matches_fun=matches_fun)
        json_dict = _o(type=generic['type'],
                       concept=self.concept.to_json(),
                       delta=self.delta.to_json())
        if self.context:
            json_dict['context'] = self.context.to_json()
        if with_evidence and 'evidence' in generic:
            json_dict['evidence'] = generic['evidence']
        json_dict.update(generic)
        json_dict['matches_hash'] = int(json_dict['matches_hash'])
        return json_dict

    @classmethod
    def _from_json(cls, json_dict):
        concept = Concept._from_json(json_dict['concept'])
        delta = Delta.from_json(json_dict['delta'])
        context_entry = json_dict.get('context')
        if context_entry:
            context = Context.from_json(context_entry)
        else:
            context = None
        stmt = cls(concept, delta=delta, context=context)
        return stmt

    def flip_polarity(self, agent_idx=None):
        if self.delta.polarity == 1:
            self.delta.polarity = -1
        elif self.delta.polarity == -1:
            self.delta.polarity = 1

    def __str__(self):
        return '%s(%s)' % (type(self).__name__, self.concept.name)


class Influence(Statement):
    """An influence on the quantity or state of a concept of interest.

    Parameters
    ----------
    subj : Event
        The event which acts as the influencer (cause).
    obj : Event
        The event which acts as the influencee (effect).
    evidence : None or Evidence or list of Evidence
        Evidence objects in support of the statement.
    """
    _agent_order = ['subj', 'obj']

    def __init__(self, subj, obj, evidence=None, supports=None, supported_by=None):
        super(Influence, self).__init__(evidence=evidence, supports=supports,
                                        supported_by=supported_by)
        self.subj = subj
        self.obj = obj

    def agent_list(self, deep_sorted=False):
        return [self.subj.concept, self.obj.concept]

    def refinement_of(self, other, ontology, entities_refined=False):
        if stmt_type(self) != stmt_type(other):
            return False
        delta_refinement = self.delta_refinement_of(other)
        if not delta_refinement:
            return False
        subj_refinement = self.subj.concept.refinement_of(
            other.subj.concept, ontology, entities_refined=entities_refined)
        obj_refinement = self.obj.concept.refinement_of(
            other.obj.concept, ontology, entities_refined=entities_refined)
        return (subj_refinement and obj_refinement)

    def delta_refinement_of(self, other):
        op = other.overall_polarity()
        sp = self.overall_polarity()
        if self.polarity_count() < other.polarity_count():
            delta_refinement = False
        elif sp is not None and op is None:
            delta_refinement = True
        else:
            delta_refinement = (op == sp)
        return delta_refinement

    def equals(self, other):
        equals = super(Influence, self).equals(other) and \
            self.subj.equals(other.subj) and self.obj.equals(other.obj)
        return equals

    def matches_key(self):
        key = (stmt_type(self, True),
               self.subj.matches_key(),
               self.obj.matches_key(),
               self.polarity_count(),
               self.overall_polarity())
        return mk_str(key)

    def contradicts(self, other, ontology):
        if stmt_type(self) != stmt_type(other):
            return False

        subj_ref = self.subj.concept.refinement_of(other.subj.concept, ontology) or \
            other.subj.concept.refinement_of(self.subj.concept, ontology)
        obj_ref = self.obj.concept.refinement_of(other.obj.concept, ontology) or \
            other.obj.concept.refinement_of(self.obj.concept, ontology)
        subj_opp = self.subj.concept.is_opposite(other.subj.concept, ontology)
        obj_opp = self.obj.concept.is_opposite(other.obj.concept, ontology)
        sp = self.overall_polarity()
        op = other.overall_polarity()

        # If all entities are compatible or mutually opposites but polarities differ
        if (subj_ref and obj_ref) or (subj_opp and obj_opp):
            if sp is not None and op is not None and sp != op:
                return True
        # If one entity is opposite and other compatible, and polarities are the same
        if (subj_ref and obj_opp) or (subj_opp and obj_ref):
            if sp is not None and op is not None and sp == op:
                return True

        return False

    def overall_polarity(self):
        p1 = self.subj.delta.polarity
        p2 = self.obj.delta.polarity
        if p1 is None and p2 is None:
            pol = None
        elif p2 is None:
            pol = p1
        elif p1 is None:
            pol = p2
        else:
            pol = p1 * p2
        return pol

    def flip_polarity(self, agent_idx):
        if agent_idx == 0:
            self.subj.flip_polarity()
        elif agent_idx == 1:
            self.obj.flip_polarity()

    def polarity_count(self):
        return ((1 if self.subj.delta.polarity is not None else 0) +
                (1 if self.obj.delta.polarity is not None else 0))

    def to_json(self, use_sbo=False, matches_fun=None):
        generic = super(Influence, self).to_json(use_sbo=use_sbo, matches_fun=matches_fun)
        json_dict = _o(type=generic['type'],
                       subj=self.subj.to_json(with_evidence=False, matches_fun=matches_fun),
                       obj=self.obj.to_json(with_evidence=False, matches_fun=matches_fun))
        json_dict.update(generic)
        json_dict['matches_hash'] = int(json_dict['matches_hash'])
        return json_dict

    @classmethod
    def _from_json(cls, json_dict):
        subj = json_dict.get('subj')
        obj = json_dict.get('obj')
        if subj:
            subj = Event._from_json(subj)
        if obj:
            obj = Event._from_json(obj)
        stmt = cls(subj, obj)
        return stmt

    def __str__(self):
        return "%s(%s, %s)" % (type(self).__name__, str(self.subj), str(self.obj))
