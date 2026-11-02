# -*- coding: utf-8 -*-
"""AI Safety Literature Processor for DaltonX.

Adapted from EidosProcessor's statement-construction mechanics in INDRA/INDRA World
(indra/sources/eidos/processor.py), converting extracted causal claims into
grounded native daltonx statements (Influence, Event, Concept, QualitativeDelta,
AISafetyContext, Evidence).
"""

from __future__ import absolute_import, print_function, unicode_literals

import copy
import logging
from typing import Any, Dict, List, Optional, Union

from daltonx.grounding import Grounder, build_grounder_from_ontology
from daltonx.ontology import AISafetyOntology, load_ai_safety_ontology
from daltonx.statements import (
    AISafetyContext,
    Concept,
    Event,
    Evidence,
    Influence,
    QualitativeDelta,
    RefContext,
    TimeContext,
)

logger = logging.getLogger(__name__)


def normalize_polarity(pol: Any) -> Optional[int]:
    """Normalize various polarity representations to 1, -1, or None."""
    if pol is None:
        return None
    if isinstance(pol, (int, float)):
        if pol > 0:
            return 1
        elif pol < 0:
            return -1
        return None
    if isinstance(pol, str):
        p = pol.strip().lower()
        if p in ('1', '+1', '+', 'inc', 'increase', 'increased', 'increasing', 'positive', 'pos'):
            return 1
        elif p in ('-1', '-', 'dec', 'decrease', 'decreased', 'decreasing', 'negative', 'neg'):
            return -1
        elif p in ('0', 'none', 'neutral', 'unknown', 'null'):
            return None
    return None


class AISafetyProcessor:
    """Processor that constructs grounded daltonx Statements from extracted causal dicts.

    Parameters
    ----------
    extractions : Optional[List[Dict[str, Any]]]
        List of raw causal extraction dictionaries.
    grounder : Optional[Grounder]
        Grounder instance for AIS concepts. If None, built from ontology.
    ontology : Optional[AISafetyOntology]
        Ontology instance used to construct the grounder if grounder is None.
    source_api : str
        Source identifier attached to extracted Evidence objects (default: 'ais').
    """

    def __init__(
        self,
        extractions: Optional[List[Dict[str, Any]]] = None,
        grounder: Optional[Grounder] = None,
        ontology: Optional[AISafetyOntology] = None,
        source_api: str = 'ais',
    ):
        self.extractions = extractions or []
        self.source_api = source_api
        if grounder is not None:
            self.grounder = grounder
        else:
            self.grounder = build_grounder_from_ontology(ontology)
        self.statements: List[Influence] = []

    def extract_causal_relations(self) -> List[Influence]:
        """Process all extractions into Influence statements and return them."""
        self.statements = []
        for item in self.extractions:
            stmt = self.get_causal_relation(item)
            if stmt is not None:
                self.statements.append(stmt)
        return self.statements

    def get_concept(self, text: str) -> Concept:
        """Ground an entity string against the AI Safety Ontology to produce a Concept."""
        if not text:
            return Concept(name='', db_refs={'TEXT': ''})

        raw_str = text.strip()
        matches = self.grounder.ground(raw_str)
        if matches:
            best_match = matches[0]
            # Store top matches scored list for db_refs['AIS']
            ais_groundings = [(m.term.id, m.score) for m in matches[:5]]
            db_refs = {
                'AIS': ais_groundings,
                'TEXT': raw_str,
            }
            # Use entry_name from ontology
            concept_name = best_match.term.entry_name or raw_str
            return Concept(name=concept_name, db_refs=db_refs)
        else:
            # Fallback to ungrounded concept with TEXT
            return Concept(name=raw_str, db_refs={'TEXT': raw_str})

    def get_event(
        self,
        text: str,
        polarity: Any = None,
        adjectives: Optional[List[str]] = None,
        context: Optional[AISafetyContext] = None,
    ) -> Event:
        """Construct a grounded Event statement."""
        concept = self.get_concept(text)
        pol = normalize_polarity(polarity)
        adjs = list(adjectives) if adjectives else []
        delta = QualitativeDelta(polarity=pol, adjectives=adjs)
        return Event(concept=concept, delta=delta, context=context)

    def extract_context(self, raw_data: Dict[str, Any]) -> Optional[AISafetyContext]:
        """Construct an AISafetyContext from raw extraction fields."""
        ctx_data = raw_data.get('context', {})
        if not isinstance(ctx_data, dict):
            ctx_data = {}

        # Merge top-level context shortcuts if present
        merged_ctx = dict(ctx_data)
        for key in (
            'model',
            'model_scale',
            'training_paradigm',
            'architecture',
            'evaluation_setting',
            'empirical_vs_theoretical',
            'environment',
            'geo_location',
            'time',
        ):
            if key in raw_data and key not in merged_ctx:
                merged_ctx[key] = raw_data[key]

        if not merged_ctx:
            return None

        # Build model RefContext
        model_name = merged_ctx.get('model')
        model_db_refs = {}
        if merged_ctx.get('model_scale'):
            model_db_refs['scale'] = str(merged_ctx['model_scale'])
        if merged_ctx.get('architecture'):
            model_db_refs['architecture'] = str(merged_ctx['architecture'])

        model_ref = None
        if model_name or model_db_refs:
            model_ref = RefContext(name=model_name or 'unspecified_model', db_refs=model_db_refs)

        # Build environment RefContext
        env_name = merged_ctx.get('evaluation_setting') or merged_ctx.get('environment')
        env_db_refs = {}
        if merged_ctx.get('training_paradigm'):
            env_db_refs['training_paradigm'] = str(merged_ctx['training_paradigm'])
        if merged_ctx.get('empirical_vs_theoretical'):
            env_db_refs['empirical_vs_theoretical'] = str(merged_ctx['empirical_vs_theoretical'])

        env_ref = None
        if env_name or env_db_refs:
            env_ref = RefContext(name=env_name or 'evaluation_environment', db_refs=env_db_refs)

        # Build geo/jurisdiction RefContext
        geo_name = merged_ctx.get('geo_location')
        geo_ref = RefContext(name=geo_name) if geo_name else None

        # Build TimeContext
        time_entry = merged_ctx.get('time')
        time_ref = None
        if time_entry:
            if isinstance(time_entry, TimeContext):
                time_ref = time_entry
            elif isinstance(time_entry, dict):
                time_ref = TimeContext.from_json(time_entry)
            elif isinstance(time_entry, str):
                time_ref = TimeContext(text=time_entry)

        return AISafetyContext(
            time=time_ref,
            environment=env_ref,
            model=model_ref,
            geo_location=geo_ref,
        )

    def get_evidence(self, raw_data: Dict[str, Any], context: Optional[AISafetyContext] = None) -> Evidence:
        """Construct an Evidence object with annotations, provenance, and epistemics."""
        # Evidence text / quote
        text = (
            raw_data.get('evidence_text')
            or raw_data.get('evidence')
            or raw_data.get('text')
            or raw_data.get('quote')
            or ''
        )

        # Paper metadata & provenance
        page_span = raw_data.get('page_span') or raw_data.get('pages')
        chunk_name = raw_data.get('chunk_name') or raw_data.get('name')
        docname = raw_data.get('docname')
        dockey = raw_data.get('dockey')

        paper_id = (
            raw_data.get('paper_id')
            or raw_data.get('doc_id')
            or raw_data.get('pmid')
            or raw_data.get('arxiv_id')
            or chunk_name
            or docname
        )
        # If paper_id is pure docname and page_span is provided, prefer full page-span citation
        if paper_id and page_span and str(page_span) not in str(paper_id):
            full_span_id = f"{paper_id} {page_span}" if not str(page_span).startswith('pages') else f"{paper_id} {page_span}"
            # preserve original paper_id in annotations but allow full citation in pmid if specified
            paper_id = full_span_id

        title = raw_data.get('title')
        section = raw_data.get('section')
        citation = raw_data.get('citation')

        provenance = []
        doc_info = {}
        if paper_id:
            doc_info['@id'] = str(paper_id)
        if title:
            doc_info['title'] = str(title)
        if citation:
            doc_info['citation'] = str(citation)

        prov_entry: Dict[str, Any] = {}
        if doc_info:
            prov_entry['document'] = doc_info
        if section:
            prov_entry['section'] = str(section)
        if text:
            prov_entry['text'] = str(text)
        if prov_entry:
            provenance.append(prov_entry)

        # Epistemics: negation & hedging
        epistemics: Dict[str, Any] = {}
        negated = raw_data.get('negated')
        if negated:
            epistemics['negated'] = True

        hedged = raw_data.get('hedged')
        if hedged:
            if isinstance(hedged, list):
                epistemics['hedgings'] = hedged
            elif isinstance(hedged, str):
                epistemics['hedgings'] = [hedged]
            else:
                epistemics['hedged'] = True

        annotations = {
            'found_by': raw_data.get('found_by', 'LLMExtractor'),
            'provenance': provenance,
        }
        if paper_id:
            annotations['paper_id'] = str(paper_id)
        if page_span:
            annotations['page_span'] = str(page_span)
        if chunk_name:
            annotations['chunk_name'] = str(chunk_name)
        if docname:
            annotations['docname'] = str(docname)
        if dockey:
            annotations['dockey'] = str(dockey)
        if section:
            annotations['section'] = str(section)
        if citation:
            annotations['citation'] = str(citation)

        return Evidence(
            source_api=self.source_api,
            text=text,
            pmid=paper_id,
            annotations=annotations,
            epistemics=epistemics,
            context=context,
        )

    def get_causal_relation(self, raw_data: Dict[str, Any]) -> Optional[Influence]:
        """Construct an Influence statement from a single causal extraction dictionary."""
        # Extract cause parameters
        cause_text = (
            raw_data.get('cause_text')
            or raw_data.get('cause')
            or raw_data.get('subj_text')
            or raw_data.get('subject')
        )
        # Extract effect parameters
        effect_text = (
            raw_data.get('effect_text')
            or raw_data.get('effect')
            or raw_data.get('obj_text')
            or raw_data.get('object')
        )

        if not cause_text or not effect_text:
            logger.warning(
                "Skipping extraction missing cause or effect: %s -> %s",
                cause_text,
                effect_text,
            )
            return None

        cause_pol = raw_data.get('cause_polarity', raw_data.get('subj_polarity'))
        cause_adjs = raw_data.get('cause_adjectives', raw_data.get('subj_adjectives', []))

        effect_pol = raw_data.get('effect_polarity', raw_data.get('obj_polarity'))
        effect_adjs = raw_data.get('effect_adjectives', raw_data.get('obj_adjectives', []))

        # Context
        context = self.extract_context(raw_data)

        # Build Events
        subj = self.get_event(
            text=cause_text,
            polarity=cause_pol,
            adjectives=cause_adjs,
            context=context,
        )
        obj = self.get_event(
            text=effect_text,
            polarity=effect_pol,
            adjectives=effect_adjs,
            context=context,
        )

        # Build Evidence
        evidence = self.get_evidence(raw_data, context=context)

        # Preserve polarity and adjective metadata in annotations (INDRA World standard)
        evidence.annotations['subj_text'] = cause_text
        evidence.annotations['obj_text'] = effect_text
        evidence.annotations['subj_polarity'] = subj.delta.polarity
        evidence.annotations['obj_polarity'] = obj.delta.polarity
        evidence.annotations['subj_adjectives'] = subj.delta.adjectives
        evidence.annotations['obj_adjectives'] = obj.delta.adjectives
        if context:
            evidence.annotations['context'] = context.to_json()

        # Build Influence statement
        stmt = Influence(subj, obj, evidence=[evidence])

        # Prior belief or confidence if specified
        confidence = raw_data.get('confidence', raw_data.get('belief'))
        if confidence is not None:
            try:
                stmt.belief = float(confidence)
            except (ValueError, TypeError):
                pass

        return stmt
