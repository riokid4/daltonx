# -*- coding: utf-8 -*-
"""Causal Analysis Graph (CAG) Assembler for DaltonX.

1-to-1 port of ``indra_world.assemblers.cag.assembler.CAGAssembler``
adapted for :mod:`daltonx.statements` (``Influence``, ``Concept``) and
the ``'AIS'`` ontology namespace.

Key differences from the INDRA World original:
- Imports ``Influence`` from ``daltonx.statements`` instead of
  ``indra.statements``.
- ``_node_name`` resolves ``'AIS'`` grounding by default
  (``grounding_ontology='AIS'``).
- ``print_tsv`` / ``_get_factor`` handles the ``'AIS'`` source API path
  in addition to the original eidos/hume/cwms/sofia branches.
- All other logic (graph construction, polarity, CytoscapeJS export,
  Jupyter JS generation) is preserved verbatim.
"""
from __future__ import absolute_import, print_function, unicode_literals

import json
import logging
import os

import networkx as nx

from daltonx.statements import Influence

# Python 2 compatibility shim
try:
    basestring  # noqa: F821
except NameError:
    basestring = str

logger = logging.getLogger(__name__)


class CAGAssembler:
    """Assembles a Causal Analysis Graph from DaltonX ``Influence`` statements.

    Parameters
    ----------
    stmts : list[daltonx.statements.Statement], optional
        Statements to assemble.  Only :class:`~daltonx.statements.Influence`
        instances are used; others are silently ignored.

    Attributes
    ----------
    statements : list
        All statements passed in (or added via :meth:`add_statements`).
    CAG : nx.MultiDiGraph or None
        The assembled graph, populated after :meth:`make_model` is called.
    grounding_threshold : float or None
        Minimum AIS grounding score for using the normalised name as node
        label; ``None`` means always use ``concept.name``.
    grounding_ontology : str
        Namespace key used to look up groundings in ``concept.db_refs``.
        Defaults to ``'AIS'``.
    """

    def __init__(self, stmts=None):
        self.statements = list(stmts) if stmts else []
        self.grounding_threshold = None
        self.grounding_ontology = 'AIS'
        self.CAG = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_statements(self, stmts):
        """Append *stmts* to the assembler's statement list."""
        self.statements += stmts

    def make_model(self, grounding_ontology='AIS', grounding_threshold=None):
        """Build and return a ``networkx.MultiDiGraph`` CAG.

        Parameters
        ----------
        grounding_ontology : str, optional
            Namespace key for concept groundings (default ``'AIS'``).
        grounding_threshold : float, optional
            Minimum grounding score to use normalised name as node label.

        Returns
        -------
        nx.MultiDiGraph
            Directed multigraph where nodes are concept names and edges carry
            polarity/style metadata.
        """
        if grounding_threshold is not None:
            self.grounding_threshold = grounding_threshold
        self.grounding_ontology = grounding_ontology

        # Filter to Influence statements only
        statements = [s for s in self.statements if isinstance(s, Influence)]

        self.CAG = nx.MultiDiGraph()

        for s in statements:
            has_both_polarity = (
                s.subj.delta.polarity is not None
                and s.obj.delta.polarity is not None
            )

            # Add subject and object nodes
            for node, delta in zip(
                (s.subj.concept, s.obj.concept),
                (s.subj.delta, s.obj.delta),
            ):
                self.CAG.add_node(
                    self._node_name(node),
                    simulable=has_both_polarity,
                    mods=delta.adjectives,
                )

            # Determine edge visual style
            linestyle = 'solid' if has_both_polarity else 'dotted'
            if has_both_polarity:
                same_polarity = s.subj.delta.polarity == s.obj.delta.polarity
                if same_polarity:
                    target_arrow_shape, linecolor = ('circle', 'green')
                else:
                    target_arrow_shape, linecolor = ('tee', 'maroon')
            else:
                target_arrow_shape, linecolor = ('triangle', 'maroon')

            # Collect provenance from first evidence piece
            provenance = []
            if s.evidence:
                provenance = s.evidence[0].annotations.get('provenance', [])
                if provenance:
                    provenance[0]['text'] = s.evidence[0].text

            self.CAG.add_edge(
                self._node_name(s.subj.concept),
                self._node_name(s.obj.concept),
                subj_polarity=s.subj.delta.polarity,
                subj_adjectives=s.subj.delta.adjectives,
                obj_polarity=s.obj.delta.polarity,
                obj_adjectives=s.obj.delta.adjectives,
                linestyle=linestyle,
                linecolor=linecolor,
                targetArrowShape=target_arrow_shape,
                provenance=provenance,
                belief=getattr(s, 'belief', 1.0),
            )

        return self.CAG

    def print_tsv(self, file_name):
        """Write assembled statements to a TSV file.

        Parameters
        ----------
        file_name : str
            Path of the output ``.tsv`` file.
        """
        def _get_factor(stmt, concept, delta, evidence, raw_name):
            src = evidence.source_api
            if src == 'ais':
                grounding_list = concept.db_refs.get('AIS', [])
                factor_norm = grounding_list[0][0] if grounding_list else ''
            elif src == 'eidos':
                grounding_list = concept.db_refs.get(
                    self.grounding_ontology, [])
                factor_norm = grounding_list[0][0] if grounding_list else ''
            elif src == 'hume':
                grounding_list = concept.db_refs.get('HUME', [])
                factor_norm = grounding_list[0][0] if grounding_list else ''
            elif src == 'cwms':
                factor_norm = concept.db_refs.get('CWMS', '')
            elif src == 'sofia':
                factor_norm = concept.name
            else:
                factor_norm = concept.name

            mods = ', '.join(delta.adjectives)
            if delta.polarity == -1:
                pol = 'decrease'
            elif delta.polarity == 1:
                pol = 'increase'
            else:
                pol = ''
            name = raw_name if raw_name else concept.name
            return name, factor_norm, mods, pol

        def _get_evidence(evidence):
            sent_id = ''
            location = evidence.annotations.get('Location', '')
            time = evidence.annotations.get('Time', '')
            ref = evidence.pmid if evidence.pmid is not None else ''
            return ref, evidence.source_api, sent_id, location, time, \
                evidence.text

        header = [
            'Source', 'System', 'Sentence ID',
            'Factor A Text', 'Factor A Normalization',
            'Factor A Modifiers', 'Factor A Polarity',
            'Relation Text', 'Relation Normalization',
            'Relation Modifiers',
            'Factor B Text', 'Factor B Normalization',
            'Factor B Modifiers', 'Factor B Polarity',
            'Location', 'Time', 'Evidence',
            'Relation ID',
        ]

        statements = [s for s in self.statements if isinstance(s, Influence)]
        all_rows = []
        for idx, stmt in enumerate(statements):
            for evidence in stmt.evidence:
                source, system, sent_id, location, time, text = \
                    _get_evidence(evidence)
                subj_text = evidence.annotations.get('subj_text', '')
                obj_text = evidence.annotations.get('obj_text', '')
                factor_a, factor_a_norm, mod_a, pol_a = _get_factor(
                    stmt, stmt.subj.concept, stmt.subj.delta,
                    evidence, subj_text)
                factor_b, factor_b_norm, mod_b, pol_b = _get_factor(
                    stmt, stmt.obj.concept, stmt.obj.delta,
                    evidence, obj_text)
                row = [
                    source, system, sent_id,
                    factor_a, factor_a_norm, mod_a, pol_a,
                    'influences', '', '',
                    factor_b, factor_b_norm, mod_b, pol_b,
                    location, time, text, str(idx),
                ]
                if row not in all_rows:
                    all_rows.append(row)

        with open(file_name, 'w') as fh:
            fh.write('\t'.join(header) + '\n')
            for row in sorted(all_rows, key=lambda x: x[0]):
                fh.write('\t'.join(str(c) for c in row) + '\n')

    def export_to_cytoscapejs(self):
        """Return the CAG as a CytoscapeJS-compatible dict.

        Returns
        -------
        dict
            ``{'nodes': [...], 'edges': [...]}`` ready for CytoscapeJS.
        """
        if self.CAG is None:
            raise RuntimeError("Call make_model() before export_to_cytoscapejs().")

        def _create_edge_data_dict(e):
            """Build edge data dict from a MultiDiGraph edge 4-tuple."""
            # e = (source_id, target_id, key, data_dict)
            provenance = e[3].get('provenance')
            if provenance:
                tooltip = provenance[0]
                tooltip.pop('@type', None)
            else:
                tooltip = None
            return {
                'id': e[0] + '_' + e[1],
                'source': e[0],
                'target': e[1],
                'linestyle': e[3]['linestyle'],
                'linecolor': e[3]['linecolor'],
                'targetArrowShape': e[3]['targetArrowShape'],
                'subj_adjectives': e[3]['subj_adjectives'],
                'subj_polarity': e[3]['subj_polarity'],
                'obj_adjectives': e[3]['obj_adjectives'],
                'obj_polarity': e[3]['obj_polarity'],
                'tooltip': tooltip,
                'simulable': not (
                    e[3]['obj_polarity'] is None
                    or e[3]['subj_polarity'] is None
                ),
                'belief': e[3].get('belief', 1.0),
            }

        return {
            'nodes': [
                {'data': {
                    'id': n[0],
                    'simulable': n[1]['simulable'],
                    'tooltip': 'Modifiers: ' + json.dumps(n[1]['mods']),
                }}
                for n in self.CAG.nodes(data=True)
            ],
            'edges': [
                {'data': _create_edge_data_dict(e)}
                for e in self.CAG.edges(data=True, keys=True)
            ],
        }

    def generate_jupyter_js(self, cyjs_style=None, cyjs_layout=None):
        """Generate a CytoscapeJS Javascript snippet for Jupyter notebooks.

        Parameters
        ----------
        cyjs_style : dict, optional
            CytoscapeJS style override.
        cyjs_layout : dict, optional
            CytoscapeJS layout override.

        Returns
        -------
        str
            Javascript string suitable for ``IPython.display.Javascript``.
        """
        cyjs_elements = self.export_to_cytoscapejs()

        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, 'cag_template.js'), 'r') as fh:
            template = fh.read()
        with open(os.path.join(here, 'cag_style.json'), 'r') as fh:
            style = json.load(fh)

        if cyjs_style is None:
            cyjs_style = style['style']
        if cyjs_layout is None:
            cyjs_layout = style['layout']

        formatted_args = tuple(
            json.dumps(x, indent=2)
            for x in (cyjs_elements, cyjs_style, cyjs_layout)
        )
        return template % formatted_args

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _node_name(self, concept):
        """Return a display label for *concept*.

        If a grounding threshold is set and the concept has an AIS grounding
        above that threshold, the normalised ontology path fragment is used.
        Otherwise ``concept.name`` is returned (capitalised).
        """
        if (
            self.grounding_threshold is not None
            and concept.db_refs.get(self.grounding_ontology)
            and concept.db_refs[self.grounding_ontology][0][1]
                > self.grounding_threshold
        ):
            entry = concept.db_refs[self.grounding_ontology][0][0]
            return entry.split('/')[-1].replace('_', ' ').capitalize()
        return concept.name.capitalize()
