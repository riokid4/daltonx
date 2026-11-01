"""daltonx.ontology.ontology
==============================
1-to-1 port of ``WorldOntology`` from
``indra_world-master/indra_world/ontology/ontology.py`` (lines 1-280),
renamed to ``AISafetyOntology`` with namespace ``'AIS'`` instead of ``'WM'``.

All methods (``initialize``, ``add_ais_ontology``, ``_load_yml``,
``build_relations_new_format``, ``build_relations``, ``add_entry``,
``dump_yml_str``) are preserved 100%.  The module-level factory
``load_ai_safety_ontology`` replaces ``load_world_ontology``.  The
``@register_pipeline`` decorator and INDRA config look-up are removed
(they pull in the full INDRA stack); a simple path/URL default is used
instead.
"""
import logging
import os
import requests
import yaml
from collections import defaultdict

from daltonx.ontology.ontology_graph import IndraOntology, with_initialize

logger = logging.getLogger(__name__)

# Default bundled YAML (ships with this package).
_DEFAULT_YML_PATH = os.path.join(os.path.dirname(__file__),
                                 'ai_safety_ontology.yml')


def get_term(node, prefix):
    """Build a node label for the AIS namespace.

    Source: indra_world-master/indra_world/ontology/ontology.py L18-L21
    (adapted: 'WM' -> 'AIS')
    """
    node = node.replace(' ', '_')
    path = prefix + '/' + node if prefix else node
    return AISafetyOntology.label('AIS', path)


def load_yaml_from_path(path):
    """Return a YAML object loaded from a YAML file path or HTTP URL.

    Source: indra_world-master/indra_world/ontology/ontology.py L24-L38
    """
    if path:
        if path.startswith('http'):
            res = requests.get(path)
            res.raise_for_status()
            yml_str = res.content
        else:
            with open(path, 'r') as fh:
                yml_str = fh.read()
    else:
        return None
    root = yaml.load(yml_str, Loader=yaml.FullLoader)
    return root


class AISafetyOntology(IndraOntology):
    """Represents the ontology used for AI-Safety knowledge graph applications.

    Parameters
    ----------
    url : str
        The file path or HTTP URL pointing to an AI-Safety ontology YAML.

    Attributes
    ----------
    url : str
        The file path or HTTP URL pointing to the ontology YAML.
    yml : list
        The ontology YAML as loaded by the yaml package.

    Source: indra_world-master/indra_world/ontology/ontology.py L41-L261
    (renamed WorldOntology -> AISafetyOntology, namespace 'WM' -> 'AIS')
    """
    name = 'ai_safety'
    version = '1.0'

    def __init__(self, url, yml=None):
        super().__init__()
        self.yml = yml
        self.url = url

    def initialize(self):
        """Load the AI-Safety ontology from the given path/URL and build
        the graph.

        Source: indra_world-master/indra_world/ontology/ontology.py L65-L72
        """
        logger.info('Initializing AI-Safety ontology from %s' % self.url)
        self.add_ais_ontology(self.url)
        self._initialized = True
        self._build_transitive_closure()
        logger.info('Ontology has %d nodes' % len(self))

    def add_ais_ontology(self, url):
        """Load and merge an additional YAML into this ontology.

        Source: indra_world-master/indra_world/ontology/ontology.py L74-L78
        (renamed add_wm_ontology -> add_ais_ontology)
        """
        yml = load_yaml_from_path(url)
        if yml:
            self.yml = yml
        self._load_yml(self.yml)

    @with_initialize
    def dump_yml_str(self):
        """Return a string-serialised form of the loaded YAML.

        Returns
        -------
        str
            The YAML string of the ontology.

        Source: indra_world-master/indra_world/ontology/ontology.py L80-L90
        """
        return yaml.dump(self.yml)

    def _load_yml(self, yml):
        """Parse the YAML structure and populate the graph.

        Supports both the new format (list of {node: ...} dicts) and the
        classic format (arbitrary nested dict).

        Source: indra_world-master/indra_world/ontology/ontology.py L92-L100
        """
        self.clear()
        if isinstance(yml, list) and set(yml[0]) == {'node'}:
            for top_entry in yml:
                self.build_relations_new_format(top_entry['node'], prefix='')
        else:
            for top_entry in yml:
                node = list(top_entry.keys())[0]
                self.build_relations(node, top_entry[node], None)

    def build_relations_new_format(self, node, prefix):
        """Build relations for the new ontology format (>= v3.0).

        Source: indra_world-master/indra_world/ontology/ontology.py L102-L129
        (namespace 'WM' -> 'AIS')
        """
        # Put together the ID and attributes for this node
        name = node['name']
        this_prefix = prefix + '/' + name if prefix else name
        node_data = {'name': name}
        attrs = ['polarity', 'semantic type', 'examples']
        for attr in attrs:
            if attr in node:
                node_data[attr] = node[attr]
        node_label = self.label('AIS', this_prefix)
        opposite = node.get('opposite')
        edges = []
        if opposite:
            opposite_label = self.label('AIS', opposite)
            edges.append((node_label, opposite_label, {'type': 'is_opposite'}))
            edges.append((opposite_label, node_label, {'type': 'is_opposite'}))

        # Now iterate over children nodes and make isa edges
        for child in node.get('children', []):
            self.build_relations_new_format(child['node'], this_prefix)
            child_label = self.label('AIS', this_prefix + '/' +
                                     child['node']['name'])
            edges.append((child_label, node_label, {'type': 'isa'}))

        # Add node and edges to graph
        self.add_nodes_from([(node_label, node_data)])
        self.add_edges_from(edges)

    def build_relations(self, node, tree, prefix):
        """Build relations for the classic ontology format (<= v3.0).

        Source: indra_world-master/indra_world/ontology/ontology.py L132-L200
        (namespace helper get_term uses 'AIS' via module-level get_term)
        """
        nodes = defaultdict(dict)
        edges = []
        this_term = get_term(node, prefix)
        node = node.replace(' ', '_')
        if prefix is not None:
            prefix = prefix.replace(' ', '_')
        this_prefix = prefix + '/' + node if prefix else node
        for entry in tree:
            # This is typically a list of examples which we don't need to
            # independently process
            if isinstance(entry, str):
                continue
            # This is the case of an entry with multiple attributes
            elif isinstance(entry, dict):
                # This is the case of an intermediate node that doesn't have
                # an "OntologyNode" attribute.
                if 'OntologyNode' not in entry:
                    # We take all its children and process them if they are
                    # child ontology entries
                    for child in entry:
                        if child[0] != '_' and child != 'examples' \
                                and isinstance(entry[child], (list, dict)):
                            self.build_relations(child, entry[child],
                                                 this_prefix)
                    # This contains information about a non-leaf node
                    # like examples
                    if 'InnerOntologyNode' in entry:
                        child = None
                # Otherwise this is a leaf term
                else:
                    child = entry['name']
            # This is the case of an intermediate ontology node
            if child is None:
                # Handle opposite entries
                opp = entry.get('opposite')
                if opp:
                    parts = opp.split('/')
                    opp_term = get_term(parts[-1], '/'.join(parts[:-1]))
                    edges.append((opp_term, this_term,
                                  {'type': 'is_opposite'}))
                    edges.append((this_term, opp_term,
                                  {'type': 'is_opposite'}))
                # Handle polarity
                pol = entry.get('polarity')
                if pol is not None:
                    nodes[this_term]['polarity'] = pol
            # This is the case of a leaf term
            elif child[0] != '_' and child != 'examples':
                # Add parenthood relationship
                child_term = get_term(child, this_prefix)
                edges.append((child_term, this_term, {'type': 'isa'}))
                # Handle opposite entries
                opp = entry.get('opposite')
                if opp:
                    parts = opp.split('/')
                    opp_term = get_term(parts[-1], '/'.join(parts[:-1]))
                    edges.append((opp_term, child_term,
                                  {'type': 'is_opposite'}))
                    edges.append((child_term, opp_term,
                                  {'type': 'is_opposite'}))
                # Handle polarity
                pol = entry.get('polarity')
                if pol is not None:
                    nodes[child_term]['polarity'] = pol
        # Now add all the nodes and edges
        self.add_nodes_from([(k, v) for k, v in dict(nodes).items()])
        self.add_edges_from(edges)

    @with_initialize
    def add_entry(self, entry, examples=None, neg_examples=None):
        """Add a new ontology entry with examples.

        This works by adding the entry to the yml attribute first
        and then reloading the entire yaml to build a new graph.

        Parameters
        ----------
        entry : str
            The new entry (slash-separated path, e.g. 'safety/alignment').
        examples : Optional[list of str]
            Examples for the new entry.
        neg_examples : Optional[list of str]
            Negative examples for the new entry.

        Source: indra_world-master/indra_world/ontology/ontology.py L202-L261
        """
        examples = examples if examples else []
        neg_examples = neg_examples if neg_examples else []
        parts = entry.split('/')
        # We start at the root of the YML tree and walk down from there
        root = self.yml
        # We iterate over all the parts of the new grounding entry
        for idx, part in enumerate(parts):
            last_part = (idx == (len(parts) - 1))
            matched_node = None
            for element in root:
                # If this is an OntologyNode
                if element['node']['name'] == part:
                    matched_node = element['node']
                    break
            # If we matched an existing node and this is the last part
            # then we just need to add the given example
            if matched_node:
                if last_part:
                    matched_node['examples'] += examples
                    if neg_examples:
                        if 'neg_examples' in matched_node:
                            matched_node['neg_examples'] += neg_examples
                        else:
                            matched_node['neg_examples'] = neg_examples
                    break
                else:
                    if 'children' not in matched_node:
                        matched_node['children'] = []
                    root = matched_node['children']
            # If we didn't match an existing node, we have to build up
            # a new subtree starting from the current part
            else:
                entry = {'node': {'name': part,
                                  'examples': examples}}
                if neg_examples:
                    entry['node']['neg_examples'] = neg_examples
                root.append(entry)
                if last_part:
                    break
                else:
                    entry['node']['children'] = []
                    root = entry['node']['children']
        self._load_yml(self.yml)


def load_ai_safety_ontology(url=None):
    """Load the AI-Safety ontology from a given path/URL or the bundled YAML.

    Parameters
    ----------
    url : Optional[str]
        A file path or HTTP URL to a YAML ontology.  When *None*, the
        bundled ``ai_safety_ontology.yml`` file is used.

    Returns
    -------
    AISafetyOntology

    Source: indra_world-master/indra_world/ontology/ontology.py L264-L276
    (adapted: removed INDRA config / @register_pipeline dependency)
    """
    use_url = url if url else _DEFAULT_YML_PATH
    return AISafetyOntology(use_url)
