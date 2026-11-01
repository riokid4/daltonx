"""daltonx.ontology.ontology_graph
===================================
1-to-1 port of ``IndraOntology`` from
``indra-master/indra/ontology/ontology_graph.py`` (lines 1-687).

Every method, decorator, and constant is preserved exactly.  Only the
module-level docstring is new; the class name remains ``IndraOntology`` so
that ``AISafetyOntology`` (in ``ontology.py``) can inherit from it without
any behavioural changes.
"""
import logging
import networkx
import functools
from collections import deque
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def with_initialize(func):
    """Decorator that ensures the ontology is initialised before any
    user-facing method is called.

    Source: indra-master/indra/ontology/ontology_graph.py L10-L16
    """
    @functools.wraps(func)
    def wrapper(obj, *args, **kwargs):
        if not obj._initialized:
            obj.initialize()
        return func(obj, *args, **kwargs)
    return wrapper


class IndraOntology(networkx.DiGraph):
    """A directed graph representing entities and their properties
    as nodes  and ontological relationships between the entities as
    edges.

    Attributes
    ----------
    name : str
        A prefix/name for the ontology, used for the purposes of caching.
    version : str
        A version for the ontology, used for the purposes of caching.

    Source: indra-master/indra/ontology/ontology_graph.py L19-L687
    """
    version = None
    name = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialized = False
        self.name_to_grounding = {}
        self.transitive_closure = set()
        self._isa_counter = 0
        self._isrel_counter = 0

    def initialize(self):
        """Initialize the ontology by adding nodes and edges.

        By convention, ontologies are implemented such that the constructor
        does not add all the nodes and edges, which can take a long time.
        This function is called automatically when any of the user-facing
        methods of IndraOntology is called. This way, the ontology is only
        fully constructed if it is used.

        Source: indra-master/indra/ontology/ontology_graph.py L42-L53
        """
        raise NotImplementedError('The initialize method needs to be '
                                  'implemented when subclassing '
                                  'IndraOntology')

    # ------------------------------------------------------------------
    # Path / relationship checks
    # ------------------------------------------------------------------

    @with_initialize
    def _check_path(self, ns1, id1, ns2, id2, edge_types):
        """Return True if there is a directed path of the given edge types.

        Source: indra-master/indra/ontology/ontology_graph.py L55-L67
        """
        try:
            target = (ns2, id2)
            if target in self._transitive_rel(ns1, id1, self.child_rel,
                                              edge_types, target):
                return True
            else:
                return False
        # This typically happens if the node is missing from
        # the graph. Is there a more specific error type?
        except networkx.NetworkXError:
            return False

    @staticmethod
    def get_ns_id(node):
        """Return the name space and ID of a given node from its label.

        Parameters
        ----------
        node : str
            A node's label.

        Returns
        -------
        tuple(str, str)
            A tuple of the node's name space and ID.

        Source: indra-master/indra/ontology/ontology_graph.py L69-L83
        """
        return IndraOntology.reverse_label(node)

    @staticmethod
    def get_ns(node):
        """Return the name space of a given node from its label.

        Parameters
        ----------
        node : str
            A node's label.

        Returns
        -------
        str
            The node's name space.

        Source: indra-master/indra/ontology/ontology_graph.py L85-L99
        """
        return IndraOntology.get_ns_id(node)[0]

    @staticmethod
    def get_id(node):
        """Return the name ID a given node from its label.

        Parameters
        ----------
        node : str
            A node's label.

        Returns
        -------
        str
            The node's ID within its name space.

        Source: indra-master/indra/ontology/ontology_graph.py L101-L115
        """
        return IndraOntology.get_ns_id(node)[1]

    @with_initialize
    def isrel(self, ns1, id1, ns2, id2, rels):
        """Return True if the two entities are related with a given rel.

        Parameters
        ----------
        ns1 : str
            The first entity's name space.
        id1 : str
            The first entity's ID.
        ns2 : str
            The second entity's name space.
        id2 : str
            The second entity's ID.
        rels : iterable of str
            A set of edge types to traverse when determining
            if the first entity is related to the second entity.

        Returns
        -------
        bool
            True if the first entity is related to the second with
            a directed path containing edges with types in `rels`.
            Otherwise False.

        Source: indra-master/indra/ontology/ontology_graph.py L117-L144
        """
        self._isrel_counter += 1
        return self._check_path(ns1, id1, ns2, id2, rels)

    @with_initialize
    def isa(self, ns1, id1, ns2, id2):
        """Return True if the first entity is related to the second as 'isa'.

        Parameters
        ----------
        ns1 : str
            The first entity's name space.
        id1 : str
            The first entity's ID.
        ns2 : str
            The second entity's name space.
        id2 : str
            The second entity's ID.

        Returns
        -------
        bool
            True if the first entity is related to the second with
            a directed path containing edges with type 'isa'.
            Otherwise False.

        Source: indra-master/indra/ontology/ontology_graph.py L146-L168
        """
        return self.isrel(ns1, id1, ns2, id2, rels={'isa'})

    @with_initialize
    def partof(self, ns1, id1, ns2, id2):
        """Return True if the first entity is related to the second as 'partof'.

        Parameters
        ----------
        ns1 : str
            The first entity's name space.
        id1 : str
            The first entity's ID.
        ns2 : str
            The second entity's name space.
        id2 : str
            The second entity's ID.

        Returns
        -------
        bool
            True if the first entity is related to the second with
            a directed path containing edges with type 'partof'.
            Otherwise False.

        Source: indra-master/indra/ontology/ontology_graph.py L170-L192
        """
        return self.isrel(ns1, id1, ns2, id2, rels={'partof'})

    @with_initialize
    def isa_or_partof(self, ns1, id1, ns2, id2):
        """Return True if the first entity is related to the second as 'isa'
        or 'partof'.

        Parameters
        ----------
        ns1 : str
            The first entity's name space.
        id1 : str
            The first entity's ID.
        ns2 : str
            The second entity's name space.
        id2 : str
            The second entity's ID.

        Returns
        -------
        bool
            True if the first entity is related to the second with
            a directed path containing edges with type 'isa' or 'partof'.
            Otherwise False.

        Source: indra-master/indra/ontology/ontology_graph.py L194-L221
        """
        self._isa_counter += 1
        if self.transitive_closure:
            return (self.label(ns1, id1),
                    self.label(ns2, id2)) in self.transitive_closure
        return self.isrel(ns1, id1, ns2, id2, rels={'isa', 'partof'})

    @with_initialize
    def maps_to(self, ns1, id1, ns2, id2):
        """Return True if the first entity has an xref to the second.

        Parameters
        ----------
        ns1 : str
            The first entity's name space.
        id1 : str
            The first entity's ID.
        ns2 : str
            The second entity's name space.
        id2 : str
            The second entity's ID.

        Returns
        -------
        bool
            True if the first entity is related to the second with
            a directed path containing edges with type 'xref'.
            Otherwise False.

        Source: indra-master/indra/ontology/ontology_graph.py L223-L245
        """
        return self._check_path(ns1, id1, ns2, id2, {'xref'})

    @with_initialize
    def map_to(self, ns1, id1, ns2):
        """Return an entity that is a unique xref of an entity
        in a given name space.

        This function first finds all mappings via xrefs edges
        from the given first entity to the given second
        name space. If exactly one such mapping target is found, the
        target is returned. Otherwise, None is returned.

        Parameters
        ----------
        ns1 : str
            The first entity's name space.
        id1 : str
            The first entity's ID.
        ns2 : str
            The second entity's name space.

        Returns
        -------
        str
            The name space of the second entity.
        str
            The ID of the second entity in the given name space.

        Source: indra-master/indra/ontology/ontology_graph.py L247-L279
        """
        targets = [target for target in
                   self.descendants_rel(ns1, id1, {'xref'})
                   if target[0] == ns2]
        if len(targets) == 1:
            return targets[0]
        return None

    # ------------------------------------------------------------------
    # Graph traversal
    # ------------------------------------------------------------------

    @with_initialize
    def _transitive_rel(self, ns, id, rel_fun, rel_types, target=None):
        """BFS traversal over edges of the given types.

        Source: indra-master/indra/ontology/ontology_graph.py L281-L302
        """
        source = (ns, id)
        visited = {source}
        queue = deque([(source,
                        rel_fun(*source, rel_types))])
        while queue:
            parent, children = queue[0]
            try:
                child = next(children)
                if target and child == target:
                    return [target]
                if child not in visited:
                    visited.add(child)
                    queue.append((child,
                                  rel_fun(*child, rel_types)))
            except networkx.NetworkXError as e:
                logger.debug(e)
                return []
            except StopIteration:
                queue.popleft()
        return list(visited - {source})

    @with_initialize
    def descendants_rel(self, ns, id, rel_types):
        """Return all descendants reachable via the given edge types.

        Source: indra-master/indra/ontology/ontology_graph.py L304-L306
        """
        return self._transitive_rel(ns, id, self.child_rel, rel_types)

    @with_initialize
    def ancestors_rel(self, ns, id, rel_types):
        """Return all ancestors reachable via the given edge types.

        Source: indra-master/indra/ontology/ontology_graph.py L308-L310
        """
        return self._transitive_rel(ns, id, self.parent_rel, rel_types)

    @with_initialize
    def child_rel(self, ns, id, rel_types):
        """Yield (ns, id) pairs of successors connected by the given edge types.

        Source: indra-master/indra/ontology/ontology_graph.py L312-L323
        """
        source = self.label(ns, id)
        try:
            succ_iter = self.successors(source)
        except networkx.NetworkXError:
            return []
        for target in succ_iter:
            if self.edges[source, target]['type'] in rel_types:
                yield self.get_ns_id(target)

    @with_initialize
    def parent_rel(self, ns, id, rel_types):
        """Yield (ns, id) pairs of predecessors connected by the given edge types.

        Source: indra-master/indra/ontology/ontology_graph.py L325-L336
        """
        target = self.label(ns, id)
        try:
            pred_iter = self.predecessors(target)
        except networkx.NetworkXError:
            return []
        for source in pred_iter:
            if self.edges[source, target]['type'] in rel_types:
                yield self.get_ns_id(source)

    # ------------------------------------------------------------------
    # Ontological relationship accessors
    # ------------------------------------------------------------------

    @with_initialize
    def get_children(self, ns, id, ns_filter=None):
        """Return all 'isa' or 'partof' children of a given entity.

        Importantly, 'isa' and 'partof' edges always point towards
        higher-level entities in the ontology but here "child" means
        lower-level entity i.e., ancestors in the graph.

        Parameters
        ----------
        ns : str
            The name space of an entity.
        id : str
            The ID of an entity.
        ns_filter : Optional[set]
            If provided, only entities within the set of given
            name spaces are returned.

        Returns
        -------
        list
            A list of entities (name space, ID pairs) that are the
            children of the given entity.

        Source: indra-master/indra/ontology/ontology_graph.py L338-L365
        """
        children = self.ancestors_rel(ns, id, {'isa', 'partof'})
        children = [(cns, cid) for cns, cid in children
                    if ns_filter is None or cns in ns_filter]
        return children

    @with_initialize
    def get_parents(self, ns, id):
        """Return all 'isa' or 'partof' parents of a given entity.

        Importantly, 'isa' and 'partof' edges always point towards
        higher-level entities in the ontology but here "parent" means
        higher-level entity i.e., descendants in the graph.

        Parameters
        ----------
        ns : str
            The name space of an entity.
        id : str
            The ID of an entity.

        Returns
        -------
        list
            A list of entities (name space, ID pairs) that are the
            parents of the given entity.

        Source: indra-master/indra/ontology/ontology_graph.py L367-L388
        """
        return self.descendants_rel(ns, id, {'isa', 'partof'})

    @with_initialize
    def get_top_level_parents(self, ns, id):
        """Return all top-level 'isa' or 'partof' parents of a given entity.

        Top level means that this function only returns parents which
        don't have any further 'isa' or 'partof' parents above them.

        Parameters
        ----------
        ns : str
            The name space of an entity.
        id : str
            The ID of an entity.

        Returns
        -------
        list
            A list of entities (name space, ID pairs) that are the
            top-level parents of the given entity.

        Source: indra-master/indra/ontology/ontology_graph.py L390-L414
        """
        parents = self.get_parents(ns, id)
        return [p for p in parents if not self.get_parents(*p)]

    @with_initialize
    def get_mappings(self, ns, id):
        """Return entities that are xrefs of a given entity.

        Parameters
        ----------
        ns : str
            An entity's name space.
        id : str
            An entity's ID.

        Returns
        -------
        list
            A list of entities (name space, ID pairs) that are
            direct or indirect xrefs of the given entity.

        Source: indra-master/indra/ontology/ontology_graph.py L416-L436
        """
        return self.descendants_rel(ns, id, {'xref'})

    @with_initialize
    def get_replacement(self, ns, id):
        """Return a replacement for a given entity or None if no replacement.

        Parameters
        ----------
        ns : str
            An entity's name space.
        id : str
            An entity's ID.

        Returns
        -------
        tuple or None
            A tuple of the form (ns, id) of the replacement entity
            or None if no replacement.

        Source: indra-master/indra/ontology/ontology_graph.py L438-L460
        """
        rep = list(self.child_rel(ns, id, {'replaced_by'}))
        if rep:
            return rep[0]

    # ------------------------------------------------------------------
    # Node property accessors
    # ------------------------------------------------------------------

    @with_initialize
    def get_name(self, ns, id):
        """Return the standard name of a given entity.

        Parameters
        ----------
        ns : str
            An entity's name space.
        id : str
            An entity's ID.

        Returns
        -------
        str or None
            The name associated with the given entity or None
            if the node is not in the ontology or doesn't have a standard name.

        Source: indra-master/indra/ontology/ontology_graph.py L462-L480
        """
        return self.get_node_property(ns, id, property='name')

    @with_initialize
    def get_type(self, ns, id):
        """Return the type of a given entity.

        Parameters
        ----------
        ns : str
            An entity's name space.
        id : str
            An entity's ID.

        Returns
        -------
        str or None
            The type associated with the given entity or None
            if the node is not in the ontology or doesn't have a type.

        Source: indra-master/indra/ontology/ontology_graph.py L482-L500
        """
        return self.get_node_property(ns, id, 'type')

    @with_initialize
    def get_polarity(self, ns, id):
        """Return the polarity of a given entity.

        Parameters
        ----------
        ns : str
            An entity's name space.
        id : str
            An entity's ID.

        Returns
        -------
        str or None
            The polarity associated with the given entity or None
            if the node is not in the ontology or doesn't have a polarity.

        Source: indra-master/indra/ontology/ontology_graph.py L502-L520
        """
        return self.get_node_property(ns, id, property='polarity')

    @with_initialize
    def get_node_property(self, ns, id, property):
        """Return a given property of a given entity.

        Parameters
        ----------
        ns : str
            An entity's name space.
        id : str
            An entity's ID.
        property : str
            The property to look for on the given node.

        Returns
        -------
        str or None
            The value of the property or None if the node or property
            is not found.

        Source: indra-master/indra/ontology/ontology_graph.py L522-L545
        """
        try:
            return self.nodes[self.label(ns, id)][property]
        except KeyError:
            return None

    @with_initialize
    def is_opposite(self, ns1, id1, ns2, id2):
        """Return True if the two entities are opposites of each other.

        Parameters
        ----------
        ns1 : str
            The first entity's name space.
        id1 : str
            The first entity's ID.
        ns2 : str
            The second entity's name space.
        id2 : str
            The second entity's ID.

        Returns
        -------
        bool
            True if the first entity is in an 'is_opposite'
            relationship with the second. False otherwise.

        Source: indra-master/indra/ontology/ontology_graph.py L547-L572
        """
        # FIXME: this assumes, as is the case in practice with our
        # ontologies that we have disjunct pairs of is_opposite entities
        # more generally, we may need to allow other edge types and
        # look at the overall "polarity" of the path.
        return self._check_path(ns1, id1, ns2, id2, {'is_opposite'})

    # ------------------------------------------------------------------
    # Name lookup
    # ------------------------------------------------------------------

    @with_initialize
    def get_id_from_name(self, ns, name) -> Optional[Tuple[str, str]]:
        """Return an entity's ID given its name space and standard name.

        Parameters
        ----------
        ns : str
            The name space in which the standard name is defined.
        name : str
            The standard name defined in the name space.

        Returns
        -------
        tuple or None
            The pair of namespace and ID corresponding to the given
            standard name in the given name space or None if not available.

        Source: indra-master/indra/ontology/ontology_graph.py L574-L594
        """
        if not self.name_to_grounding:
            self._build_name_lookup()
        return self.name_to_grounding.get((ns, name))

    @with_initialize
    def _build_name_lookup(self):
        """Build the name_to_grounding reverse index.

        Source: indra-master/indra/ontology/ontology_graph.py L596-L603
        """
        self.name_to_grounding = {
            (self.get_ns(node), data['name']): self.get_ns_id(node)
            for node, data in self.nodes(data=True)
            if 'name' in data
            and not data.get('obsolete', False)
        }

    @with_initialize
    def nodes_from_suffix(self, suffix):
        """Return all node labels which have a given suffix.

        This is useful for finding entities in ontologies where
        the IDs consist of paths like a/b/c/...

        Parameters
        ----------
        suffix : str
            A label suffix.

        Returns
        -------
        list
            A list of node labels that have the given suffix.

        Source: indra-master/indra/ontology/ontology_graph.py L605-L623
        """
        return [node for node in self.nodes
                if node.endswith(suffix)]

    # ------------------------------------------------------------------
    # Label encoding / decoding
    # ------------------------------------------------------------------

    @staticmethod
    def label(ns, id):
        """Return the label corresponding to a given entity.

        Parameters
        ----------
        ns : str
            An entity's name space.
        id : str
            An entity's ID.

        Returns
        -------
        str
            The label corresponding to the given entity.

        Source: indra-master/indra/ontology/ontology_graph.py L625-L646
        """
        return '%s:%s' % (ns, id)

    @staticmethod
    def reverse_label(label):
        """Return the name space and ID from a given label.

        Parameters
        ----------
        label : str
            A node label.

        Returns
        -------
        str
            The name space corresponding to the label.
        str
            The ID corresponding to the label.

        Source: indra-master/indra/ontology/ontology_graph.py L648-L667
        """
        return tuple(label.split(':', maxsplit=1))

    # ------------------------------------------------------------------
    # Transitive closure
    # ------------------------------------------------------------------

    def _build_transitive_closure(self):
        """Pre-compute all (child, parent) pairs for O(1) isa_or_partof checks.

        Source: indra-master/indra/ontology/ontology_graph.py L669-L681
        """
        if self.transitive_closure:
            return
        logger.info('Building transitive closure for faster '
                    'isa/partof lookups...')
        self.transitive_closure = set()
        for node in self.nodes():
            ns, id = self.get_ns_id(node)
            for pns, pid in self.descendants_rel(ns, id,
                                                 rel_types={'isa',
                                                            'partof'}):
                self.transitive_closure.add((self.label(ns, id),
                                             self.label(pns, pid)))

    @with_initialize
    def print_stats(self):
        """Log node and edge counts.

        Source: indra-master/indra/ontology/ontology_graph.py L683-L687
        """
        logger.info('Number of nodes: %d' % len(self.nodes))
        logger.info('Number of edges: %d' % len(self.edges))
