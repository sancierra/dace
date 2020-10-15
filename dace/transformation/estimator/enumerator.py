""" This file implements the Enuerator class """

from dace.transformation.subgraph import SubgraphFusion, helpers
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

from scoring_function import ScoringFunction, ExecutionScore

import dace.sdfg.nodes as nodes

from collections import deque, defaultdict, ChainMap
from typing import Set, Union, List, Callable
import itertools


@make_properties
class Enumerator:
    '''
    Base Enumerator Class
    '''
    mode = Property(desc = "Data type the Iterator should return. "
                           "Choice between Subgraph and List of Map Entries.",
                    default = "map_entries",
                    choices = ["subgraph", "map_entries"],
                    dtype = str)

    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 subgraph: SubgraphView = None,
                 condition: Callable = None,
                 scoring_function: ScoringFunction = None):

        self._sdfg = sdfg
        self._graph = graph
        self._scope_dict = graph.scope_dict(node_to_children = True)
        self._condition = condition
        self._scoring_function = scoring_function


    def iterator(self):
        # Interface to implement
        raise NotImplementedError

    def list(self):
        return list(e[0] for e in self.iterator())

    def __iter__(self):
        yield from self.iterator()


@make_properties
class ConnectedEnumerator(Enumerator):
    '''
    Enumerates all subgraphs that are connected through Access Nodes
    '''

    local_maxima = Property(desc = "List local maxima while enumerating",
                     default = False,
                     dtype = bool)

    prune = Property(desc = "Perform Greedy Pruning during Enumeration",
                     default = True,
                     dtype = bool)

    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 subgraph: SubgraphView = None,
                 condition: Callable = None,
                 scoring_function = None):

        # initialize base class
        super().__init__(sdfg, graph, subgraph, condition, scoring_function)
        self._local_maxima = []

        # get hightest scope maps
        map_entries = helpers.get_highest_scope_maps(sdfg, graph, subgraph)

        # create adjacency list
        self._adjacency_list = {m: set() for m in map_entries}
        for map_entry in map_entries:
            map_exit = graph.exit_node(map_entry)
            for edge in graph.out_edges(map_exit):
                current_node = edge.dst
                if not isinstance(current_node, nodes.AccessNode):
                    continue
                for dst_edge in graph.out_edges(current_node):
                    if dst_edge.dst in map_entries:
                        self._adjacency_list[map_entry].add(dst_edge.dst)
                        self._adjacency_list[dst_edge.dst].add(map_entry)

        # create adjacency list
        # connect everything that is not

    def traverse(self, current: List, forbidden: Set, prune = False):
        if len(current) > 0:
            # get current subgraph we are inspecting
            current_subgraph = helpers.subgraph_from_maps(self._sdfg, self._graph, current, self._scope_dict)

            # yield element if possible, and explore neighboring sets to go to
            if self._condition and self._prune and \
                        not self.contition(sdfg, graph, current_subgraph):
                go_next = []
            else:
                score = self._scoring_function(current_subgraph) if self._scoring_function else 0
                if self.mode == 'map_entries':
                    current_entries = current.copy()
                yield (current_entries, score) if self.mode == 'map_entries' else (current_subgraph, score)
                # calculate where to backtrack next
                go_next = set(m for c in current for m in self._adjacency_list[c] if m not in current and m not in forbidden)
        else:
            # special case at very beginning: explore every node
            go_next = set(m for m in self._adjacency_list.keys())
        if len(go_next) > 0:
            # we can explore
            forbidden_current = set()
            for child in go_next:
                current.append(child)
                yield from self.traverse(current, forbidden | forbidden_current, prune)
                pp = current.pop()
                forbidden_current.add(child)

        else:
            # we cannot explore - possible local maximum candidate
            # TODO if self.local_maxima and if not any(....)
            self._local_maxima.append(current.copy())


    def iterator(self):
        '''
        returns an iterator that iterates over
        search space yielding tuples (subgraph, score)
        '''
        self._local_maxima = []
        yield from self.traverse([], set(), False)

    def histogram(self, visual = True):
        old_mode = self.mode
        self.mode = 'map_entries'
        lst = self.list()
        print("Subgraph Statistics")
        print("-------------------")
        for i in range(1, 1 + len(self._adjacency_list)):
            no_elements = sum([len(sg) == i for sg in lst])
            if visual:
                print(i, no_elements, "*" * no_elements)
            else:
                print("Subgraphs with", i, "elements:", no_elements)
        self.mode = old_mode


@make_properties
class BruteForceEnumerator(Enumerator):
    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 subgraph: SubgraphView = None,
                 condition: Callable = None,
                 scoring_function = None):
        # initialize base class
        super().__init__(sdfg, graph,
                         subgraph = subgraph,
                         condition = condition,
                         scoring_function = scoring_function)

        # get hightest scope maps
        self._map_entries = helpers.get_highest_scope_maps(sdfg, graph, subgraph)

    def brute_force(self):
        for i in range(1, len(self._map_entries)):
            for sg in itertools.combinations(self._map_entries, i):
                # check whether either
                # (1) no path between all maps
                # (2) if path, then only AccessNode
                # Topo BFS the whole graph is the most efficient (ignoring the outer loops above...)
                subgraph = helpers.subgraph_from_maps(self._sdfg, self._graph, sg, self._scope_dict)
                if SubgraphFusion.can_be_applied(self._sdfg, subgraph):
                    yield subgraph


    def iterator(self):
        yield from self.brute_force()
