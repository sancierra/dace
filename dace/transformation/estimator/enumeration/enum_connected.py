""" This file implements the ConnectedEnumerator class """

from dace.transformation.estimator.enumeration import Enumerator

from dace.transformation.subgraph import SubgraphFusion, helpers
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

import dace.sdfg.nodes as nodes

from collections import deque, defaultdict, ChainMap
from typing import Set, Union, List, Callable
import itertools


@make_properties
class ConnectedEnumerator(Enumerator):
    '''
    Enumerates all subgraphs that are connected through Access Nodes
    '''

    local_maxima = Property(desc="List local maxima while enumerating",
                            default=False,
                            dtype=bool)

    prune = Property(desc="Perform Greedy Pruning during Enumeration",
                     default=True,
                     dtype=bool)

    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 subgraph: SubgraphView = None,
                 condition_function: Callable = None,
                 scoring_function=None,
                 **kwargs):

        # initialize base class
        super().__init__(sdfg, graph, subgraph, condition_function,
                         scoring_function)
        self._local_maxima = []
        self._function_args = kwargs

        # create adjacency list (improved version)
        # connect everything that shares an edge (any direction)
        # to an access node
        self._adjacency_list = {m: set() for m in self._map_entries}
        # helper dict needed for a quick build
        exit_nodes = {graph.exit_node(me): me for me in self._map_entries}
        for node in (subgraph.nodes() if subgraph else graph.nodes()):
            if isinstance(node, nodes.AccessNode):
                adjacent_entries = set()
                for e in graph.in_edges(node):
                    if isinstance(e.src, nodes.MapExit) and e.src in exit_nodes:
                        adjacent_entries.add(exit_nodes[e.src])
                for e in graph.out_edges(node):
                    if isinstance(
                            e.dst,
                            nodes.MapEntry) and e.dst in self._map_entries:
                        adjacent_entries.add(e.dst)
                # now add everything to everything
                for entry in adjacent_entries:
                    for other_entry in adjacent_entries:
                        if entry != other_entry:
                            self._adjacency_list[entry].add(other_entry)
                            self._adjacency_list[other_entry].add(entry)

    def traverse(self, current: List, forbidden: Set, prune=False):
        if len(current) > 0:
            # get current subgraph we are inspecting
            #print("*******")
            #print(current)
            current_subgraph = helpers.subgraph_from_maps(
                self._sdfg, self._graph, current, self._scope_dict)
            #print(current_subgraph.nodes())

            # evaluate condition if specified
            conditional_eval = True
            if self._condition_function:
                conditional_eval = self._condition_function(
                    self._sdfg, current_subgraph)
                #print("EVALUATED TO", conditional_eval)
            # evaluate score if possible
            score = 0
            if conditional_eval and self._scoring_function:
                score = self._scoring_function(current_subgraph)

            # calculate where to backtrack next if not prune
            go_next = set()
            if conditional_eval or not self._prune or len(current) == 1:
                go_next = set(m for c in current
                              for m in self._adjacency_list[c]
                              if m not in current and m not in forbidden)
                if self.debug:
                    go_next = list(go_next)
                    go_next.sort(key=lambda e: e.map.label)
            # yield element if condition is True
            if conditional_eval:
                self._histogram[len(current)] += 1
                yield (current.copy(),
                       score) if self.mode == 'map_entries' else (
                           current_subgraph, score)

        else:
            # special case at very beginning: explore every node
            go_next = set(m for m in self._adjacency_list.keys())
            if self.debug:
                go_next = list(go_next)
                go_next.sort(key=lambda e: e.map.label)
        if len(go_next) > 0:
            # we can explore
            forbidden_current = set()
            for child in go_next:
                current.append(child)
                yield from self.traverse(current, forbidden | forbidden_current,
                                         prune)
                pp = current.pop()
                forbidden_current.add(child)

        else:
            # we cannot explore - possible local maximum candidate
            # TODO continue work
            self._local_maxima.append(current.copy())

    def iterator(self):
        '''
        returns an iterator that iterates over
        search space yielding tuples (subgraph, score)
        '''
        self._local_maxima = []
        self._histogram = defaultdict(int)
        yield from self.traverse([], set(), False)
