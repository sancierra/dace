""" This file implements the GreedyEnumerator class """

from dace.transformation.estimator.enumeration import Enumerator

from dace.transformation.subgraph import SubgraphFusion, CompositeFusion, helpers
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

import dace.sdfg.nodes as nodes

from collections import deque, defaultdict, ChainMap
from typing import Set, Union, List, Callable
import itertools

import heapq



class QueuedEntry:
    def __init__(self, map_entry, index):
        self.map_entry = map_entry 
        self.index = index 
    
    def __lt__(self, other):
        return self.index < other.index 


@make_properties
class GreedyEnumerator(Enumerator):
    '''
    Enumerates all maximally fusible subgraphs, 
    each of the corresponding map sets from an 
    iteration being disjoint
    '''

    def __init__(self, 
                 sdfg: SDFG,
                 graph: SDFGState,
                 subgraph: SubgraphView = None,
                 condition_function: Callable = CompositeFusion.can_be_applied):
    

        super().__init__(sdfg, graph, subgraph, condition_function)
        
        # create a neighbors mapping: adjacency_list[m] = set of all the *neighbors* of m
        # neighbor = another map succeeding / preceeding with just an access node in between
        self._adjacency_list = {m: set() for m in self._map_entries}
        # also create a directed DAG neighbors mapping 
        children_list = {m: set() for m in self._map_entries}

        # helper dict needed for a quick build
        exit_nodes = {graph.exit_node(me): me for me in self._map_entries}
        if subgraph:
            proximity_in = set(ie.src for ie in graph.in_edges(me) for me in self._map_entries)
            proximity_out = set(ie.dst for ie in graph.out_edges(me) for me in exit_nodes)
            extended_subgraph = SubgraphView(graph, set(itertools.chain(subgraph.nodes(), proximity_in, proximity_out)))

        
        for node in (extended_subgraph.nodes() if subgraph else graph.nodes()):
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

                # bidirectional mapping
                for entry in adjacent_entries:
                    for other_entry in adjacent_entries:
                        if entry != other_entry:
                            self._adjacency_list[entry].add(other_entry)
                            self._adjacency_list[other_entry].add(entry)
        
        

        # get DAG children and parents 
        children_dict = defaultdict(set)
        parent_dict = defaultdict(set)

        for map_entry in map_entries:
            map_exit = graph.exit_node(map_entry)
            for e in graph.out_edges(map_exit):
                if isinstance(e.dst, nodes.AccessNode):
                    for oe in graph.out_edges(e.dst):
                        if oe.dst in map_entries:
                            other_entry = oe.dst
                            children_dict[map_entry].add(other_entry)
                            parent_dict[other_entry].add(map_entry)
   

        # find out source nodes 
        self._source_maps = [me for me in self._map_entries if len(parent_dict[me]) == 0]
        # assign a unique id to each map entry according to topological
        # ordering. If on same level, sort according to ID for determinism

        self._labels = {} # map -> ID 
        current_id = 0
        while current_id < len(self._map_entries):
            # get current ids whose in_degree is 0 
            candidates = list(me for (me, s) in parent_dict.items() if len(s) == 0 and me not in self._labels)
            candidates.sort(key = lambda me: me.id)
            for c in candidates:
                self._labels[c] = current_id 
                current_id += 1
                # remove candidate for each players adjacency list 
                for c_child in children_dict[c]:
                    parent_dict[c_child].remove(s)
            
        
        
    def iterator(self):
        # iterate through adjacency list starting with map with lowest label.
        # then greedily explore neighbors with next lowest label and see whether set is fusible 
        # if not fusible, cancel and create a new set 

        first_map = next(me for me in self._adjacency_list if self._labels[me] == 0)

        # define queue / visited set which helps us find starting points
        # for the next inner iterations 
        added = set() 
        outer_queued = set(self._source_maps)
        outer_queue = [QueuedEntry(me, self._labels[me]) for me in self._source_maps]
        while len(outer_queue) > 0
            
            # current iteration: define queue / set with which we are going 
            # to find current components 
            
            next_iterate = heapq.heappop(outer_queue)
            while next_iterate in visited:
                next_iterate = heapq.heappop(outer_queue)

            current_set = set(next_iterate.map_entry)
            inner_queue = [next_iterate]

            while len(inner_queue) > 0:
                # select starting map 
                current = heapq.heappop(inner_queue)
                current_map = current.map_entry 

                # check current + current set can be fused 
                add_current_map = False 
                if len(current_set) == 0:
                    add_current_map = True 
                else:
                    subgraph = helpers.subgraph_from_maps(self._sdfg, self._graph, current_set | current_map)
                    if self._condition_function(self._sdfg, self._graph, subgraph):
                        add_current_map = True 
                    
                    
                if add_current_map:
                    # add it to current set and continue BFS 
                    added.add(current_map)
                    # recurse further
                    for current_neighbor_map in self._adjacency_list[current_map]:
                        # add to outer queue and set 
                        if current_neighbor_map not in outer_queued:
                            heapq.heappush(outer_queue, QueuedEntry(current_neighbor_map, self._labels[current_neighbor_map])) 
                            outer_queued.add(current_neighbor_map)
                        # add to inner queue and set 
                        if current_neighbor_map not in inner_queued:
                            heapq.heappush(inner_queue, QueuedEntry(current_neighbor_map, self._labels[current_neighbor_map]))
                            inner_queued.add(current_neighbor_map)

            # yield 
            if self.mode == 'map_entries':
                yield tuple(map_entries)
            else:
                yield helpers.subgraph_from_maps(self._sdfg, self._graph, map_entries)
