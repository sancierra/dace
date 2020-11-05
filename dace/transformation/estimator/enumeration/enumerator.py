""" This file implements the Enuerator class """

from dace.transformation.subgraph import SubgraphFusion, helpers
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

from dace.transformation.estimator import ScoringFunction
import dace.sdfg.nodes as nodes

from collections import deque, defaultdict, ChainMap
from typing import Set, Union, List, Callable
import itertools


@make_properties
class Enumerator:
    '''
    Base Enumerator Class
    '''
    mode = Property(desc="Data type the Iterator should return. "
                    "Choice between Subgraph and List of Map Entries.",
                    default="map_entries",
                    choices=["subgraph", "map_entries"],
                    dtype=str)

    debug = Property(desc="Debug mode", default=True, dtype=bool)

    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 subgraph: SubgraphView = None,
                 condition_function: Callable = None,
                 scoring_function: ScoringFunction = None):

        self._sdfg = sdfg
        self._graph = graph
        self._scope_dict = graph.scope_children()
        self._condition_function = condition_function
        self._scoring_function = scoring_function

        # get hightest scope maps
        self._map_entries = helpers.get_outermost_scope_maps(
            sdfg, graph, subgraph)
        self._max_length = len(self._map_entries)

        if self._condition_function is None and self._scoring_function is not None:
            warnings.warn('Initialized with no condition function but scoring'
                          'function. Will try to score all subgraphs!')
        # for memorization purposes
        self._histogram = None

    def iterator(self):
        '''
        iterator interface to implement
        '''
        # Interface to implement
        raise NotImplementedError

    def list(self, include_score=True):
        if include_score:
            return list(e for e in self.iterator())
        else:
            return list(e[0] for e in self.iterator())

    def __iter__(self):
        yield from self.iterator()

    def histogram(self, visual=True, cached=True):
        if self._histogram is None and cached:
            old_mode = self.mode
            self.mode = 'map_entries'
            lst = self.list(include_score=False)
            self._histogram = {}
            for i in range(1, 1 + self._max_length):
                no_elements = sum([len(sg) == i for sg in lst])
                self._histogram[i] = no_elements
            self.mode = old_mode

        if visual:
            print("*** Subgraph Statistics ***")
            for (i, no_elements) in sorted(self._histogram.items(),
                                           key=lambda a: a[0]):
                print(i, no_elements, "*" * no_elements)

        return self._histogram
