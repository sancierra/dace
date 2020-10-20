""" This file implements the Scoring Function class """

from dace.transformation.subgraph import SubgraphFusion, helpers
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

import dace.sdfg.nodes as nodes
import dace.dtypes as dtypes

from collections import deque, defaultdict, ChainMap
from typing import Set, Union, List, Callable

import json

@make_properties
class ScoringFunction:
    '''
    Class used to Score Subgraphs in order to
    rank them for their fusion applicability
    '''
    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 subgraph: SubgraphView = None,
                 scope_dict = None):

        self._sdfg = sdfg
        self._graph = graph
        self._subgraph = subgraph
        self._scope_dict = scope_dict

    def score(self, subgraph: SubgraphView, ** kwargs):
        # NOTE: self._subgraph and subgraph are not the same!
        raise NotImplementedError

    def __call__(self, subgraph: SubgraphView, ** kwargs):
        return self.score(subgraph, **kwargs)
