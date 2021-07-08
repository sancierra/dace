# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This file implements the Scoring Function class """

from dace.transformation.subgraph import helpers
from dace.transformation.subgraph.composite import CompositeFusion
from dace.properties import make_properties
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

import dace.dtypes as dtypes

from typing import Type, Dict, Tuple


@make_properties
class ScoringFunction:
    '''
    Abstract base class that accepts an SDFG and state of interest,
    in which arbitrary subgraphs can be scored. Scoring 
    occurs after a dataflow transformation function is applied to the 
    subgraph of interest
    Due to non-reversibility of transformations, scoring 
    transformation application does not occur in-place.
    '''
    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 device: dtypes.DeviceType = dtypes.DeviceType.CPU,
                 transformation_function: Type = CompositeFusion,
                 transformation_properties: Dict = None,
                 io: Tuple[Dict] = None):
        ''' 
        :param sdfg: SDFG of interest 
        :param graph: Graph of interest
        :param device: Device target 
        :param transformation_function: Transformation function to be instantiated
                                        and applied to a subgraph before scoring
        :param transformation_properties: Dictionary of transformation properties 
                                          to set for the transformation function 
        :param io: Tuple of (input_dict, output_dict, symbols_dict), where each of 
                   these dictionaries contains a mapping from argument/symbol name 
                   to its corresponding argument. If None, IO arguments are not
                   required.
        '''
        # set sdfg-related variables
        self._sdfg = sdfg
        self._sdfg_id = sdfg.sdfg_id
        self._graph = graph
        self._state_id = sdfg.nodes().index(graph)
        self._scope_dict = graph.scope_dict() 
        self._transformation = transformation_function
        self._transformation_properties = transformation_properties
        self._device = device 

        # search for outermost map entries
        self._map_entries = helpers.get_outermost_scope_maps(
            sdfg, graph, scope_dict = self._scope_dict)

        # set IO if defined
        self._inputs, self._outputs, self._symbols = io if io else None, None, None


    def score(self, subgraph: SubgraphView):
        raise NotImplementedError

    def __call__(self, subgraph: SubgraphView):
        return self.score(subgraph)

    @staticmethod
    def name():
        return NotImplementedError
    
    def label():
        return NotImplementedError
