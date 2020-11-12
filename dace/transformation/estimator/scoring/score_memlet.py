""" This file implements the ExecutionScore class """

from dace.transformation.subgraph import SubgraphFusion, StencilTiling, helpers
from dace.transformation.subgraph.composite import CompositeFusion
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

from dace.perf.movement_counter import count_moved_data_state
from dace.perf.movement_counter import count_moved_data_subgraph


from typing import Set, Union, List, Callable, Dict, Type

from dace.transformation.estimator.scoring import ScoringFunction

import json
import warnings
import os
import numpy as np
import sys


@make_properties
class MemletScore(ScoringFunction):
    '''
    Evaluates Subgraphs by just running their
    SDFG and returning the runtime
    '''
    debug = Property(desc = "Debug Mode",
                     dtype = bool,
                     default = True)

    exit_on_error = Property(desc = "Exit program if error occurs, else return -1",
                             dtype = bool,
                             default = False)
    view_on_error = Property(desc = "View program if faulty",
                             dtype = bool,
                             default = False)
    save_on_error = Property(desc = "Save SDFG if faulty",
                             dtype = bool,
                             default = True)
    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 symbols: Dict,
                 subgraph: SubgraphView = None,
                 gpu: bool = None,
                 transformation_function: Type = CompositeFusion,
                 **kwargs):
        super().__init__(sdfg=sdfg,
                         graph=graph,
                         subgraph=subgraph,
                         gpu=gpu,
                         transformation_function=transformation_function,
                         **kwargs)

        self._symbols = symbols
        self._base_traffic = self.estimate_traffic(sdfg, graph)

    def estimate_traffic(self, sdfg, graph):
        try:
            traffic = count_moved_data_state(sdfg, graph, self._symbols)
        except Exception as e:
            print("ERROR in score_memlet")
            print(e)
            traffic = 0

        return traffic

    def score(self, subgraph: SubgraphView):
        '''
        Applies CompositeFusion to the Graph and compares Memlet Volumes
        with the untransformed SDFG passed to __init__().
        '''

        sdfg_copy = SDFG.from_json(self._sdfg.to_json())
        graph_copy = sdfg_copy.nodes()[self._state_id]
        subgraph_copy = SubgraphView(graph_copy, [
            graph_copy.nodes()[self._graph.nodes().index(n)] for n in subgraph
        ])

        if self.debug:
            print("ScoreMemlet::Debug::Subgraph to Score:",
                   subgraph_copy.nodes())

        transformation_function = self._transformation(subgraph_copy)
        # assign properties to transformation
        for (arg, val) in self._kwargs.items():
            try:
                setattr(transformation_function, arg, val)
            except AttributeError:
                warnigns.warn(f"Attribute {arg} not found in transformation")
            except (TypeError, ValueError):
                warnings.warn(f"Attribute {arg} has invalid value {val}")

        transformation_function.apply(sdfg_copy)
        current_traffic = self.estimate_traffic(sdfg_copy, graph_copy)
        return current_traffic / self._base_traffic
