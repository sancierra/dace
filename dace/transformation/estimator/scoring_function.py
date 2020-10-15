""" This file implements the Scoring Function class """

from dace.transformation.subgraph import SubgraphFusion, helpers
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState,
from dace.sdfg.graph import SubgraphView

import dace.sdfg.nodes as nodes
import dace.dtypes as dtypes

from collections import deque, defaultdict, ChainMap
from typing import Set, Union, List


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
                 ** kwargs):

        self._sdfg = sdfg
        self._graph = graph
        self._subgraph = subgraph

    def score(subgraph: SubgraphView):
        # NOTE: self._subgraph and subgraph are not the same!
        raise NotImplementedError

@mmake_properties
class ExecutionScore(ScoringFunction):
    '''
    Evaluates Subgraphs by just running their
    SDFG and returning the runtime
    '''
    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 subgraph: SubgraphView,
                 gpu: bool = None,
                 nruns = 30
                 ** kwargs):
        super().__init__(sdfg, graph, subgraph, kwargs)

        if gpu is None:
            # detect whether the state is assigned to GPU
            map_entries = heleprs.get_highest_scope_maps(sdfg, graph, subgraph)
            schedule = next(map_entries).schedule
            if any([m.schedule != schedule for m in map_entries]):
                raise RuntimeError("Schedules in maps to analyze should be the same")
            self._gpu = True if schedule in [dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_ThreadBlock] else False
        else:
            self._gpu = gpu

        self._sdfg_id = sdfg.sdfg_id
        self._state_id = sdfg.nodes().index(graph)
        # TODO: modify nruns
        self.nruns = 30


    def score(subgraph: SubgraphView, **kwargs):
        '''
        scores a subgraph (within the given graph / subgraph
        that was passed to the initializer) by running it
        and returning the runtime
        '''
        # generate an instance of SubgraphFusion
        # deepcopy the subgraph via json and apply transformation
        sdfg_copy = SDFG.from_json(self._sdfg.to_json)
        subgraph_copy = set(sdfg_copy.nodes(self._state_id)[self.graph.index(n)] for n in subgraph)
        fusion = SubgraphFusion(subgraph_copy, self._sdfg_id, self.state_id)
        fusion.apply(sdfg_copy)
        # mark graph with instrumentation
        sdfg_copy.instrument = dtypes.InstrumentationType.Timer
        # TODO
        ###csdfg = sdfg_copy.compile()
        ###csdfg(kwargs)
        # get timing results
        # TODO
        return 42.0
