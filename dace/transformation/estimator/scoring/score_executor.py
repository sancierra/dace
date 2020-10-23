""" This file implements the ExecutionScore class """

from dace.transformation.subgraph import SubgraphFusion, helpers
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

import dace.sdfg.nodes as nodes
import dace.dtypes as dtypes

from collections import deque, defaultdict, ChainMap
from typing import Set, Union, List, Callable, Dict

from dace.transformation.estimator.scoring import ScoringFunction

import json
import warnings
import os
import numpy as np

@make_properties
class ExecutionScore(ScoringFunction):
    '''
    Evaluates Subgraphs by just running their
    SDFG and returning the runtime
    '''
    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 inputs: Dict,
                 outputs: Dict,
                 symbols: Dict,
                 subgraph: SubgraphView = None,
                 gpu: bool = None,
                 nruns = None,
                 ):
        super().__init__(sdfg, graph, subgraph)

        map_entries = helpers.get_outermost_scope_maps(sdfg, graph, subgraph)
        if gpu is None:
            # detect whether the state is assigned to GPU
            schedule = next(iter(map_entries)).schedule
            if any([m.schedule != schedule for m in map_entries]):
                raise RuntimeError("Schedules in maps to analyze should be the same")
            self._gpu = True if schedule in [dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_ThreadBlock] else False
        else:
            self._gpu = gpu

        self._sdfg_id = sdfg.sdfg_id
        self._state_id = sdfg.nodes().index(graph)
        self._map_entries = map_entries

        # input arguments: we just create a class variable
        self._inputs = inputs
        self._outputs = outputs
        self._symbols = symbols

        # run the graph to create a baseline
        self._median_rt_base = self.run_with_instrumentation(self._sdfg, self._graph, self._map_entries, check = False)
        self._sdfg.view()

        # if nruns is defined, change config
        if nruns is not None:
            dace.config.Config.set('treps', value=nruns)

    def run_with_instrumentation(self, sdfg, graph, map_entries = None, check = True):

        '''
        runs an sdfg with instrumentation on all outermost scope
        maps and returns their runtimes
        '''
        if map_entries is None:
            map_entries = helpers.get_outermost_scope_maps(sdfg, graph)

        # instrumentation:
        # mark all mapentries  with instrumentation
        for map_entry in map_entries:
            # GPU_Events TODO
            map_entry.map.instrument = dtypes.InstrumentationType.Timer

        # run and go
        # create a copy of all the outputs
        outputs_local = {ok: ov.copy() for (ok, ov) in self._outputs.items()}
        sdfg(**self._inputs, **self._outputs, **self._symbols)

        # assert outputs are the same
        if check:
            nv = True
            for (ok, ov) in self._outputs.items():
                if not np.allclose(outputs_local[ok], ov):
                    warnings.warn('Wrong output!')
                    if nv:
                        sdfg.view()
                        nv = False
                    print('Original Output:',np.linalg.norm(ov))
                    print('Transformed Ouput:', np.linalg.norm(outputs_local[ok]))
                else:
                    print("PASS! YES")

        # remove old maps instrumentation
        for map_entry in map_entries:
            map_entry.map.instrument = dtypes.InstrumentationType.No_Instrumentation

        # get timing results
        path = os.path.join(sdfg.build_folder, 'perf')
        files = [f for f in os.listdir(path) if f.startswith('report-')]
        assert len(files) > 1
        # Avoid import loops
        json_file = sorted(files, reverse = True)[0]
        runtime = 0
        with open(os.path.join(sdfg.build_folder, 'perf', json_file)) as f:
            data = json.load(f)
            for _, runtime_vec in data.items():
                runtime += np.mean(runtime_vec)

        print("DONE.")
        print("RUNTIME", runtime)
        return runtime

    def score(self, subgraph: SubgraphView, **kwargs):
        '''
        scores a subgraph (within the given graph / subgraph
        that was passed to the initializer) by running it
        and returning the runtime
        '''
        # generate an instance of SubgraphFusion
        # deepcopy the subgraph via json and apply transformation

        sdfg_copy = SDFG.from_json(self._sdfg.to_json())
        graph_copy = sdfg_copy.nodes()[self._state_id]
        subgraph_copy = SubgraphView(graph_copy, \
                                     [graph_copy.nodes()[self._graph.nodes().index(n)] for n in subgraph])
        fusion = SubgraphFusion(subgraph_copy)
        fusion.apply(sdfg_copy)

        # run and measure
        median_rt_fuse = self.run_with_instrumentation(sdfg_copy, graph_copy)


        return median_rt_fuse / self._median_rt_base
