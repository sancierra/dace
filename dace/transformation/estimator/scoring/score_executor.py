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
                 subgraph: SubgraphView = None,
                 gpu: bool = None,
                 nruns = 30,
                 ):
        super().__init__(sdfg, graph, subgraph)

        if gpu is None:
            # detect whether the state is assigned to GPU
            map_entries = helpers.get_outermost_scope_maps(sdfg, graph, subgraph)
            schedule = next(iter(map_entries)).schedule
            if any([m.schedule != schedule for m in map_entries]):
                raise RuntimeError("Schedules in maps to analyze should be the same")
            self._gpu = True if schedule in [dtypes.ScheduleType.GPU_Device, dtypes.ScheduleType.GPU_ThreadBlock] else False
        else:
            self._gpu = gpu

        self._sdfg_id = sdfg.sdfg_id
        self._state_id = sdfg.nodes().index(graph)
        self._nruns = nruns

        # input arguments: we just create a class variable
        self._inputs = inputs
        self._outputs = outputs

    def score(self, subgraph: SubgraphView, **kwargs):
        '''
        scores a subgraph (within the given graph / subgraph
        that was passed to the initializer) by running it
        and returning the runtime
        '''
        # generate an instance of SubgraphFusion
        # deepcopy the subgraph via json and apply transformation

        sdfg_copy = SDFG.from_json(self._sdfg.to_json())
        subgraph_copy = SubgraphView(sdfg_copy.nodes()[self._state_id], \
                                     [sdfg_copy.nodes()[self._state_id].nodes()[self._graph.nodes().index(n)] for n in subgraph])
        fusion = SubgraphFusion(subgraph_copy)
        fusion.apply(sdfg_copy)
        return 42
        # instrumentation:
        # mark old maps with instrumentation
        map_entries = helpers.get_outermost_scope_maps(self._sdfg, self._graph, subgraph, self._scope_dict)
        for map_entry in map_entries:
            map_entry.map.instrument = dtypes.InstrumentationType.Timer
        # mark new maps with instrumentation
        fusion._global_map_entry.map.instrument = dtypes.InstrumentationType.Timer

        # run and go
        self._sdfg(**self._arguments)
        sdfg_copy(**self._arguments)

        # remove old maps instrumentation
        for map_entry in map_entries:
            map_entry.map.instrument = None

        # get timing results
        report_old = self._sdfg.get_latest_report()
        report_new = sdfg_copy.get_latest_report()

        path = os.path.join(self._sdfg.build_folder, 'perf')
        files = [f for f in os.listdir(path) if f.startswith('report-')]
        assert len(files) > 1
        # Avoid import loops
        json_original = sorted(files, reverse = True)[1]
        json_transformed = sorted(files, reverse = True)[0]
        runtime_original, runtime_improved = 0,0
        with open(json_original) as f:
            data = json.load(f)
            for _, runtime_vec in data:
                runtime_original += np.mean(runtime_vec)
        with open(json_transformed) as f:
            data = json.load(f)
            rumtime_original = np.mean(data['outer_fused'])

        return runtime_improved
