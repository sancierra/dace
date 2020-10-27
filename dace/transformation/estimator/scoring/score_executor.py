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
        self._median_rt_base = self.run_with_instrumentation(
                sdfg = self._sdfg,
                graph = self._graph,
                map_entries = self._map_entries,
                check = False,
                set = True)

        # if nruns is defined, change config
        if nruns is not None:
            dace.config.Config.set('treps', value=nruns)

    def run_with_instrumentation(self,
                                 sdfg: SDFG,
                                 graph: SDFGState,
                                 map_entries = None,
                                 check = True,
                                 set = False):

        '''
        runs an sdfg with instrumentation on all outermost scope
        maps and returns their added runtimes
        '''
        if map_entries is None:
            map_entries = helpers.get_outermost_scope_maps(sdfg, graph)

        # instrumentation:
        # mark all mapentries  with instrumentation
        for map_entry in map_entries:
            # GPU_Events TODO
            if self._gpu:
                map_entry.map.instrument = dtypes.InstrumentationType.GPU_Events
            else:
                map_entry.map.instrument = dtypes.InstrumentationType.Timer

        # create a local copy of all the outputs and set all outputs
        # to zero. this will serve as an output for the current iteration
        outputs_local = {ok: ov.copy() for (ok, ov) in self._outputs.items()}
        for ok, kv in outputs_local.items():
            kv.fill(0)
        for ok, kv in outputs_local.items():
            print(ok)
            print(np.linalg.norm(kv))
            print(np.linalg.norm(self._outputs[ok]))
        try:
            r = sdfg(**self._inputs, **outputs_local, **self._symbols)
            if '__return' in outputs_local:
                outputs_local['__return'] = r
        except:
            print("ERROR")
            print("Runtime Error in current Configuration")

        if check:
            # this block asserts whether outputs are the same
            nv = True
            for (ok, ov) in self._outputs.items():
                if not np.allclose(outputs_local[ok], ov):
                    warnings.warn('Wrong output!')
                    if nv:
                        #sdfg.view()
                        nv = False
                    print('Original Output:',np.linalg.norm(ov))
                    print('Transformed Ouput:', np.linalg.norm(outputs_local[ok]))
                else:
                    print("PASS")
                    print(np.linalg.norm(ov))
                    print(np.linalg.norm(outputs_local[ok]))

        if set:
            # this block sets self._outputs according to the local
            # result. used for initialization.
            for (ok, ov) in outputs_local.items():
                self._outputs[ok] = ov

        # remove old maps instrumentation
        for map_entry in map_entries:
            map_entry.map.instrument = dtypes.InstrumentationType.No_Instrumentation

        # get timing results
        files = [f for f in os.listdir(os.path.join(sdfg.build_folder, 'perf'))
                                                    if f.startswith('report-')]
        print(files)
        assert len(files) > 0

        json_file = sorted(files, reverse = True)[0]
        runtime = 0.0
        path = os.path.join(sdfg.build_folder, 'perf', json_file)
        with open(path) as f:
            data = json.load(f)
            for _, runtime_vec in data.items():
                runtime += np.mean(runtime_vec)
        if runtime == 0.0:
            print("????? Runtime == 0")
            print("map_entries", map_entries)
            sdfg.view()
        else:
            os.remove(path)
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
