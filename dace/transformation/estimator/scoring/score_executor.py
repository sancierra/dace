""" This file implements the ExecutionScore class """

from dace.transformation.subgraph import SubgraphFusion, StencilTiling, helpers
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

import dace
import dace.sdfg.nodes as nodes
import dace.dtypes as dtypes

from collections import deque, defaultdict, ChainMap
from typing import Set, Union, List, Callable, Dict, Type

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
                 transformations: List[Type] = [SubgraphFusion]
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
        self._transformations = transformations
        # input arguments: we just create a class variable
        self._inputs = inputs
        self._outputs = outputs
        self._symbols = symbols

        # if nruns is defined, change config
        if nruns is not None:
            dace.config.Config.set('treps', value=nruns)

        # run the graph to create a baseline
        self._median_rt_base = self.run_with_instrumentation(
                sdfg = self._sdfg,
                graph = self._graph,
                map_entries = self._map_entries,
                check = False,
                set = True)


    def run_with_instrumentation(self,
                                 sdfg: SDFG,
                                 graph: SDFGState,
                                 map_entries = None,
                                 check = True,
                                 set = False):

        '''
        runs an sdfg with instrumentation on all outermost scope
        maps and returns their added runtimes
        :param sdfg:        SDFG
        :param graph:       SDFGState of interest
        :param map_entries: List of outermost scope map entries
                            If None, those get calculated
        :param check:       Check whether the outputs are the
                            same as the output class variable
        :param set:         Set the output class variable to
                            the output produced by this call

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
        outputs_local = {}
        for ok, ov in self._outputs.items():
            if ok in self._inputs:
                # create a copy of the input
                outputs_local[ok] = self._inputs[ok].copy()
            elif ok != '__return':
                # pure output value, just do a setzero
                outputs_local[ok] = ov.copy()
                outputs_local[ok].fill(0)

        for ok, kv in outputs_local.items():
            print(ok)
            print(np.linalg.norm(kv))
            print(np.linalg.norm(self._outputs[ok]))

        # execute and instrument
        try:
            sdfg_inputs = {k:v for (k,v) in self._inputs.items()
                           if k not in outputs_local}
            r = sdfg(**sdfg_inputs, **outputs_local, **self._symbols)
            outputs_local['__return'] = r

        except Exception as e:
            warnings.warn("ERROR")
            print("Runtime Error in current Configuration:")
            print(e)

        if check:
            # this block asserts whether outputs are the same
            nv = True
            for (ok, ov) in self._outputs.items():
                if ov is not None and not np.allclose(outputs_local[ok], ov):
                    # the output is wrong, generate warning and output info
                    warnings.warn('WRONG OUTPUT!')
                    if nv:
                        sdfg.view()
                        nv = False
                    print('ERROR in array')
                    print(ok)
                    print('L2 Original Output:',np.linalg.norm(ov))
                    print('L2 Transformed Ouput:', np.linalg.norm(outputs_local[ok]))
                elif ov is not None:
                    # the output appears to be correct
                    print("PASS")
                    print(np.linalg.norm(ov))
                    print(np.linalg.norm(outputs_local[ok]))
                else:
                    # function has no return value
                    pass
        if set:
            # this block sets self._outputs according to the local
            # result. used for initialization.
            for (ok, ov) in outputs_local.items():
                print("##")
                print(ok)
                self._outputs[ok] = ov


        # remove old maps instrumentation
        for map_entry in map_entries:
            map_entry.map.instrument = dtypes.InstrumentationType.No_Instrumentation

        # get timing results
        files = [f for f in os.listdir(os.path.join(sdfg.build_folder, 'perf'))
                                                    if f.startswith('report-')]
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

        map_entries_copy = helpers.get_outermost_scope_maps(sdfg_copy, graph_copy)
        print("SUBGRAPH:", subgraph_copy)
        for trafo_type in self._transformations:
            transformation = trafo_type(subgraph_copy)
            transformation.apply(sdfg_copy)

            # StencilTiling: add to nodes
            if isinstance(trafo_type, StencilTiling):
                for map_entry in map_entries_copy:
                    outer_entry = graph_copy.in_edges(map_entry)[0].src
                    outer_exit = graph.exit_node(outer_entry)
                    subgraph_copy._subgraph_nodes += [outer_entry, outer_exit]


        # run and measure
        median_rt_fuse = self.run_with_instrumentation(sdfg_copy, graph_copy, map_entries_copy)


        return median_rt_fuse / self._median_rt_base
