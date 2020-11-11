""" This file implements the ExecutionScore class """

from dace.transformation.subgraph import SubgraphFusion, StencilTiling, helpers
from dace.transformation.subgraph.composite import CompositeFusion
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
import sys 


@make_properties
class ExecutionScore(ScoringFunction):
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
    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 inputs: Dict,
                 outputs: Dict,
                 symbols: Dict,
                 subgraph: SubgraphView = None,
                 gpu: bool = None,
                 nruns=None,
                 transformation_function: Type = CompositeFusion,
                 **kwargs):
        super().__init__(sdfg=sdfg,
                         graph=graph,
                         subgraph=subgraph,
                         gpu=gpu,
                         transformation_function=transformation_function,
                         **kwargs)

        # input arguments: we just create a class variable
        self._inputs = inputs
        self._outputs = outputs
        self._symbols = symbols
        
        # if nruns is defined, change config
        if nruns is not None:
            dace.config.Config.set('treps', value=nruns)

        # run the graph to create a baseline
        # Use sdfg_init for this it if is not None, else just our sdfg
        self._median_rt_base = self.run_with_instrumentation(
            sdfg=self._sdfg,
            graph=self._graph,
            map_entries=self._map_entries,
            check=False,
            set=True)

    def run_with_instrumentation(self,
                                 sdfg: SDFG,
                                 graph: SDFGState,
                                 map_entries=None,
                                 check=True,
                                 set=False):
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

        # TODO: remove 
        for ok, kv in outputs_local.items():
            print(ok)
            print(np.linalg.norm(kv))
            print(np.linalg.norm(self._outputs[ok]))

        # execute and instrument
        try:
            # grab all inputs needed 
            sdfg_inputs = {
                k: v
                for (k, v) in self._inputs.items() if k not in outputs_local
            }
            # specialize sdfg
            sdfg.specialize(self._symbols)
            print("SPECIALIZING", self._symbols)
            r = sdfg(**sdfg_inputs, **outputs_local)
            outputs_local['__return'] = r

        except Exception as e:
            warnings.warn("ERROR")
            print("Runtime Error in current Configuration:")
            print(e)
            # in debug mode, exit and fail 
            if self.debug:
                i = 0
                while(os.path.exists(f"error{i}.sdfg")):
                    i += 1
                sdfg.save('error.sdfg')
                if self.exit_on_error:
                    sys.exit(0)
            
            return -1

        if check:
            # this block asserts whether outputs are the same
            faulty = False
            for (ok, ov) in self._outputs.items():
                if ov is not None and not np.allclose(outputs_local[ok], ov):
                    faulty = True
                    # the output is wrong, generate warning and output info
                    warnings.warn('WRONG OUTPUT!')
                    print('ERROR in array')
                    print(ok)
                    print('L2 Original Output:', np.linalg.norm(ov))
                    print('L2 Transformed Ouput:',
                          np.linalg.norm(outputs_local[ok]))
                elif ov is not None:
                    # the output appears to be correct
                    print("PASS")
                    print(np.linalg.norm(ov))
                    print(np.linalg.norm(outputs_local[ok]))
                else:
                    # function has no return value
                    pass
            if faulty and self.view_on_error:
                sdfg.view()
            if faulty and self.exit_on_error:
                sys.exit(0)
        if set:
            # this block sets self._outputs according to the local
            # result. used for initialization.
            for (ok, ov) in outputs_local.items():
                self._outputs[ok] = ov
                if ov is not None and np.linalg.norm(ov) == 0.0:
                    warnings.warn(f"Output has norm Zero for Array{ok}")
                    if self.view_on_error:
                        sdfg.view()
                    if self.exit_on_error:
                        sys.exit(0)

        # remove old maps instrumentation
        for map_entry in map_entries:
            map_entry.map.instrument = dtypes.InstrumentationType.No_Instrumentation

        # get timing results
        files = [
            f for f in os.listdir(os.path.join(sdfg.build_folder, 'perf'))
            if f.startswith('report-')
        ]
        assert len(files) > 0

        json_file = sorted(files, reverse=True)[0]
        runtime = 0.0
        path = os.path.join(sdfg.build_folder, 'perf', json_file)
        with open(path) as f:
            data = json.load(f)
            print(data)
            for _, runtime_vec in data.items():
                runtime += sum(runtime_vec)
        if runtime == 0.0:
            warnings.warning("Runtime is equal to Zero")
            if self.view_on_error:
                sdfg.view()
            if self.exit_on_error:
                sys.exit(0)
            print("map_entries", map_entries)
        else:
            os.remove(path)
        
        print("DONE.")
        print("RUNTIME", runtime)
        return runtime

    def score(self, subgraph: SubgraphView):
        '''
        scores a subgraph (within the given graph / subgraph
        that was passed to the initializer) by running it
        and returning the runtime
        '''
        # generate an instance of SubgraphFusion
        # deepcopy the subgraph via json and apply transformation_function

        sdfg_copy = SDFG.from_json(self._sdfg.to_json())
        graph_copy = sdfg_copy.nodes()[self._state_id]
        subgraph_copy = SubgraphView(graph_copy, [
            graph_copy.nodes()[self._graph.nodes().index(n)] for n in subgraph
        ])

        print("SUBGRAPH:", subgraph_copy.nodes())

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

        # run and measure
        median_rt_fuse = self.run_with_instrumentation(sdfg_copy, graph_copy)

        #return median_rt_fuse / self._median_rt_base
        return median_rt_fuse
