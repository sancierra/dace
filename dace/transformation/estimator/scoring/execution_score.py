# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This file implements the ExecutionScore class """

from dace.transformation.subgraph import helpers
from dace.transformation.subgraph.composite import CompositeFusion
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

import dace
import dace.dtypes as dtypes
import dace.sdfg.nodes as nodes 

from typing import Dict, Type

from dace.transformation.estimator.scoring import ScoringFunction

import json
import warnings
import os
import numpy as np
import sys


@make_properties
class ExecutionScore(ScoringFunction):
    '''
    Evaluates Subgraphs by executing their associated 
    binary and measuring subgraph or program runtime.
    Score returned corresponds to runtime or fraction 
    thereof.
    '''

    debug = Property(desc = "Debug Mode",
                     dtype = bool,
                     default = True)

    run_baseline = Property(desc = "Run baseline SDFG inputted upon construction"
                                   "as a comparision baseline. Score returned is then calculated"
                                   "as a fraction compared to baseline runtime. Also used"
                                   "for output error checking.",
                            dtype = bool,
                            default = False)

    exit_on_error = Property(desc = "Perform sys.exit if a numerical or execution error" 
                                    "error occurs, else the score returned is -1",
                             dtype = bool,
                             default = False)

    time_graph_only = Property(desc = "Only measure runtime execution of the specified graph"
                                      "else the whole SDFG is timed.",
                              dtype = bool,
                              default = True)
    
    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 device: dtypes.DeviceType = dtypes.DeviceType.CPU,
                 transformation_function: Type = CompositeFusion,
                 transformation_properties: Dict = None,
                 io: Dict = None,
                 nruns = None):
        ''' 
        :param sdfg: SDFG of interest 
        :param graph: State of interest 
        :param subgraph: Subgraph 
        :param io: Tuple of (input_dict, output_dict, symbols_dict), where each of 
                   these dictionaries contains a mapping from argument/symbol name 
                   to its corresponding argument. If None, IO arguments are not
                   required.
        :param device: Device target 
        :param nruns: Determines how many runs are executed for time measurements.
                      Median runtime is chosen.
        :param transformation_function: Transformation function to be instantiated
                                        and applied to a subgraph before scoring
        
        '''
        super().__init__(sdfg=sdfg,
                         graph=graph,
                         device=device,
                         transformation_function=transformation_function,
                         transformation_properties=transformation_properties,
                         io=io)

        # if nruns is defined, change config
        if nruns is not None:
            dace.config.Config.set('treps', value=nruns)


        # run the graph to create a baseline
        # Use sdfg_init for this it if is not None, else just our sdfg
        if self.run_baseline:
            self._median_rt_base = self.run_with_instrumentation(
                sdfg=self._sdfg,
                graph=self._graph,
                map_entries=self._map_entries,
                numerical_check=False,
                set_outputs=True)

    def run_with_instrumentation(self,
                                 sdfg: SDFG,
                                 graph: SDFGState,
                                 nodes_to_instrument=None,
                                 numerical_check=True,
                                 set_outputs=False):
        '''
        runs an sdfg with instrumentation on all outermost scope
        maps and returns their added runtimes
        :param sdfg:        SDFG
        :param graph:       SDFGState of interest
        :param map_entries: List of outermost scope map entries
                            If None, those get calculated
        :param numerical_check: Check whether the outputs are the
                                same as the output class variable

        :param set_outputs: Set the output class variable to
                            the output produced by this call

        '''
        if nodes_to_instrument is None:
            nodes_to_instrument = set()
            nodes_to_instrument |= helpers.get_outermost_scope_maps(sdfg, graph)
            nodes_to_instrument |= [n for n in graph.nodes() if (isinstance(n, nodes.Tasklet) and self._scope_dict[n] is None)]
            
        # instrumentation:
        # mark all mapentries  with instrumentation
        for node in nodes_to_instrument:
            if self._device == dtypes.DeviceType.CPU:
                if isinstance(node, nodes.AccessNode):
                    node.instrument = dtypes.InstrumentationType.Timer
                elif isinstance(node, nodes.EntryNode):
                    node.map.instrument = dtypes.InstrumentationType.Timer 
                else:
                    raise TypeError("Instrumentation has to be performed" 
                                    "on map entry or tasklet")

            elif self._device == dtypes.DeviceType.GPU: 
                if isinstance(node, nodes.AccessNode): 
                    node.instrument = dtypes.InstrumentationType.GPU_Events
                elif isinstance(node, nodes.EntryNode):
                    node.map.instrument = dtypes.InstrumentationType.GPU_Events 
                else:
                    raise TypeError("Instrumentation has to be performed" 
                                    "on map entry or tasklet") 

            else:
                raise TypeError("ExecutionScore Instrumentation only" 
                                "defined for CPU and GPU") 

        # clear the runtime folder
        if os.path.exists(os.path.join(sdfg.build_folder, 'perf')):
            for f in os.listdir(os.path.join(sdfg.build_folder, 'perf')):
                if f.startswith('report-'):
                    os.remove(os.path.join(sdfg.build_folder, 'perf',f))
        # create a local copy of all the outputs and set these
        # to zero. this will serve as an output for the current iteration
        outputs_local = {}
        for out_k in self._outputs.keys():
            if out_k in self._inputs:
                # link to corresponding input
                outputs_local[out_k] = self._inputs[out_k].copy()

        # execute and instrument
        try:
            # grab all inputs needed
            sdfg_inputs = { 
                int_k: int_v
                for (int_k, int_v) in self._inputs.items() if int_k not in outputs_local
            }
            # specialize sdfg
            sdfg.specialize(self._symbols)
            r = sdfg(**sdfg_inputs, **outputs_local)
            if r is not None:
                if isinstance(r, tuple):
                    for (i, result) in enumerate(r):
                        outputs_local[f'__return{i}'] = r
                else:
                    outputs_local[f'__return'] = r
            success = True

        except Exception as e:
            warnings.warn("ExecutionScore:: Exception Caught")
            print(e)
            success = False

        # this block asserts whether outputs are correct
        if success and numerical_check and self.run_baseline:
            for (ok, ov) in self._outputs.items():
                if ov is not None and not np.allclose(outputs_local[ok], ov):
                    success = False 
                    diff = np.linalg.norm(ov - outputs_local[ok])
                    warnings.warn('ExecutionScore:: Output diff too large for {ok} = {diff}')

        if success and set_outputs:
            # this block sets self._outputs according to the local result
            # used for initialization
            for (ok, ov) in outputs_local.items():
                self._outputs[ok] = ov
                if ov is not None and np.linalg.norm(ov) == 0.0:
                    warnings.warn(f"ExecutionScore:: Output has norm Zero for Array{ok}")
                    success = False

        # revert instrumentation
        for node in nodes_to_instrument:
            if isinstance(node, nodes.MapEntry):
                node.map.instrument = dtypes.InstrumentationType.No_Instrumentation
            if isinstance(node, nodes.Tasklet):
                node.instrument = dtypes.InstrumentationType.No_Instrumentation
        

        # if not succeeded in any part, output if necessary and return -1
        if not success:
            if self.exit_on_error:
                sys.exit(0)
            else:
                return -1 

        # get timing results
        files = [
            f for f in os.listdir(os.path.join(sdfg.build_folder, 'perf'))
            if f.startswith('report-')
        ]
        assert len(files) > 0

        runtime = []
        for json_file in files:
            path = os.path.join(sdfg.build_folder, 'perf', json_file)
            with open(path) as f:
                data = json.load(f)['traceEvents']
                for runtime_dict in data:
                    runtime.append(runtime_dict['dur'])

        median_runtime = np.median(runtime)

        if median_runtime == 0.0:
            warnings.warn("ExecutionScore:: Median Runtime is equal to Zero")

        return median_runtime

    def score(self, subgraph: SubgraphView):
        '''
        scores a subgraph that resides within the graph that was passed 
        to the initializer. It is scored by running it directly and measuring 
        runtime according to property policies. 
        :param subgraph: Subgraph to which transformation function is applied 
                         before scoring (-> execution runtime) 
        '''
        # generate an instance of the transformation function and apply 
        # it to a deepcopied version of the SDFG 
        
        # FORNOW: JSON copy for better speed (TODO) 
        sdfg_copy = SDFG.from_json(self._sdfg.to_json())
        graph_copy = sdfg_copy.nodes()[self._state_id]
        subgraph_copy = SubgraphView(graph_copy, [
            graph_copy.nodes()[self._graph.nodes().index(n)] for n in subgraph
        ])

        transformation_function = self._transformation(subgraph_copy)
        # assign properties to transformation
        for (arg, val) in self._transformation_properties.items():
            try:
                setattr(transformation_function, arg, val)
            except AttributeError:
                warnings.warn(f"Attribute {arg} not found in transformation")
            except (TypeError, ValueError):
                warnings.warn(f"Attribute {arg} has invalid value {val}")

        transformation_function.apply(sdfg_copy)

        # run and measure
        median_rt_fuse = self.run_with_instrumentation(sdfg_copy, graph_copy)

        if self.run_baseline:
            return median_rt_fuse / self._median_rt_base if median_rt_fuse != -1 else -1
        else:
            return median_rt_fuse 
    
    @staticmethod
    def name():
        return "Runtime Measurement"
    
    def label(self):
        if self.run_baseline:
            return "Transformed Subgraph RT / SDFG RT"
        else:
            return "Transformed Subgraph RT"

