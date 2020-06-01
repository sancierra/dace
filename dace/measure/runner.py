import dace
import numpy as np

from dace.perf.roofline import Roofline
from dace.perf.specs import *
from dace.perf.optimizer import SDFGRooflineOptimizer

from dace.transformation.heterogeneous import ReduceMap
from dace.transformation.heterogeneous import SubgraphFusion
from dace.transformation.heterogeneous import MultiExpansion

import dace.libraries.standard as stdlib

import dace.sdfg.nodes as nodes

import timeit

from dace.measure.pipeline import expand_reduce, expand_maps, fusion

from collections import OrderedDict

class Runner():
    # A measurement and correctness checker for an SDFG
    # with different transformation pipelines

    measures = {'max': lambda x : max(x),
                'median': lambda x : np.median(x),
                'avg': lambda x: np.mean(x)}

    def __init__(self,
                 debug = True, verbose = False,
                 measure_mode = 'median', measure_time_all = False
                 view = False, view_init = False, view_final = False, view_all = False):

        self.debug = debug
        self.verbose = verbose

        self.measure_time = measure_time
        self.measure_mode = measure_mode

        self.view = view
        self.view_init = view_init
        self.view_final = view_final
        self.view_all = view_all

    def _setzero_outputs(self, outputs):
        for element in outputs:
            if isinstance(outputs[element], (np.ndarray, dace.ndarray)):
                # generic vector -> set zero
                outputs[element][:] = 0
            else:
                print("WARNING: Non-vector type in outputs -> SetZero failed.")


    def _get_runtimes(self):
        runtimes = []
        with open('results.log','r') as file:
            for line in file.readlines():
                runtimes.append(float(line.split(" ")[3]))
        return runtimes



    def _print_norms(self, outputs, result):
        print(f"{fun_name} norms:")
        for element in outputs:
            if isinstance(element, (np.ndarray, dace.ndarray)):
                if outputs[element]:
                    print(np.linalg.norm(outputs[element]))
                else:
                    print(np.linalg.norm(result))



    def _print_arrays(self, outputs, result):
        for element in outputs:
            print(element)
            if outputs[element]
                print(outputs[element])
            else:
                print(result)




    def pipeline(self,
                 sdfg, graph,
                 inputs, outputs,
                 subgraph = None,
                 roofline = None,
                 pipeline = [expand_reduce, expand_maps, fusion]):

        none_counter = 0
        for element in outputs:
            if not outputs[element]:
                none_counter += 1
        if none_counter > 1:
            raise RuntimeError('Multiple return types not supported')


        if self.view_init:
            sdfg.view()

        # name and lists used for storing all the results
        name = sdfg.name
        runtimes = []
        diffs = []

        # establish a baseline
        self._setzero_outputs(outputs)
        csdfg = sdfg.compile_directly()
        result = csdfg(**inputs)

        outputs_baseline = {}
        for element in outputs:
            if outputs[element]:
                outputs_baseline[element] = dcpy(outputs[element])
            else:
                outputs_baseline[element] = result

        # get runtimes:
        runtimes.append(self.measures[self.measure_mode](self._get_runtimes()))

        if roofline:
            roofline.evaluate(name, runtimes[-1])

        if debug:
            self._print_norms(outputs_baseline, result)
            if verbose:
                self._print_arrays(outputs_baseline, result)

        for fun in pipeline:
            # apply transformation
            fun_name = name + fun.__name__
            fun(sdfg, graph, subgraph)
            if self.view:
                sdfg.view()

            csdfg = sdfg.compile_directly()
            result = csdfg(**inputs)

            current_runtime = None
            if self.measure_time_all:
                current_runtime = self.measures[self.measure_mode](self._get_runtimes())
            runtimes.append(current_runtime)

            if self.roofline:
                self.roofline.evaluate(sdfg,
                                       subgraph if subgraph else graph,
                                       running_time = current_runtime)

            # process outputs
            if debug:
                self._print_norms(outputs, result)
                if verbose:
                    self._print_arrays(outputs, result)

            # see whether equality with baseline holds
            difference_dict = {}
            for element in outputs:
                if outputs[element]:
                    current = outputs[element]
                else:
                    current = result
                # NaN check
                if any(np.isnan(current)):
                    print(f"WARNING: NaN detected in output {element} in {fun_name}")

                difference_dict[element] = np.linalg.norm(current - outputs_baseline[element])
