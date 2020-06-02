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
                'avg': lambda x: np.mean(x),
                'std': lambda x: np.std(x)}

    def __init__(self,
                 debug = True, verbose = False,
                 measure_mode = ['median', 'std'],
                 view = False, view_all = False,
                 view_roofline = True,
                 error_abs = 1e-7, error_rel = 1e-7):

        """ A runner wrapper for DaCe programs for testing runtimes and
            correctness of heterogeneous transformations.
            :param debug: display additional information
            :param verbose: display a lot of information, print all arrays
            :param measure_mode: statistical parameters for runtime analysis.
                                 must be a vector, currently supported: max, median, std, avg
                                 The first element of the vector will also go into the general
                                 summary at the end.
            :param view: view graph at the beginning and at the end
            :param view_all: view graph at every transformation step
            :param view_roofline: view graph at end with roofline plot
            :param error_abs: absolut error tolerance for array checks
            :param error_rel: relative error tolerance for array checks

        """

        self.debug = debug
        self.verbose = verbose

        self.measure_time = measure_time
        self.measure_mode = measure_mode

        self.view = view
        self.view_init = view_init
        self.view_final = view_final
        self.view_all = view_all

        self.error_abs = error_abs
        self.error_rel = error_rel

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

    def _get_runtime_stats(self, runtimes):
        stats = []
        for measure in measure_mode:
            stats.append(Runner.measures[measure](current_runtimes))
        return stats

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
            if outputs[element]:
                print(outputs[element])
            else:
                print(result)

    @staticmethod
    def generate_arguments(sdfg,
                           symbols_dict,
                           outputs = [],
                           outputs_setzero = True):
        """ Auto argument generator. Uses numpy.random.random.
        :param sdfg: SDFG to invoke
        :param symbols_dict: Mapping of all symbols(str) to their size
        :param outputs: Arguments in this list are treated as outputs.
                        iff outputs_setzero, all outputs are set to zero
                        instead of a random value
        :param outputs_setzero: Set arguments contained in outputs to zero

        :return: A dictionary mapping of argument to constructed array
        """

        arglist = sdfg.arglist()
        free_symbols = sdfg.free_symbols
        for symbol in free_symbols:
            if symbol not in symbols_dict:
                raise RuntimeError("Not all Symbols defined! \
                                    Need complete symbol dict for operation")
        result = {}
        for (argument, array_reference) in arglist.items():
            if argument in symbols:
                # do not care -- symbols have to be defined before
                continue
            # infer numpy dtype
            array_dtype = array_reference.dtype.type
            # infer shape with the aid of symbols_dict
            array_shape = tuple([symbols_dict[str(e)] if isinstance(e, (str, dace.symbol, sympy.symbol)) \
                                 else symbols_dict[e] \
                                 for e in array_reference.shape])
            if argument in outputs and outputs_setzero:
                result[argument] = np.zeros(shape, dtype=array_dtype)
            else:

                result[argument] = np.random.random(shape=array_shape, dtype = array_dtype)

        return result

    def go(self,
           sdfg, graph,
           inputs, outputs,
           subgraph = None,
           roofline = None,
           pipeline = [expand_reduce, expand_maps, fusion]):

        """ Test a pipeline specified as the argument 'pipeline'.
            :param sdfg: SDFG object
            :param graph: corresponding state to be analyzed
            :param inputs: dictionary containing all input arrays as a value and
                           their corresponding (input) name as a key
            :param outputs: dictionary containing all output arrays as a value
                            and their corresponding (output) name as a key.
                            Can specify any arbitrary input array as an output array
                            if one wants to assert its correctness.
                            If the output is a return value, the corresponding value
                            item should be set to None. There can only be one None
                            value (multiple return types are not supported).
            :param subgraph: corresponding subgraph to be analyzed. Must be either
                             a SubgraphView instance or None. In the latter case,
                             the whole graph is just taken.
            :param roofline: Roofline object that can be passed along. At every
                             transformation step the roofline model is evaluated
                             and the corresponding runtime added.
            :param pipeline: Pipeline to be tested. Default is heterogeneous
                             fusion pipeline (one by one), there is also a
                             combined version to be found in pipeline.py
        """


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
        diffs_abs = []
        diffs_rel = []
        verdicts = []

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
        current_runtimes = self._get_runtimes()
        runtimes.append(self._get_runtime_stats(current_runtimes))

        if roofline:
            roofline.evaluate(name, runtimes[-1])

        if debug:
            self._print_norms(outputs_baseline, result)
            if verbose:
                self._print_arrays(outputs_baseline, result)

        # NaN check
        for output in outputs:
            if any(np.isnan(output)):
                print(f"WARNING: NaN detected in output {output} in Baseline")

        for fun in pipeline:
            # apply transformation
            fun_name = name + fun.__name__
            fun(sdfg, graph, subgraph)
            if self.view:
                sdfg.view()

            csdfg = sdfg.compile_directly()
            result = csdfg(**inputs)

            current_runtimes = self._get_runtimes()
            runtimes.append(self._get_runtime_stats(current_runtimes))

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
            difference_dict_abs = {}
            difference_dict_rel = {}
            verdicts_dict = {}
            for element in outputs:
                if outputs[element]:
                    current = outputs[element]
                else:
                    current = result
                # NaN check
                if any(np.isnan(current)):
                    print(f"WARNING: NaN detected in output {element} in {fun_name}")

                difference_dict_abs[element] = np.linalg.norm(current - outputs_baseline[element])
                difference_dict_rel[element] = np.linalg.norm(current - outputs_baseline[element]) / np.linalg.norm(current)
                verdicts_dict[element] = 'PASS' if difference_dict_abs[element] < self.error_abs and \
                                                   difference_dict_rel[element] < self.error_rel \
                                          else 'FAIL'


            diffs_abs.append(difference_dict_abs)
            diffs_rel.append(difference_dict_rel)
            verdicts.append(verdicts_dict)

        ###############################################################
        ###################### Print Results ##########################
        ###############################################################

        print("### Transformation Correctness ###")
        print("Transformation".rjust(15,' '),
              "Output".rjust(8,' '),
              "Diff Abs".rjust(10,' '),
              "Diff Rel".rjust(10,' '),
              "Vertict")
        for transformation, diff_abs_dict, diff_rel_dict, verdicts_dict \
                        in zip(pipeline, diff_abs, diff_rel, verdicts):

            arrays = sorted(list(diff_abs_dict.keys()))
            for array in arrays:
                print(transformation.__name__.rjust(15,' '),
                      array.rjust(8,' '),
                      diff_abs_dict[array].rjust(10,' '),
                      diff_rel_dict[array].rjust(10,' '),
                      verdicts_dict[array])

        print("##################################")
        print("########### Runtimes #############")
        print("Transformation".rjust(15,' '), end='')
        for measure in self.measure_mode:
            print(measure.rjust(8,' '), end='')
        print('\n')

        for transformation, runtime_list in zip(pipeline, runtimes):
            print(transformation.rjust(15,' '), end='')
            for runtime in runtime_list:
                print(runtime.rjust(8,' '), end='')
            print('\n')

        print("############ Summary #############")

        print("Transformation".rjust(15,' '),
              f"Runtime {self.measure_mode[0]}".rjust(15, ' '),
              "Diff Verdict")

        for transformation, runtime_list, verdicts_dict in zip(pipeline, runtimes, verdicts):
            print(transformation.rjust(15,' '),
                  runtimes[0].rjust(15,' '),
                  'PASS' if all([v == 'PASS' for v in verdicts.values()]) else 'FAIL')

        print("##################################")
