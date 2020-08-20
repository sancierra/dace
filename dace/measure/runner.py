import dace
import numpy as np

from dace.perf.roofline import Roofline
from dace.perf.specs import *
from dace.perf.optimizer import SDFGRooflineOptimizer

import dace.libraries.standard as stdlib
import dace.sdfg.nodes as nodes

import timeit
import sympy
from copy import deepcopy as dcpy

from dace.transformation.subgraph.pipeline import expand_reduce, expand_maps, fusion

import dace.symbolic as sym

from collections import OrderedDict
from typing import List

class Runner():
    # A measurement and correctness checker for an SDFG
    # with different transformation pipelines

    measures = {'max': lambda x : max(x),
                'min': lambda x : min(x),
                'median': lambda x : np.median(x),
                'avg': lambda x: np.mean(x),
                'std': lambda x: np.std(x),
                'q75': lambda x: np.quantile(x, 0.75),
                'q25': lambda x: np.quantile(x, 0.25)}

    def __init__(self,
                 debug = True, verbose = False,
                 measure_mode = ['median', 'avg', 'max', 'std'],
                 view = False,
                 view_roofline = True, save_roofline = False,
                 error_tol_abs = 1e-6, error_tol_rel = 1e-7,
                 sequential = False):

        """ A runner wrapper for DaCe programs for testing runtimes and
            correctness of subgraph transformations.
            :param debug: display additional information
            :param verbose: display a lot of information, print all arrays
            :param measure_mode: statistical parameters for runtime analysis.
                                 must be a vector, currently supported: max, median, std, avg
                                 The first element of the vector will also go into the general
                                 summary at the end.
            :param view: view graph at the beginning and at the end
            :param view_roofline: view graph at end with roofline plot
            :param error_tol_abs: absolut error tolerance for array checks
            :param error_tol_rel: relative error tolerance for array checks
            :param sequential: if True, transformations are executed sequentially on top of each other,

        """

        self.debug = debug
        self.verbose = verbose

        self.measure_mode = measure_mode

        self.view = view
        self.view_roofline = view_roofline

        self.error_tol_abs = error_tol_abs
        self.error_tol_rel = error_tol_rel

        self.sequential = sequential
        self.save_roofline = save_roofline

    def _setzero_outputs(self, outputs):
        for element in outputs:
            if isinstance(outputs[element], (np.ndarray, dace.dtypes.typeclass)):
                # generic vector -> set zero
                outputs[element][:] = 0
            elif isinstance(outputs[element], type(None)):
                # this must be the return value
                pass
            else:
                print("WARNING: Non-vector type in outputs -> SetZero failed.")


    def _get_runtimes(self):
        runtimes = []
        with open('results.log','r') as file:
            for line in file.readlines():
                runtimes.append(float(line.split("\t")[3]))

        return runtimes

    def _get_runtime_stats(self, runtimes):
        stats = []
        for measure in self.measure_mode:
            stats.append(Runner.measures[measure](runtimes))
        return stats

    def _print_norms(self, outputs, result):
        for element in outputs:
            print(f"Array {element}:",end=' ')
            try:
                print(np.linalg.norm(outputs[element] if outputs[element] is not None else result))
            except ValueError:
                print("Error: Could not print norm -- np.linalg.norm does not work on this type")

    def _print_arrays(self, outputs, result):
        for element in outputs:
            print(f"Array {element}:", end = ' ')
            print(outputs[element] if outputs[element] else result)

    @staticmethod
    def build_symbols_dict(*args):
        symbols_dict = {}
        for arg in args:
            symbols_dict[str(arg)] = arg.get()
        return symbols_dict

    def _run(self, sdfg, **args):
        csdfg = sdfg.compile_directly()
        return csdfg(**args)

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

        :return: Two dictionaries, input_dict and output_dict
                 pointing to the created arrays
                 output_dict contains all args in output plus a possbile return
                 array, in_dict all the input arguments found (except symbols)
        """
        arglist = sdfg.arglist()

        result_input = {}
        result_output = {}
        for (argument, array_reference) in arglist.items():
            if argument in symbols_dict or argument == '__return':
                # do not care -- symbols have to be defined in andvance
                # we don't need to initialize return value
                continue
            # infer numpy dtype
            array_dtype = array_reference.dtype.type
            # infer shape with the aid of symbols_dict
            array_shape = tuple([sym.evaluate(e, symbols_dict) for e in array_reference.shape])
            print(argument, "|", array_shape,"|", type(sdfg.data(argument)))
            if isinstance(sdfg.data(argument), dace.data.Array):
                new_data = np.random.random(size = array_shape).astype(array_dtype)
                if outputs_setzero:
                    new_data = np.zeros(shape = array_shape, dtype = array_dtype)

            #if isinstance(sdfg.data(argument), dace.data.Scalar):
            elif isinstance(sdfg.data(argument), dace.data.Scalar):
                new_data = np.random.rand()
                if outputs_setzero:
                    new_data = 0
            else:
                # fornow: just set to one
                new_data = 1

            result_input[argument] = new_data
            if argument in outputs:
                result_output[argument] = new_data


        return (result_input, result_output)

    def test_run(self,
                 sdfg, graph,
                 inputs, outputs, symbols,
                 subgraph = None,
                 roofline = None,
                 pipeline = [expand_reduce, expand_maps, fusion],
                 pipeline_args = None,
                 report = False):

        """ Test a pipeline specified as the argument 'pipeline'.
            :param sdfg: SDFG object
            :param graph: corresponding state to be analyzed
            :param inputs: dictionary containing all input arrays as a value and
                           their corresponding (input) name as a key
            :param outputs: dictionary containing all output arrays as a value
                            and their corresponding (output) name as a key.
                            Can specify any arbitrary input array as an output array
                            if one wants to assert its correctness.
                            Return values always get asserted
            :param symbols: symbols given as caller argument, Dict
            :param subgraph: corresponding subgraph to be analyzed. Must be either
                             a SubgraphView instance or None. In the latter case,
                             the whole graph is just taken.
            :param roofline: Roofline object that can be passed along. At every
                             transformation step the roofline model is evaluated
                             and the corresponding runtime added.
            :param pipeline: Pipeline to be tested. Default is subgraph
                             fusion pipeline (one by one), there is also a
                             combined version to be found in pipeline.py
        """

        # check whether the sdfg has a return value
        # if so, add it to the outputs array with value
        if '__return' in sdfg.arglist():
            outputs['__return'] = None

        if not pipeline_args:
            pipeline_args = []
            for element in pipeline:
                if isinstance(element, List):
                    pipeline_args.append([{}]*len(element))
                else:
                    pipeline_args.append({})

        if self.view:
            sdfg.view()

        if not self.sequential:
            sdfg_base = dcpy(sdfg)
            graph_index = sdfg.nodes().index(graph)
            if subgraph:
                subgraph_index = [graph.nodes().index(e) for e in subgraph.nodes()]

        # name and lists used for storing all the results
        names = ['baseline']
        runtimes = []
        diffs_abs = []
        diffs_rel = []
        verdicts = []
        nan_detected = False

        # establish a baseline
        self._setzero_outputs(outputs)
        result = self._run(sdfg, **inputs, **symbols)

        outputs_baseline = {}
        for element in outputs:
            outputs_baseline[element] = dcpy(outputs[element]) if outputs[element] is not None \
                                        else result

        # get runtimes:
        try:
            current_runtimes = self._get_runtimes()
        except FileNotFoundError:
            print("ERROR in runner.py: Please enable profiling in .dace.conf!")
        runtimes.append(self._get_runtime_stats(current_runtimes))

        if roofline:
            # just pass the first runtime measure into
            roofline.evaluate('baseline', graph, runtimes = current_runtimes)
        if self.debug:
            print(f"Baseline Norms:")
            self._print_norms(outputs_baseline, result)
            if self.verbose:
                print(f"Baseline Arrays:")
                self._print_arrays(outputs_baseline, result)

        # NaN check
        for element in outputs:
            current = outputs[element] if outputs[element] is not None else result
            if np.isnan(current).any():
                print(f"WARNING: NaN detected in output {current} in Baseline")
                nan_detected = True

        for fun, args in zip(pipeline, pipeline_args):
            if not self.sequential:
                sdfg = dcpy(sdfg_base)
                graph = sdfg.nodes()[graph_index]
                if subgraph:
                    subgraph = nodes.SubgraphView([graph.nodes()[i] for i in subgraph_index])

            # determine name of transformation
            if isinstance(fun, List):
                name = ''
                for index,func in enumerate(fun):
                    if index > 0:
                        name += '|'
                    name += func.__name__[0:3]
            else:
                name = fun.__name__
            names.append(name)

            # apply transformation
            if isinstance(fun, List):
                for func, arg in zip(fun, args):
                    func(sdfg, graph, subgraph, **arg)
            else:
                fun(sdfg, graph, subgraph, **args)


            self._setzero_outputs(outputs)
            result = self._run(sdfg, **inputs, **symbols)
            sdfg.apply_strict_transformations()

            current_runtimes = self._get_runtimes()
            runtimes.append(self._get_runtime_stats(current_runtimes))

            if roofline:
                roofline.evaluate(name,
                                  graph,
                                  runtimes = current_runtimes)

            # process outputs
            if self.debug:
                print(f"{name} Norms:")
                self._print_norms(outputs, result)
                if self.verbose:
                    print(f"{name} Arrays:")
                    self._print_arrays(outputs, result)

            # see whether equality with baseline holds
            difference_dict_abs = {}
            difference_dict_rel = {}
            verdicts_dict = {}
            for element in outputs:
                current = outputs[element] if outputs[element] is not None else result
                # NaN check
                if np.isnan(current).any():
                    print(f"WARNING: NaN detected in output {element} in {name}")
                    nan_detected = True

                try:
                    difference_dict_abs[element] = np.linalg.norm(current - outputs_baseline[element])
                    difference_dict_rel[element] = np.linalg.norm(current - outputs_baseline[element]) / np.linalg.norm(current)

                    verdicts_dict[element] = 'PASS' if np.allclose(current, outputs_baseline[element], \
                                                                   atol = self.error_tol_abs, \
                                                                   rtol = self.error_tol_rel) \
                                                    else 'FAIL'
                except ValueError:
                    print(f"Runner::Test::ValueError: \
                            Could not apply np.linalg.norm onto {element} \
                            of type {type(outputs[element])}")
                    raise ValueError()

            diffs_abs.append(difference_dict_abs)
            diffs_rel.append(difference_dict_rel)
            verdicts.append(verdicts_dict)

            if self.view:
                sdfg.view()


        ###############################################################
        ###################### Print Results ##########################
        ###############################################################

        print("################################################################")
        print("################## TRANSFORMATION CORRECTNESS ##################")
        if nan_detected:
            print("WARNING: NaN detected. See debug log for further information")
        print("Transformation".ljust(15,' '),
              "Output".ljust(15,' '),
              "Diff Abs".ljust(12,' '),
              "Diff Rel".ljust(12,' '),
              "Verdict")
        if len(outputs) == 0:
            print("                      No Outputs specified                      " )
        for idx, (transformation, diff_abs_dict, diff_rel_dict, verdicts_dict) \
                        in enumerate(zip(pipeline, diffs_abs, diffs_rel, verdicts)):

            arrays = sorted(list(diff_abs_dict.keys()))
            for array in arrays:
                print("transformation"+str(idx),
                      f"{diff_abs_dict[array]:.6g}".ljust(12,' '),
                      f"{diff_rel_dict[array]:.6g}".ljust(12,' '),
                      verdicts_dict[array])
        '''
        print("################################################################")
        print("########################### CONFIG #############################")
        print("Operation Mode".ljust(30, ' '), 'seq' if self.sequential else 'non-seq')
        print("Number of Runs per Call".ljust(30, ' '), dace.config.Config.get('treps'))
        '''

        print("################################################################")
        print("########################## RUNTIMES ############################")
        print(f"*Using a batch size of {dace.config.Config.get('treps')}*")
        print("Transformation".ljust(15,' '), end='')
        for measure in self.measure_mode:
            print(measure.ljust(12,' '), end='')
        print('\n')

        for idx, (transformation_name, runtime_list) in enumerate(zip(names, runtimes)):
            print(transformation_name.ljust(15,' '), end='')
            for runtime in runtime_list:
                print(f"{runtime:.6g}".ljust(12,' '), end='')
            print('\n')

        if roofline:
            print("################################################################")
            print("######################### Operational  #########################")
            print("Transformation".ljust(15,' '),
                  "Op. Intensity".ljust(15,' '),
                  "GFLOP Median".ljust(15,' '),
                  "GFLOP Roofline")
            for transformation_name, runtime_list in zip(names, runtimes):
                if isinstance(transformation_name, str):
                    print(transformation_name.ljust(15,' '),
                          f"{roofline.data[transformation_name]:.6g}".ljust(15,' '),
                          f"{np.median(roofline.gflops_measured[transformation_name]):.6g}".ljust(15,' '),
                          f"{np.median(roofline.gflops_roof[transformation_name]):.6g}")

        print("################################################################")
        print("########################### SUMMARY ############################")
        if nan_detected:
            print("WARNING: NaN detected. See debug log for further information")

        print("Transformation".ljust(15,' '),
              "Op. Intensity".ljust(15,' ') if roofline else '',
              f"Runtime {self.measure_mode[0]}".ljust(20, ' '),
              "Verdict")

        for transformation_name, runtime_list, verdicts_dict in zip(names, runtimes, ['_'] + verdicts):
            print(transformation_name.ljust(15,' '),
                  f"{roofline.data[transformation_name]:.6g}".ljust(15,' ') if roofline else '',
                  f"{runtime_list[0]:.6g}".ljust(20,' '), end='')
            if transformation_name == 'baseline':
                print('----')
            if transformation_name != 'baseline':
                print('PASS' if all([v == 'PASS' for v in verdicts_dict.values()]) else 'FAIL')

        print("################################################################")


        if roofline:
            if self.view_roofline:
                roofline.plot(show = True, save_path = '' if self.save_roofline else None)


        for verdicts_dict in verdicts:
            if not all([v == 'PASS' for v in verdicts_dict.values()]):
                return False

        return True

    def go(self, sdfg, graph, subgraph, *symbols,
        pipeline = [expand_reduce, expand_maps, fusion],
        pipeline_args = None,
        performance_spec = dace.perf.specs.PERF_GPU_DAVINCI,
        output = [],
        name = "Runner::Go",
        report = False):
        """ Method that tests the underlying SDFG fully automatically
        """

        # build symbols dict
        symbols_dict = self.build_symbols_dict(*symbols)

        # TODO: auto infer floating point -> performance spec counter
        # construct Roofline Object
        if self.debug:
            print("Runner::Go::Constructing Roofline")
        roofline = dace.perf.Roofline(specs = performance_spec,
                                      symbols = symbols_dict,
                                      debug = self.debug,
                                      name = name)
        if self.debug:
            print("Runner::Go::Constructing Roofline")

        (input_dict, output_dict) = Runner.generate_arguments(sdfg = sdfg,
                                                              symbols_dict = symbols_dict,
                                                              outputs = output,
                                                              outputs_setzero = True)

        print("OUTPUT_DICT", len(output_dict))
        # call and go
        self.test_run(sdfg=sdfg, graph=graph, subgraph = subgraph,
                      symbols = symbols_dict,
                      inputs = input_dict, outputs = output_dict,
                      roofline = roofline, pipeline = pipeline,
                      pipeline_args = pipeline_args,
                      report = report)