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

class Runner():
    # A measurement and correctness checker for an SDFG
    # with different transformation pipelines

    def __init__(debug = True, verbose = False,
                 measure_time = True,
                 view = False, view_init = False):
        self.debug = debug
        self.verbose = verbose
        self.measure_time = measure_time
        self.view = view
        self.view_init = view_init

    def pipeline(sdfg, graph,
                 inputs, outputs,
                 subgraph = None,
                 roofline = None,
                 pipeline = [expand_reduce, expand_maps, fusion]):

        #
        name = sdfg.name
        if roofline:
            roofline.evaluate(name)

        if self.view_init:
            sdfg.view()

        return_output = None
        for output in outputs:
            if output not in inputs:
                return_output = output

        for fun in pipeline:
            # apply transformation
            fun(sdfg, graph, subgraph)
            if self.view:
                sdfg.view()
            if roofline:
                roofline.evaluate(name + fun.__name__, subgraph if subgraph else graph)
            csdfg = sdfg.compile_directly()
