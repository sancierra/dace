# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This file implements the ExecutionScore class """

from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.dataflow import DeduplicateAccess
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

from dace.transformation.estimator.movement_counter import count_moved_data_state
from dace.transformation.estimator.movement_counter import count_moved_data_state_composite
from dace.transformation.estimator.movement_counter import count_moved_data_subgraph

import dace.sdfg.propagation as propagation
import dace.symbolic as symbolic
import dace.dtypes as dtypes
import sympy

from typing import Dict, Type

from dace.transformation.estimator.scoring import ScoringFunction

import warnings
import os
import numpy as np
import sys


@make_properties
class MemletScore(ScoringFunction):
    '''
    Evaluation policy that uses outermost scoped data movement 
    as a performance proxy. Scorer returns 
    estimated traffic in bytes runtime or ratio thereof.
    '''

    propagate_all = Property(desc="Do a final propagation step of each graph"
                             "before evaluation.",
                             dtype=bool,
                             default=True)

    deduplicate = Property(desc="Do a deduplication step"
                           "after applying the transformation function.",
                           dtype=bool,
                           default=False)

    exit_on_error = Property(desc="Perform sys.exit if error occurs during"
                                  "estimation, else return -1",
                             dtype=bool,
                             default=False)

    run_baseline = Property(desc = "Estimate traffic in baseline SDFG that"
                                   "gets passed upon initialization. Score returned"
                                   "is then calculated as a fraction compared to baseline"
                                   "traffic.",
                            dtype = bool,
                            default = False)

    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 device: dtypes.DeviceType = dtypes.DeviceType.CPU,
                 transformation_function: Type = CompositeFusion,
                 transformation_properties: Dict = None,
                 io: Dict = None):
        '''
        :paramsdfg: SDFG of interest 
        :param grpah: State of interest 
        :param subgraph: Subgraph 
        :param io: Tuple of (input_dict, output_dict, symbols_dict), where each 
                   of these dictionaries contains a mapping from  argument/symbol name 
                   to its corresponding argument. If None, IO a
        
        '''
        super().__init__(sdfg=sdfg,
                         graph=graph,
                         device = device,
                         transformation_function=transformation_function,
                         transformation_properties=transformation_properties,
                         io = io) 
                        

        # inputs and outputs not needed
        self._outputs, self._inputs = None, None

        # estimate traffic on base sdfg
        # apply deduplication on baseline graph if necessary
        if self.deduplicate:
            # need to copy
            sdfg_copy = SDFG.from_json(self._sdfg.to_json())
            graph_copy = sdfg_copy.nodes()[self._state_id]
            sdfg_copy.apply_transformations_repeated(DeduplicateAccess,
                                                     states=[graph_copy])
            self._base_traffic = self.estimate_traffic(sdfg_copy, graph_copy)
        else:
            # use directly
            self._base_traffic = self.estimate_traffic(sdfg, graph)


    def symbolic_evaluation(self, term):
        '''
        Evaluate a term using self._symbols. Throws a TypeError
        if evaluation cannot be performed numerically.
        '''
        if isinstance(term, (int, float)):
            return term 
        # take care of special functions appearing in term and resolve those
        x, y = sympy.symbols('x y')
        rounding_funcs = {
            sympy.Function('int_floor'):
            sympy.Lambda((x, y),
                         sympy.functions.elementary.integers.floor(x / y)),
            sympy.Function('int_ceil'):
            sympy.Lambda((x, y),
                         sympy.functions.elementary.integers.ceiling(x / y)),
            sympy.Function('floor'):
            sympy.Lambda((x), sympy.functions.elementary.integers.floor(x)),
            sympy.Function('ceiling'):
            sympy.Lambda((x), sympy.functions.elementary.integers.ceiling(x)),
        }
        for fun, lam in rounding_funcs.items():
            term.replace(fun, lam)
        try:
            result = symbolic.evaluate(term, self._symbols)
        except TypeError:
            missing_symbols = set() 
            for sym in term.symbols:
                if sym not in self._symbols:
                    missing_symbols.add(sym) 
            warnings.warn(f"MemletScore::Cannot evaluate{term}. Missing arguments: {list(missing_symbols)}")
            raise TypeError(f"Missing Symbols to evaluate {term}")
        result = int(result)
        return result

    def estimate_traffic(self, sdfg, graph):
        traffic = 0
        try:
            # get traffic count
            traffic_symbolic = count_moved_data_state_composite(graph)
            # evaluate w.r.t. symbols
            traffic = self.symbolic_evaluation(traffic_symbolic)
            if traffic == 0:
                warnings.warn("MemletScore:: Measured traffic is 0")
        except Exception as e:
            warnings.warn(f"MemletScore:: Exception occurred: {e}")
            if isinstance(e, TypeError):
                if self.exit_on_error:
                    sys.exit(0)
            else:
                raise e
        return traffic

    def score(self, subgraph: SubgraphView):
        '''
        Applies transformation to the (deepcopied) SDFG / graph and 
        outputs a numerical value for outermost scoped data movement volume. 
        If a baseline is established, it outputs a fraction of movement 
        compared to baseline traffic. 
        '''

        sdfg_copy = SDFG.from_json(self._sdfg.to_json())
        graph_copy = sdfg_copy.nodes()[self._state_id]
        subgraph_copy = SubgraphView(graph_copy, [
            graph_copy.nodes()[self._graph.nodes().index(n)] for n in subgraph
        ])

        transformation_function = self._transformation(subgraph_copy)
        # assign properties to transformation
        for (arg, val) in self._kwargs.items():
            try:
                setattr(transformation_function, arg, val)
            except AttributeError:
                warnings.warn(f"MemletScore:: Attribute {arg} not found in transformation")
            except (TypeError, ValueError):
                warnings.warn(f"MemletScore:: Attribute {arg} has invalid value {val}")

        transformation_function.apply(sdfg_copy)
        if self.deduplicate:
            sdfg_copy.apply_transformations_repeated(DeduplicateAccess,
                                                     states=[graph_copy])
        if self.propagate_all or self.deduplicate:
            propagation.propagate_memlets_scope(sdfg_copy, graph_copy,
                                                graph_copy.scope_leaves())

        current_traffic = self.estimate_traffic(sdfg_copy, graph_copy)
        return current_traffic / self._base_traffic

    @staticmethod
    def name():
        return "Estimated Memlet Traffic"
