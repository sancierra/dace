# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.transformation.subgraph.helpers as helpers
import dace.sdfg.nodes as nodes
import dace.dtypes as dtypes

import numpy as np
import sys

from dace.sdfg.graph import SubgraphView
from dace.transformation.subgraph import SubgraphFusion, StencilTiling
from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.estimator import ConnectedEnumerator, BruteForceEnumerator, ExecutionScore
from dace.transformation.estimator.programs import factory
from typing import Type, List


def prep(sdfg, graph):
    expand_reduce(sdfg, graph)
    expand_maps(sdfg, graph)


def enumerate(sdfg, graph, enumerator_type, scoring_function,
              condition_function):
    '''
    Enumerate all possibilities and score
    '''

    enum = enumerator_type(sdfg,
                           graph,
                           condition_function=condition_function,
                           scoring_function=scoring_function)

    subgraph_list = enum.list(include_score=True)
    for sg in subgraph_list:
        print(sg)
    enum.histogram()
    return subgraph_list


def test_listing(program_name: str,
                 enumerator_type: Type,
                 view: bool = False,
                 gpu: bool = False):
    '''
    Tests listing all subgraphs without any condition funtions
    enabled
    '''
    # search up program
    sdfg = factory.get_program(program_name)
    sdfg.apply_strict_transformations()
    if gpu:
        sdfg.apply_gpu_transformations()
    graph = sdfg.nodes()[0]
    if view:
        sdfg.view()
    enumerate(sdfg, graph, enumerator_type, None, None)


def test_executor(program_name: str,
                  enumerator_type: Type,
                  view: bool = False,
                  gpu: bool = False,
                  nruns: int = None,
                  transformation_function=CompositeFusion,
                  condition_function=CompositeFusion.can_be_applied,
                  **kwargs):
    '''
    Tests listing all subgraphs with an ExecutionScore
    as a scoring function
    '''
    sdfg = factory.get_program(program_name)
    if gpu:
        sdfg.apply_gpu_transformations()
    graph = sdfg.nodes()[0]
    if view:
        sdfg.view()
    # Define Input / Output Dict for ExecutionScore class
    # create ExecutionScore class
    io = factory.get_args(program_name)
    (inputs, outputs, symbols) = io
    scoring_func = ExecutionScore(
        sdfg=sdfg,
        graph=graph,
        inputs=inputs,
        outputs=outputs,
        symbols=symbols,
        gpu=gpu,
        nruns=nruns,
        transformation_function=transformation_function,
        **kwargs)
    subgraph_list = enumerate(sdfg, graph, enumerator_type, scoring_func,
                              condition_function)
    print(subgraph_list)
    print("*** Results ***")
    print("Top 10")
    for (subgraph, runtime) in sorted(subgraph_list, key=lambda a: a[1])[0:10]:
        print("-------")
        print("Runtime:", runtime)
        print(subgraph)
        print("-------")


if __name__ == "__main__":
    program_options = [
        'synthetic', 'softmax', 'vadv'
        'hdiff', 'hdiff_mini', 'transformer', 'correlation'
    ]

    # Part I: Just list up all the subgraphs
    '''
    test_listing('vadv',
                 ConnectedEnumerator,
                 view = False)
    test_listing('softmax',
            ¨     BruteForceEnumerator,
                 view = False)
    '''

    # Part II: List up all the subgraphs and execute them
    test_executor('softmax',
                  ConnectedEnumerator,
                  nruns=5,
                  gpu = True,
                  transient_allocation = dtypes.StorageType.GPU_Shared,
                  schedule_innermaps = dtypes.ScheduleType.GPU_ThreadBlock,
                  debug = True)
    '''
    test_executor('vadv',
                  ConnectedEnumerator,
                  nruns = 30)
    test_executor('vadv',
                  ConnectedEnumerator,
                  nruns = 30)
    test_executor('hdiff_mini',
                  ConnectedEnumerator,
                  nruns = 30)
    test_executor('hdiff_mini',
                  ConnectedEnumerator,
                  nruns = 30,
                  condition_function = CompositeFusion.can_be_applied)
    '''
