# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.transformation.subgraph.helpers as helpers
import dace.sdfg.nodes as nodes
import numpy as np

from dace.sdfg.graph import SubgraphView

import sys

from dace.transformation.subgraph import SubgraphFusion
from util import expand_maps, expand_reduce, fusion

from dace.transformation.estimator import ConnectedEnumerator, BruteForceEnumerator
from dace.transformation.estimator import ExecutionScore

from dace.transformation.estimator.programs import factory



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

    subgraph_list = enum.list(include_score = True)
    for sg in subgraph_list:
        print(sg)
    enum.histogram()
    return subgraph_list


def test_listing(program_name, enumerator_type, view=False, gpu=False):
    '''
    Tests listing all subgraphs without any condition funtions
    enabled
    '''
    # search up program
    sdfg = factory.get_program(program_name)
    sdfg.apply_strict_transformations()
    graph = sdfg.nodes()[0]
    prep(sdfg, graph)
    if view:
        sdfg.view()
    enumerate(sdfg, graph, enumerator_type, None, None)


def test_executor(program_name, enumerator_type, view=False, gpu=False):
    '''
    Tests listing all subgraphs with an ExecutionScore
    as a scoring function
    '''
    sdfg = factory.get_program(program_name)
    sdfg.apply_strict_transformations()
    #sdfg.view()
    graph = sdfg.nodes()[0]
    prep(sdfg, graph)
    if view:
        sdfg.view()
    # Define Input / Output Dict for ExecutionScore class
    # create ExecutionScore class
    io = factory.get_args(program_name)
    (inputs, outputs, symbols) = io
    scoring_func = ExecutionScore(sdfg=sdfg,
                                  graph=graph,
                                  inputs=inputs,
                                  outputs=outputs,
                                  symbols=symbols,
                                  gpu=gpu)
    condition_func = SubgraphFusion.can_be_applied
    subgraph_list = enumerate(sdfg, graph, enumerator_type, scoring_func,
                              condition_func)
    print(subgraph_list)
    print("*** Results ***")
    print("Top 10")
    for (subgraph, runtime) in sorted(subgraph_list, key=lambda a: a[1])[0:10]:
        print("-------")
        print("Runtime:", runtime)
        print(subgraph)
        print("-------")


if __name__ == "__main__":
    program_options = ['synthetic',
                       'softmax',
                       'vadv'
                       'hdiff',
                       'hdiff_mini',
                       'transformer',
                       'correlation']

    # Part I: Just list up all the subgraphs
    test_listing('softmax', ConnectedEnumerator, view = False)
    test_listing('softmax', BruteForceEnumerator, view = False)

    # Part II: List up all the subgraphs and execute them
    test_executor('synthetic', ConnectedEnumerator, view = False)
    test_executor('synthetic', BruteForceEnumerator, view = False)
