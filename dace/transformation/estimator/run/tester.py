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
from dace.transformation.subgraph.pipeline import expand_reduce, expand_maps
from dace.transformation.estimator import ConnectedEnumerator, BruteForceEnumerator, ScoringFunction, ExecutionScore, MemletScore
from dace.transformation.estimator.programs import factory
from typing import Type, List, Dict, Callable


def list_top(subgraph_scores, n=10, list_all = False):
    ''' Lists the n best subgraph scores '''
    # first do a hard sort on subgraph scores
    subgraph_scores.sort(key = lambda a: a[1])
    if list_all:
        print("********** All Subgraphs **********")
        print(subgraph_scores)
    print(f"********** Top {n} **********")
    for (i,(subgraph, runtime)) in enumerate(subgraph_scores[0:n]):
        print(f"------------{i+1}-------------")
        print("Objective:", runtime)
        print(subgraph)

def score(sdfg, graph, enumerator_type, scoring_function,
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



def test_scorer(sdfg: dace.sdfg.SDFG,
                graph: dace.sdfg.SDFGState,
                io: Dict,
                scoring_type: Type,
                enumerator_type: Type,
                view: bool = False,
                gpu: bool = False,
                transformation_function = CompositeFusion,
                condition_function = CompositeFusion.can_be_applied,
                **kwargs):
    '''
    Tests listing all subgraphs with a ScoringFunction
    '''

    scoring_func = scoring_type(
        sdfg = sdfg,
        graph = graph,
        io = io,
        gpu = gpu,
        transformation_function = transformation_function,
        ** kwargs
    )
    subgraph_list = score(sdfg, graph, enumerator_type, scoring_func,
                              condition_function)
    list_top(subgraph_list)
    return subgraph_list

def test(program_name: str,
         enumerator_type: Type,
         scoring_type: Type,
         view: bool = False,
         gpu: bool = False,
         transformation_function: Callable = CompositeFusion,
         condition_function: Callable = CompositeFusion.can_be_applied,
         **kwargs):

    # get sdfg, graph and IO
    sdfg = factory.get_program(program_name)
    if gpu:
        sdfg.apply_gpu_transformations()
    sdfg.apply_strict_transformations()
    graph = sdfg.nodes()[0]
    expand_reduce(sdfg, graph, reduce_implementation = 'pure' if not gpu else 'CUDA (block allreduce)')
    expand_maps(sdfg, graph)
    sdfg.save('program.sdfg')
    io = factory.get_args(program_name)
    test_scorer(sdfg = sdfg,
                graph = graph,
                io = io,
                enumerator_type = enumerator_type,
                scoring_type = scoring_type,
                view = view,
                gpu = gpu,
                transformation_function = transformation_function,
                condition_function = condition_function,
                **kwargs)


if __name__ == "__main__":
    program_options = [
        'synthetic', 'softmax', 'vadv'
        'hdiff', 'hdiff_mini', 'transformer', 'gemver'
    ]

    test(program_name = 'softmax',
         enumerator_type = ConnectedEnumerator,
         scoring_type = ExecutionScore,
         gpu = True,
         debug = True,
         transient_allocation = dace.dtypes.StorageType.GPU_Shared,
         schedule_innermaps = dace.dtypes.ScheduleType.GPU_ThreadBlock)
