import matplotlib.pyplot as plt
import numpy as np
import dace

from dace.sdfg.graph import SubgraphView
from dace.transformation.estimator import ExecutionScore, MemletScore, RegisterScore
from dace.transformation.estimator import ConnectedEnumerator, ScoringFunction
from dace.transformation.estimator.programs import factory
from dace.transformation.subgraph.composite import CompositeFusion
from dace.properties import make_properties, Property
from dace.transformation.estimator.run.tester import test_scorer, get_sdfg, list_top

from typing import List, Type, Callable
import sys

  
def score_enum_index(sdfg: dace.SDFG,
                    graph: dace.sdfg.SDFGState,
                    iteration_index: int,
                    scoring_function: ScoringFunction,
                    gpu: bool = False,
                    enumerator_type: Type = ConnectedEnumerator,
                    **kwargs):
    
    ''' 
    Runs an SDFG through an enumerator and scores the graph at 
    * exactly * the given iteration index. 
    Enumerator is put in debug mode for reproducibility 
    ''' 
    enumerator = enumerator_type(sdfg = sdfg,
                                 graph = graph,
                                 condition_function=CompositeFusion.can_be_applied,
                                 scoring_function=None)
    enumerator.debug = True 
    enumerator.mode = 'subgraph'

    for (i, (current_subgraph, _)) in enumerate(enumerator):
        print(iteration_index, i, current_subgraph)
        if i == iteration_index:
            print(f"Iteration index {i}: Subgraph = {current_subgraph.nodes()}")
            subgraph = current_subgraph
            break 
    else:
        raise RuntimeError("Index not found!")

    # fuse manually and score 
    score = scoring_function(subgraph)
    return score 


def run_enum_index(program_name, iteration_index, gpu = False, **kwargs):

    (sdfg, graph) = get_sdfg(program_name, gpu)

    io = factory.get_args(program_name)
    
    #scoring_function = ExecutionScore(sdfg, graph, io, gpu = gpu, **kwargs)
    scoring_function = RegisterScore(sdfg, graph, io, gpu = gpu, **kwargs)
    
    print("Calculating Enum Index....")
    return score_enum_index(sdfg, graph, iteration_index, scoring_function)


if __name__ == '__main__':
    program_name = 'vadv'
    transient_allocation = dace.dtypes.StorageType.Register
    schedule_innermaps = dace.dtypes.ScheduleType.Sequential
    stencil_unroll_loops = True
    gpu = True 

    if len(sys.argv) > 1:
        print(f"Running with index input {sys.argv[1]}")
        run_enum_index(program_name = program_name, 
                       iteration_index = int(sys.argv[1]), 
                       gpu = gpu,
                       transient_allocation = transient_allocation,
                       schedule_innermaps = schedule_innermaps,
                       stencil_unroll_loops = stencil_unroll_loops)

    else:
        raise RuntimeError("Input missing.")

