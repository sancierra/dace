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
    print(enumerator.list())
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


def run_enum_index(program_name, iteration_index, gpu = False):

    (sdfg, graph) = get_sdfg(program_name, gpu)

    io = factory.get_args(program_name)
    scoring_function = ExecutionScore(sdfg, graph, io, gpu = gpu)
    return score_enum_index(sdfg, graph, iteration_index, scoring_function)


if __name__ == '__main__':
    run_enum_index('softmax', 0, False)

