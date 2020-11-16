import matplotlib.pyplot as plt
import numpy as np
import dace

from dace.sdfg.graph import SubgraphView
from dace.transformation.estimator import ExecutionScore, MemletScore
from dace.transformation.estimator import ConnectedEnumerator
from dace.transformation.estimator.programs import factory
from dace.transformation.subgraph.composite import CompositeFusion

from dace.transformation.estimator.run import test_scorer

from typing import List, Type, Callable

def obtain_subgraphs(program_name: str,
                     score1: Type,
                     score2: Type,
                     gpu: bool = False,
                     enumerator_type: Type = ConnectedEnumerator,
                     **kwargs):
    ''' Run and Score Subgraphs with two
        different Scoring Functions
    '''
    sdfg = factory.get_program(program_name)
    if gpu:
        sdfg.apply_gpu_transformations()
    graph = sdfg.nodes()[0]
    io = factory.get_args(program_name)

    subgraphs = []
    for score in (score1, score2):
        r = test_scorer(sdfg = sdfg,
                        graph = graph,
                        io = io,
                        scoring_type = score,
                        enumerator_type = enumerator_type,
                        gpu = gpu,
                        **kwargs)
        subgraphs.append(r)

    return tuple(subgraphs)

def correlation_plot(program_name: str,
                     score1: Type,
                     score2: Type,
                     enumerator_type: Type = ConnectedEnumerator,
                     gpu: bool = False,
                     nruns: int = 30,
                     transformation_function: Type = CompositeFusion,
                     condition_function: Callable = CompositeFusion.can_be_applied,
                     save: bool = False,
                     view: bool = True,
                     save_name: str = 'correlation',
                     **kwargs):
    '''
    Creates a Correlation Plot of two ScoringFunctions
    on a graph of choice
    '''
    # get subgraphs
    r = obtain_subgraphs(program_name = program_name,
                         score1 = score1,
                         score2 = score2,
                         enumerator_type = enumerator_type,
                         gpu = gpu,
                         nruns = nruns,
                         transformation_function = transformation_function,
                         condition_function = condition_function,
                         **kwargs)

    datapoints = list()
    # subgraphs to analyze
    (s1, s2) = r
    for (subgraph1, score1) in s1:
        subgraph1 = set(subgraph1)
        for (subgraph2, score2) in s2:
            print([n in subgraph1 for n in subgraph2])
            if all([n in subgraph1 for n in subgraph2]) and \
               len(subgraph1) == len(subgraph2):
                datapoints.append((score1, score2))

    x = list(d[0] for d in datapoints)
    y = list(d[1] for d in datapoints)

    plt.plot(x, y, 'o')

    m,b = np.polyfit(x,y,1)
    corr = np.corrcoef(x,y)
    plt.plot(x, [m*e + b for e in x])
    plt.annotate(f"m={m}, c={corr}", (0,0))

    if view:
        plt.show()
    if save:
        plt.save(save_name)



if __name__ == '__main__':
    correlation_plot('softmax',
                     ExecutionScore,
                     MemletScore)
