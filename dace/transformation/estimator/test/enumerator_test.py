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

N, M, O = [dace.symbol(s) for s in ['N', 'M', 'O']]
N.set(50)
M.set(60)
O.set(70)

A = np.random.rand(N.get()).astype(np.float64)
B = np.random.rand(M.get()).astype(np.float64)
C = np.random.rand(O.get()).astype(np.float64)
out1 = np.zeros((N.get(), M.get()), np.float64)
out2 = np.zeros((1), np.float64)
out3 = np.zeros((N.get(), M.get(), O.get()), np.float64)


@dace.program
def test_program(A: dace.float64[N], B: dace.float64[M], C: dace.float64[O],
                 out1: dace.float64[N, M], out2: dace.float64[1],
                 out3: dace.float64[N, M, O]):

    tmp1 = np.ndarray([N, M, O], dtype=dace.float64)
    tmp2 = np.ndarray([N, M, O], dtype=dace.float64)
    tmp3 = np.ndarray([N, M, O], dtype=dace.float64)
    tmp4 = np.ndarray([N, M, O], dtype=dace.float64)
    tmp5 = np.ndarray([N, M, O], dtype=dace.float64)

    t1 = np.ndarray([N, M], dtype=dace.float64)
    t2 = np.ndarray([N, M], dtype=dace.float64)
    t3 = np.ndarray([N, M], dtype=dace.float64)

    for i, j, k in dace.map[0:N, 0:M, 0:O]:
        tp = np.ndarray([1], dtype=dace.float64)
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            in3 << C[k]
            out >> tp
            out = in1 + in2 + in3

        with dace.tasklet:
            in1 << tp
            out >> tmp1[i, j, k]
            out = in1 + 42

    dace.reduce(lambda a, b: a + b, tmp1, t1, axis=2, identity=0)

    for i, j in dace.map[0:N, 0:M]:
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            out >> t2[i, j]
            out = in1 + in2 + 42

    for i, j in dace.map[0:N, 0:M]:
        with dace.tasklet:
            in1 << t2[i, j]
            in2 << A[i]
            out >> out1[i, j]
            out = in1 * in1 * in2 + in2

    for i, j, k in dace.map[0:N, 0:M, 0:O]:
        with dace.tasklet:
            in1 << t1[i, j]
            in2 << t2[i, j]
            in3 << C[k]
            out >> tmp3[i, j, k]
            out = in1 + in2 + in3

    for i, j, k in dace.map[0:N, 0:M, 0:O]:
        with dace.tasklet:
            in1 << tmp3[i, j, k]
            in2 << tmp1[i, j, k]
            out >> out3[i, j, k]
            out = in1 + in2

    @dace.tasklet
    def fun():
        in1 << tmp3[0, 0, 0]
        out >> out2
        out = in1 * 42


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
    enum.histogram()
    return subgraph_list


def test_listing(enumerator_type, view=False, gpu=False):
    '''
    Tests listing all subgraphs without any condition funtions
    enabled
    '''
    sdfg = test_program.to_sdfg()
    sdfg.apply_strict_transformations()
    graph = sdfg.nodes()[0]
    prep(sdfg, graph)
    if view:
        sdfg.view()
    enumerate(sdfg, graph, enumerator_type, None, None)


def test_executor(enumerator_type, view=False, gpu=False):
    '''
    Tests listing all subgraphs with an ExecutionScore
    as a scoring function
    '''
    sdfg = test_program.to_sdfg()
    sdfg.apply_strict_transformations()
    #sdfg.view()
    graph = sdfg.nodes()[0]
    prep(sdfg, graph)
    if view:
        sdfg.view()
    # Define Input / Output Dict for ExecutionScore class
    # create ExecutionScore class
    inputs = {'A': A, 'B': B, 'C': C}
    outputs = {'out1': out1, 'out2': out2, 'out3': out3}
    symbols = {'N': N.get(), 'M': M.get(), 'O': O.get()}
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

    # Part I: Just list up all the subgraphs
    #test_listing(ConnectedEnumerator, view=False)
    #test_listing(BruteForceEnumerator, view=False)

    # Part II: List up all the subgraphs and execute them
    test_executor(ConnectedEnumerator, view = False)
    #test_executor(BruteForceEnumerator, view = False)