# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import dace.transformation.subgraph.helpers as helpers
import dace.sdfg.nodes as nodes
import numpy as np

from dace.sdfg.graph import SubgraphView

import sys

from dace.transformation.subgraph import SubgraphFusion
from util import expand_maps, expand_reduce, fusion

from enumerator import ConnectedEnumerator, BruteForceEnumerator
from scoring_function import ExecutionScore
N, M, O = [dace.symbol(s) for s in ['N', 'M', 'O']]
N.set(50)
M.set(60)
O.set(70)

A = np.random.rand(N.get()).astype(np.float64)
B = np.random.rand(M.get()).astype(np.float64)
C = np.random.rand(O.get()).astype(np.float64)
out1 = np.ndarray((N.get(), M.get()), np.float64)
out2 = np.ndarray((1), np.float64)
out3 = np.ndarray((N.get(), M.get(), O.get()), np.float64)


@dace.program
def test_program(A: dace.float64[N], B: dace.float64[M], C: dace.float64[O], \
         out1: dace.float64[N,M], out2: dace.float64[1], out3: dace.float64[N,M,O]):

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

def enumerate(sdfg,
              graph,
              enumerator_type,
              scoring_function_type = None):
    scoring_function = scoring_function_type(sdfg, graph) if scoring_function_type else None
    condition_function = SubgraphFusion.can_be_applied if scoring_function_type else None
    enum = enumerator_type(sdfg,
                           graph,
                           condition = condition_function,
                           scoring_function = scoring_function)
    print("***************************")
    print("Enumerator Test")
    for subgraph, score in enum:
        print(score, ":", subgraph)
    print("***************************")
    print("Histogram Test")
    enum.histogram()


def test_enumerator(enumerator_type,
                    scoring_function_type,
                    view = False):
    sdfg = test_program.to_sdfg()
    sdfg.apply_strict_transformations()
    graph = sdfg.nodes()[0]
    prep(sdfg, graph)
    if view:
        sdfg.view()
    enumerate(sdfg,
              graph,
              enumerator_type,
              scoring_function_type)

if __name__ == "__main__":
    test_enumerator(ConnectedEnumerator, None, view = False)
    test_enumerator(ConnectedEnumerator, ExecutionScore, view = False)
    test_enumerator(BruteForceEnumerator, None, view = False)
    test_enumerator(BruteForceEnumerator, ExecutionScore, view = False)
