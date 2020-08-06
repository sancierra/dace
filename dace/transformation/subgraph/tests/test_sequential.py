
import dace
from dace.transformation.subgraph import MultiExpansion, SubgraphFusion, pipeline
from dace.transformation.subgraph.helpers import *
from dace.measure import Runner
import dace.sdfg.nodes as nodes
from dace.sdfg.graph import SubgraphView
import numpy as np

import sys

N = dace.symbol('N')


@dace.program
def TEST(A: dace.float64[N], B:dace.float64[N],
          C: dace.float64[N]):

    for i in dace.map[0:N]:
        with dace.tasklet:
            in1 << A[i]
            out1 >> B[i]
            out1 = in1 + 1

    for i in dace.map[0:N]:
        with dace.tasklet:
            in1 << B[i]
            out1 >> C[i]
            out1 = in1 + 1


@dace.program
def TEST2(A: dace.float64[N],
          C: dace.float64[N]):

    B = np.ndarray([N], dtype = np.float64)
    for i in dace.map[0:N]:
        with dace.tasklet:
            in1 << A[i]
            out1 >> B[i]
            out1 = in1 + 1

    for i in dace.map[0:N]:
        with dace.tasklet:
            in1 << B[i]
            out1 >> C[i]
            out1 = in1 + 1
if __name__ == "__main__":
    N.set(1000)

    sdfg1 = TEST.to_sdfg()
    state1 = sdfg1.nodes()[0]
    subgraph1 = SubgraphView(state1, [node for node in state1.nodes()])
    pipeline.fusion(sdfg1, state1)
    sdfg1.view()

    sdfg2 = TEST2.to_sdfg()
    state2 = sdfg2.nodes()[0]
    subgraph2 = SubgraphView(state2, [node for node in state2.nodes()])
    pipeline.fusion(sdfg2, state2)
    sdfg2.view()

    sys.exit(0)
    runner = Runner()
    runner.go(sdfg1, state1, None, N,
              output = ['B','C'])

    sdfg1.view()
    runner.go(sdfg2, state2, None, N,
              output = ['C'])

    sdfg2.view()
