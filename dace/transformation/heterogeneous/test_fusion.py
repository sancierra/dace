import dace
from expansion import MultiExpansion
from helpers import *
import dace.graph.nodes as nodes
import numpy as np

from dace.sdfg import replace


N, M, O, P, Q, R = [dace.symbol(s) for s in ['N', 'M', 'O', 'P', 'Q', 'R']]

@dace.program
def TEST(A: dace.float64[N], B: dace.float64[M], C: dace.float64[O],
         D: dace.float64[M], E: dace.float64[N], F: dace.float64[P]):

    tmp1 = np.ndarray([N,M,O], dtype = dace.float64)
    for i, j, k in dace.map[0:N, 0:M, 0:O]:
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            in3 << C[k]
            out >> tmp1[i,j,k]

            out = in1 + in2 + in3

    tmp2 = np.ndarray([M,N,P], dtype = dace.float64)
    for j, l, k in dace.map[0:M, 0:P, 0:N]:
        with dace.tasklet:
            in1 << A[k]
            in2 << E[j]
            in3 << F[l]
            out >> tmp2[j,k,l]

            out = in1 + in2 + in3


# test multiple entries of same
@dace.program
def TEST2(A: dace.float64[N], B:dace.float64[M],
          C: dace.float64[N], D:dace.float64[M]):

    tmp1 = np.ndarray([N,M,N], dtype = dace.float64)
    for i,j,k in dace.map[0:N, 0:M, 0:N]:
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            in3 << C[k]

            out >> tmp1[i,j,k]

            out = in1+in2+in3

    tmp2 = np.ndarray([M,N], dtype = dace.float64)
    for n,m in dace.map[0:N, 0:M]:
        with dace.tasklet:
            in1 << tmp1[:,m,n]
            in2 << B[m]
            in3 << D[m]

            out >> tmp2[m,n]

            out = in1[0]*in2*in3


if __name__ == "__main__":
    N.set(50)
    M.set(60)
    O.set(70)
    P.set(80)
    Q.set(90)
    R.set(100)
    symbols = {str(var):var.get() for var in [N,M,O,P,Q,R]}

    sdfg1 = TEST.to_sdfg()
    sdfg2 = TEST2.to_sdfg()


    # first, let us test the helper functions

    #roof = dace.perf.roofline.Roofline(dace.perf.specs.PERF_CPU_DAVINCI, symbols, debug = True)
    #optimizer = dace.perf.optimizer.SDFGRooflineOptimizer(sdfg1, roof, inplace = False)
    #optimizer.optimize()


    for sdfg in [sdfg1, sdfg2]:
        print("################################################")
        sdfg.view()
        state = sdfg.nodes()[0]
        map_entries = [node for node in state.nodes() if isinstance(node, nodes.MapEntry)]
        maps = [node.map for node in state.nodes() if isinstance(node, nodes.MapEntry)]

        common_base_ranges = common_map_base_ranges(maps)
        print("COMMON BASE RANGES")
        print(common_base_ranges)


        reassignment_dict = find_reassignment(maps, common_base_ranges)
        print(reassignment_dict)


        # test transformation
        transformation = MultiExpansion()
        transformation.expand(sdfg, state, map_entries)

        # test subgraphFusion
        transformation = SubgraphFusion()
        transformation.fuse(sdfg, state, map_entries)

        sdfg.view()
