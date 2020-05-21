import dace
from expansion import MultiExpansion
from helpers import *
import dace.graph.nodes as nodes
import numpy as np

from dace.sdfg import replace


N, M, O, P, Q, R = [dace.symbol(s) for s in ['N', 'M', 'O', 'P', 'Q', 'R']]

@dace.program
def TEST(A: dace.float64[N], B: dace.float64[M], C: dace.float64[O],
         D: dace.float64[M], E: dace.float64[N], F: dace.float64[P],
         G: dace.float64[M], H: dace.float64[P], I: dace.float64[N], J: dace.float64[R],
         X: dace.float64[N], Y: dace.float64[M], Z: dace.float64[P]):

    tmp1 = np.ndarray([N,M,O], dtype = dace.float64)
    for i, j, k in dace.map[0:N, 0:M, 0:O]:
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            in3 << C[k]
            out >> tmp1

            out1 = in1 + in2 + in3

    tmp2 = np.ndarray([M,N,P], dtype = dace.float64)
    for j, k, l in dace.map[0:M, 0:N, 0:P]:
        with dace.tasklet:
            in1 << D[k]
            in2 << E[j]
            in3 << F[l]
            out >> tmp2

            out = in1 + in2 + in3

    tmp3 = np.ndarray([M,P,N,R,O], dtype = dace.float64)
    for asdf, asdf2, asdf3, asdf4 in dace.map[0:M, 0:P, 0:N, 0:R]:
        with dace.tasklet:
            in1 << G[asdf]
            in2 << H[asdf2]
            in3 << I[asdf3]
            in4 << J[asdf4]

            out >> tmp3

            out = in1 + in2 + in3 + in4

    tmp4 = np.ndarray([N,M,P], dtype = dace.float64)
    for i,j,k in dace.map[0:N, 0:M, 0:P]:
        with dace.tasklet:
            in1 << X[i]
            in2 << Y[j]
            in3 << Z[k]
            out >> tmp4

            out = in1 + in2 + in3

# test multiple entries of same
@dace.program
def TEST2(A: dace.float64[N], AA:dace.float64[N], B:dace.float64[M], BB:dace.float64[M],
          C: dace.float64[N], CC:dace.float64[N], D:dace.float64[M], DD:dace.float64[M]):

    tmp1 = np.ndarray([N,N,M], dtype = dace.float64)
    for i,j,k in dace.map[0:N, 0:N, 0:M]:
        with dace.tasklet:
            in1 << A[i]
            in2 << AA[j]
            in3 << B[k]

            out >> tmp1[i,j,k]

            out = in1+in2+in3

    tmp2 = np.ndarray([M,M,N,N], dtype = dace.float64)
    for n,m,o,p in dace.map[0:M, 0:M, 0:N, 0:N]:
        with dace.tasklet:
            in1 << D[n]
            in2 << DD[m]
            in3 << C[o]
            in4 << CC[p]

            out >> tmp2[n,m,o,p]

            out = in1+in2+in3+in4


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


        # next up, test transformation
        transformation = MultiExpansion()


        transformation.expand(sdfg, state, map_entries)



        sdfg.view()
