import dace
from expansion import MultiExpansion
from subgraph_fusion import SubgraphFusion
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

@dace.program
def TEST3(A: dace.float64[N], B:dace.float64[M],
          C: dace.float64[N], D:dace.float64[M]):

    tmp = np.ndarray([1], dtype = dace.float64)
    tmp1 = np.ndarray([N,M,N], dtype = dace.float64)
    tmp2 = np.ndarray([N,M,N], dtype = dace.float64)
    tmp3 = np.ndarray([N,M,N], dtype = dace.float64)
    tmp4 = np.ndarray([N,M,N,M], dtype = dace.float64)
    tmp5 = np.ndarray([N,M,N,M], dtype = dace.float64)


    for i,j,k in dace.map[0:N, 0:M, 0:N]:
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            in3 << C[k]

            out1 >> tmp1[i,j,k]
            out2 >> tmp2[i,j,k]
            out3 >> tmp3[i,j,k]

            out1 = in1+in2+in3
            out2 = in1*in2+in3
            out3 = in1*in2*in3

    for i,j,k,l in dace.map[0:N, 0:M, 0:N, 0:M]:
        with dace.tasklet:
            in1 << tmp1[i,j,k]
            in2 << D[l]
            out >> tmp4[i,j,k,l]

            out = in1*in2

    for q,r,s,t in dace.map[0:N, 0:M, 0:N, 0:M]:
        with dace.tasklet:
            in1 << B[t]
            in2 << tmp2[q,r,s]

            out >> tmp5[q,r,s,t]

            out = in1 + in2 - 42


    dace.reduce('lambda a,b: a+2*b', tmp1, tmp)



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
    sdfg3 = TEST3.to_sdfg()
    sdfg3.view()
    exit()


    # first, let us test the helper functions

    #roof = dace.perf.roofline.Roofline(dace.perf.specs.PERF_CPU_DAVINCI, symbols, debug = True)
    #optimizer = dace.perf.optimizer.SDFGRooflineOptimizer(sdfg1, roof, inplace = False)
    #optimizer.optimize()


    for sdfg in [sdfg2]:
        print("################################################")
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

        print("################################################")
        print("SubgraphFusion Test")
        # test subgraphFusion
        transformation = SubgraphFusion()
        sdfg.view()
        transformation.fuse(sdfg, state, map_entries)
        sdfg.view()
