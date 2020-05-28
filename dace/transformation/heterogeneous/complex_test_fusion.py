import dace
from expansion import MultiExpansion
from subgraph_fusion import SubgraphFusion
from reduce_map import ReduceMap
from helpers import *
import dace.graph.nodes as nodes
import numpy as np

from dace.sdfg import replace


N, M, O, P, Q, R = [dace.symbol(s) for s in ['N', 'M', 'O', 'P', 'Q', 'R']]
N.set(50)
M.set(60)
O.set(70)
P.set(80)
Q.set(90)
R.set(100)


@dace.program
def TEST(A: dace.float64[N], B: dace.float64[M], C: dace.float64[O], Z1: dace.float64[N,M],
         D: dace.float64[N], E: dace.float64[M], F: dace.float64[O], Z2: dace.float64[N,M]):

    tmp1 = np.ndarray([N,M,O], dtype = dace.float64)
    tmp2 = np.ndarray([N,M,O], dtype = dace.float64)
    tmp3 = np.ndarray([N,M,O], dtype = dace.float64)
    tmp4 = np.ndarray([N,M,O], dtype = dace.float64)

    t1 = np.ndarray([N,M], dtype = dace.float64)
    t2 = np.ndarray([N,M], dtype = dace.float64)
    t3 = np.ndarray([N,M], dtype = dace.float64)

    for i, j, k in dace.map[0:N, 0:M, 0:O]:
        tp = np.ndarray([1], dtype = dace.float64)
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            in3 << C[k]
            out >> tp

            out = in1 + in2 + in3

        with dace.tasklet:
            in1 << tp
            out >> tmp1[i,j,k]

            out = in1 + 42

    dace.reduce(lambda a,b: a+b, tmp1, t1, axis = 2, identity = 0)

    for i,j in dace.map[0:N, 0:M]:
        with dace.tasklet:
            in1 << A[i]
            in2 << B[j]
            out >> t2[i,j]
            out = in1 + in2 + 42

    for i,j in dace.map[0:N, 0:M]:
        with dace.tasklet:
            in1 << t2[i,j]
            in2 << A[i]
            out >> t3[i,j]

            out = in1*in1*in2 + in2

    for i,j,k in dace.map[0:N, 0:M, 0:O]:
        with dace.tasklet:
            in1 << t1[i,j]
            in2 << t2[i,j]
            in3 << C[k]
            out >> tmp3[i,j,k]

            out = in1 + in2 + in3

    for i,j,k in dace.map[0:N, 0:M, 0:O]:
        with dace.tasklet:
            in1 << tmp3[i,j,k]
            in2 << tmp1[i,j,k]
            out >> tmp4[i,j,k]

            out = in1 + in2




if __name__ == "__main__":

    symbols = {str(var):var.get() for var in [N,M,O,P,Q,R]}

    sdfg = TEST.to_sdfg()

    sdfg.apply_strict_transformations()
    sdfg.view()

    # first, let us test the helper functions

    #roof = dace.perf.roofline.Roofline(dace.perf.specs.PERF_CPU_DAVINCI, symbols, debug = True)
    #optimizer = dace.perf.optimizer.SDFGRooflineOptimizer(sdfg1, roof, inplace = False)
    #optimizer.optimize()

    graph = sdfg.nodes()[0]
    map_entries = [node for node in graph.nodes() if isinstance(node, nodes.MapEntry)]
    maps = [node.map for node in map_entries]
    reduce_nodes = [node for node in graph.nodes() if isinstance(node, dace.libraries.standard.Reduce)]

    print("**** ReduceMap Test")
    transformation = ReduceMap(0,0,{},0)
    for reduce_node in reduce_nodes:
        transformation.expand(sdfg, graph, reduce_node)
        map_entries.append(transformation._outer_entry)


    # test transformation
    print("**** MultiExpansion Test")
    transformation = MultiExpansion()
    #transformation.expand(sdfg, graph, map_entries)
    transformation.expand(sdfg, graph, map_entries)
    print("Done.")
    sdfg.view()

    print("**** SubgraphFusion Test")
    transformation = SubgraphFusion()
    #exit()
    transformation.fuse(sdfg, graph, map_entries)
    print("Done")
    sdfg.view()
    sdfg.validate()
    print("VALIDATION PASS")
