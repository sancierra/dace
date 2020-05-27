import dace
import numpy as np

from dace.perf.roofline import Roofline
from dace.perf.specs import *
from dace.perf.optimizer import SDFGRooflineOptimizer

from dace.transformation.heterogeneous import ReduceMap
from dace.transformation.heterogeneous import SubgraphFusion
from dace.transformation.heterogeneous import MultiExpansion

import dace.libraries.standard as stdlib

import timeit

dace_dtype = dace.float32
H, B, SN, SM = (dace.symbol(s) for s in ('H', 'B', 'SN', 'SM'))


@dace.program
def softmax(X_in: dace_dtype[H, B, SN, SM]):
    tmp_max = dace.reduce(lambda a, b: max(a, b), X_in, axis=3)
    tmp_out = np.ndarray([H, B, SN, SM], dtype=dace_dtype)
    out = np.ndarray([H, B, SN, SM], dtype=dace_dtype)

    # No broadcasting rules
    for i, j, k, l in dace.map[0:H, 0:B, 0:SN, 0:SM]:
        with dace.tasklet:
            inp << X_in[i, j, k, l]
            mx << tmp_max[i, j, k]
            o >> tmp_out[i, j, k, l]
            o = math.exp(inp - mx)
    #tmp_out = np.exp(X_in - tmp_max)

    tmp_sum = dace.reduce(lambda a, b: a + b, tmp_out, identity=0, axis=3)
    for i, j, k, l in dace.map[0:H, 0:B, 0:SN, 0:SM]:
        with dace.tasklet:
            inp << tmp_out[i, j, k, l]
            sm << tmp_sum[i, j, k]
            o >> out[i, j, k, l]
            o = inp / sm

    return out


if __name__ == '__main__':
    sdfg = softmax.to_sdfg()
    sdfg.view()
    roofline = Roofline(PERF_GPU_DAVINCI, symbols = {H:30, B:30, SN:300, SM:300})
    graph = sdfg.nodes()[0]

    ################ first, expand the reduce node
    reduce_nodes = []
    for node in graph.nodes():
        if isinstance(node, stdlib.Reduce):
            reduce_nodes.append(node)

    start = timeit.timeit()
    for reduce_node in reduce_nodes:
        trafo_reduce = ReduceMap(sdfg_id = sdfg.sdfg_list.index(sdfg),
                                 state_id = 0,
                                 subgraph = {ReduceMap._reduce: graph.nodes().index(reduce_node)},
                                 expr_index = 0)
        trafo_reduce.apply(sdfg)
    end = timeit.timeit()
    print("***** Reduction timer =",end-start,"s")
    sdfg.view()

    ############### second, do MultiExpansion
    trafo_expansion = MultiExpansion()
    map_entries = []
    for node in graph.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            map_entries.append(node)
    start = timeit.timeit()
    trafo_expansion.expand(sdfg, graph, map_entries)
    end = timeit.timeit()
    print("***** Expansion timer =",end-start,"s")

    sdfg.view()


    ############ third, do MapFusion
    map_fusion = SubgraphFusion()
    start = timeit.timeit()
    map_fusion.fuse(sdfg, graph, map_entries)
    end = timeit.timeit()
    print("***** MapFusion timer =",end-start,"s")

    sdfg.apply_strict_transformations()
    sdfg.view()
    sdfg.validate()
