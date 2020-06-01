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

import pipeline

dace_dtype = dace.float32
H, B, SN, SM = (dace.symbol(s) for s in ('H', 'B', 'SN', 'SM'))


@dace.program
def softmax(X_in: dace_dtype[H, B, SN, SM], TEST: dace_dtype[H, B, SN]):
    tmp_max = dace.reduce(lambda a, b: max(a, b), X_in, axis=3, identity = 0)
    TEST[:] = tmp_max[:]

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

def expand_reduce(sdfg, graph):
    reduce_nodes = []
    for node in graph.nodes():
        if isinstance(node, stdlib.Reduce):
            reduce_nodes.append(node)

    trafo_reduce = ReduceMap(0,0,{},0)
    start = timeit.default_timer()
    for reduce_node in reduce_nodes:
        trafo_reduce.expand(sdfg,graph,reduce_node)
    end = timeit.default_timer()
    print("***** Reduction timer =",end-start,"s")

def expand_maps(sdfg, graph):
    trafo_expansion = MultiExpansion()
    map_entries = []
    for node in graph.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            map_entries.append(node)
    start = timeit.default_timer()
    trafo_expansion.expand(sdfg, graph, map_entries)
    end = timeit.default_timer()
    print("***** Expansion timer =",end-start,"s")

def subgraph_fusion(sdfg, graph, map_entries):
    map_fusion = SubgraphFusion()
    start = timeit.default_timer()
    map_fusion.fusion(sdfg, graph, map_entries)
    end = timeit.default_timer()
    print("***** MapFusion timer =",end-start,"s")



sdfg = softmax.to_sdfg()
roofline = Roofline(PERF_GPU_DAVINCI, symbols = {H:5, B:5, SN:100, SM:100})
graph = sdfg.nodes()[0]
H.set(10); B.set(10); SN.set(40); SM.set(40)

def test_graph():
    ################ first, expand the reduce node
    pipeline.expand_reduce(sdfg, graph)
    sdfg.view()

    ############### second, do MultiExpansion
    pipeline.expand_maps(sdfg, graph)
    sdfg.view()

    ############ third, do MapFusion
    pipeline.fusion(sdfg, graph)

    sdfg.apply_strict_transformations()
    sdfg.view()

    sdfg.validate()

def test_result():
    X_in = np.random.rand(H.get(), B.get(), SN.get(), SM.get()).astype(np.float32)

    print("X_in")
    print(X_in)

    TEST = np.zeros([H.get(), B.get(), SN.get()], dtype = np.float32)

    X_out_baseline = np.zeros([H.get(), B.get(), SN.get(), SM.get()], dtype = np.float32)
    X_out_1 = np.zeros([H.get(), B.get(), SN.get(), SM.get()], dtype = np.float32)
    X_out_2 = np.zeros([H.get(), B.get(), SN.get(), SM.get()], dtype = np.float32)
    X_out_3 = np.zeros([H.get(), B.get(), SN.get(), SM.get()], dtype = np.float32)

    csdfg = sdfg.compile_directly()
    X_out_baseline = csdfg(X_in = X_in, TEST=TEST, H=H, B=B, SN=SN, SM=SM)
    print("X_out_baseline")
    print(X_out_baseline)
    print("TEST")
    print(TEST)
    #sdfg.view()


    pipeline.expand_reduce(sdfg, graph)
    #sdfg.view()

    csdfg = sdfg.compile_directly()
    X_out_1 = csdfg(X_in = X_in, TEST=TEST, H=H, B=B, SN=SN, SM=SM)
    print("X_out_1")
    print(X_out_1)
    print("TEST")
    print(TEST)

    pipeline.expand_maps(sdfg, graph)
    #sdfg.view()

    csdfg = sdfg.compile_directly()
    X_out_2 = csdfg(X_in = X_in, TEST=TEST, H=H, B=B, SN=SN, SM=SM)
    print("X_out_2")
    print(X_out_2)
    print("TEST")
    print(TEST)

    pipeline.fusion(sdfg, graph)
    #sdfg.view()

    csdfg = sdfg.compile_directly()
    X_out_3 = csdfg(X_in = X_in, TEST=TEST, H=H, B=B, SN=SN, SM=SM)
    print("X_out_3")
    print(X_out_3)
    print("TEST")
    print(TEST)

    sdfg.apply_strict_transformations()
    #sdfg.view()

    print("Diff1", np.linalg.norm(X_out_baseline - X_out_1))
    print("Diff2", np.linalg.norm(X_out_baseline - X_out_2))
    print("Diff3", np.linalg.norm(X_out_baseline - X_out_3))
    print(np.linalg.norm(X_out_baseline))
    print("#######")
    print(np.linalg.norm(X_out_1))
    print(np.linalg.norm(X_out_2))
    print(np.linalg.norm(X_out_3))
    print("#######")
    #print("BASELINE:")
    #print(X_out_baseline)
    #print("O1")
    #print(X_out_1)
    #print("O2")
    #print(X_out_2)
    #print("O3")
    #print(X_out_3)

if __name__ == "__main__":
    test_result()
