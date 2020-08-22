import dace
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.subgraph.pipeline import fusion
from dace.sdfg.propagation import propagate_memlets_sdfg

import dace.subsets as subsets
import numpy as np


N = dace.symbol('N')
T = dace.symbol('T')
datatype = dace.float64

@dace.program
def stencil2d(A:datatype[N,N],B:datatype[N,N]):
    for t in range(T):
        @dace.map
        def a(i: _[1:N-1], j:_[1:N-1]):
            a1 << A[i, j]
            a2 << A[i, j - 1]
            a3 << A[i, j + 1]
            a4 << A[i + 1, j]
            a5 << A[i - 1, j]
            b >> B[i, j]

            b = 2*(a1+a2+a3+a4+a5)
        @dace.map
        def b(i: _[2:N-2], j:_[2:N-2]):
            a1 << B[i, j]
            a2 << B[i, j - 1]
            a3 << B[i, j + 1]
            a4 << B[i + 1, j]
            a5 << B[i - 1, j]
            b >> A[i, j]

            b = 2*(a1+a2+a3+a4+a5)

def init_array(A):
    n = N.get()
    for i in range(n):
        for j in range(n):
            A[i,j] = datatype(i * j + 2)/n


def pre_tiling(sdfg, graph, tile_size = 64, tile_offsets = (1,1), sequential = False):

    for node in graph.nodes():
        if isinstance(node, dace.sdfg.nodes.MapEntry):
            if node.label == 'a':
                entry1 = node
            if node.label == 'b':
                entry2 = node
        if isinstance(node, dace.sdfg.nodes.AccessNode) and node.data == 'B':
            node.setzero = True

    print(f"Operating on graph {graph}")
    print(f"with map entries {entry1} and {entry2}")

    d1 = {MapTiling._map_entry: graph.nodes().index(entry1)}
    d2 = {MapTiling._map_entry: graph.nodes().index(entry2)}

    t1 = dace.transformation.dataflow.tiling.MapTiling(sdfg.sdfg_id,
         sdfg.nodes().index(graph), d1, 0)
    t2 = dace.transformation.dataflow.tiling.MapTiling(sdfg.sdfg_id,
         sdfg.nodes().index(graph), d2, 0)

    t1.strides    = (tile_size, tile_size)
    t1.tile_sizes = (tile_size + 2, tile_size + 2) ## !!
    t1.strides_offset = tile_offsets

    t2.strides    = (tile_size, tile_size)
    t2.tile_sizes = (tile_size, tile_size)

    t1.apply(sdfg)
    t2.apply(sdfg)

    if sequential:
        for node in graph.nodes():
            if isinstance(node,dace.sdfg.nodes.MapEntry) and (node.label == 'a' or node.label == 'b'):
                node.map.schedule = dace.dtypes.ScheduleType.Sequential


def evaluate(sdfg, graph, view = False, compile = False):
    result = None
    if view:
        sdfg.view()
    if compile:
        A = np.zeros([N.get(), N.get()], dtype = np.float64)
        B = np.zeros([N.get(), N.get()], dtype = np.float64)
        init_array(A)
        print(np.linalg.norm(A))
        csdfg = sdfg.compile()
        csdfg(A=A,N=N,T=T,B=B)
        print(np.linalg.norm(A))
        print(np.linalg.norm(B))
    result = A if compile else None
    return result


def run(sdfg, tile_size, view = True, compile = False, gpu = False, sequential = False):
    sdfg.apply_strict_transformations()
    graph = None
    for g in sdfg.nodes():
        if len(g.nodes()) == 9:
            graph = g
            break

    if gpu:
        sdfg.apply_gpu_transformations(options={'sequential_innermaps': False})
        sdfg.apply_strict_transformations()
        for g in sdfg.nodes():
            if len(g.nodes()) > 5:
                graph = g

    # establish baseline
    R1 = evaluate(sdfg, graph, view, compile)

    pre_tiling(sdfg, graph, tile_size, sequential = sequential)
    R2 = evaluate(sdfg, graph, view, compile)

    fusion(sdfg, graph)
    R3 = evaluate(sdfg, graph, view, compile)

    if compile:
        #print(R1)
        #print(R2)
        assert np.allclose(R1,R2)
        assert np.allclose(R1,R3)



if __name__ == '__main__':
    sdfg = stencil2d.to_sdfg()
    #propagate_memlets_sdfg(sdfg)
    sdfg.propagate = False
    TILE_SIZE = 8
    N.set(20)
    T.set(1)
    sdfg.specialize({'N':N})
    run(sdfg, TILE_SIZE, compile = True, gpu = False, view = False, sequential = True)
