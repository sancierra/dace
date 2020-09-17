import dace
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.subgraph.pipeline import fusion
from dace.transformation.subgraph.stencil_tiling import StencilTiling
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.sdfg.graph import SubgraphView

import dace.subsets as subsets
import numpy as np
import sys

N = dace.symbol('N')
T = dace.symbol('T')
datatype = dace.float64

@dace.program
def stencil2d_nontransient(A:datatype[N,N],B:datatype[N,N],C:datatype[N,N]):
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
        b >> C[i, j]

        b = 2*(a1+a2+a3+a4+a5)

@dace.program
def stencil2d_transient(A:datatype[N,N], C:datatype[N,N]):
    B = np.ndarray([N,N], dtype = datatype)
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
        b >> C[i, j]

        b = 2*(a1+a2+a3+a4+a5)

def init_array(A):
    n = N.get()
    for i in range(n):
        for j in range(n):
            A[i,j] = i*j/(n*n)

def stencil_tiling(sdfg, graph, tile_size = (1,1), gpu = False, sequential = False):
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

    subgraph = SubgraphView(graph, graph.nodes())
    t = StencilTiling(subgraph, sdfg.sdfg_id,
                      sdfg.nodes().index(graph))

    t.strides = (tile_size[0], tile_size[1])
    t.apply(sdfg)

    if gpu:
        for node in graph.nodes():
            if isinstance(node,dace.sdfg.nodes.MapEntry) and node.label in ['a','b']:
                node.map.schedule = dace.dtypes.ScheduleType.Default

def pre_tiling(sdfg, graph, tile_size = (1,1), tile_offsets = (0,0), gpu = False, sequential = False):

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

    t1.strides    = (tile_size[0], tile_size[1])
    t1.tile_sizes = (tile_size[0]+2, tile_size[1]+2) ## !!
    t1.tile_offset = (0,0)#tile_offsets

    t2.strides    = (tile_size[0], tile_size[1])
    t2.tile_sizes = (tile_size[0], tile_size[1])
    t2.tile_offset = (0,0)

    t1.apply(sdfg)
    t2.apply(sdfg)

    if gpu:
        for node in graph.nodes():
            if isinstance(node,dace.sdfg.nodes.MapEntry) and node.label in ['a','b']:
                node.map.schedule = dace.dtypes.ScheduleType.Default


def evaluate(sdfg, graph, view = False, compile = False):
    result = None
    if view:
        sdfg.view()
    if compile:
        A = np.zeros([N.get(), N.get()], dtype = np.float64)
        B = np.zeros([N.get(), N.get()], dtype = np.float64)
        C = np.zeros([N.get(), N.get()], dtype = np.float64)

        init_array(A)
        print(np.linalg.norm(C))
        csdfg = sdfg.compile()
        csdfg(A=A,N=N,T=T,B=B, C=C)
        print(np.linalg.norm(A))
    result = C if compile else None
    return result


def run(tile_size, view = True, compile = False, gpu = False, sequential = False, transient = True):
    if transient:
        sdfg = stencil2d_transient.to_sdfg()
    else:
        sdfg = stencil2d_nontransient.to_sdfg()
    #sdfg.specialize({'N':N})
    #propagate_memlets_sdfg(sdfg)
    sdfg.propagate = False
    #sdfg.specialize({'N':N})
    sdfg.apply_strict_transformations()

    graph = None
    for g in sdfg.nodes():
        if len(g.nodes()) == 9:
            graph = g
            break

    if gpu:
        sdfg.apply_gpu_transformations(options={'sequential_innermaps': sequential})
        sdfg.apply_strict_transformations()
        for g in sdfg.nodes():
            if len(g.nodes()) > 5:
                graph = g

    # establish baseline
    R1 = evaluate(sdfg, graph, view, compile)

    stencil_tiling(sdfg, graph, tile_size, sequential = sequential, gpu = gpu)
    R2 = evaluate(sdfg, graph, view, compile)

    if gpu:
        fusion(sdfg, graph, transient_allocation = dace.dtypes.StorageType.GPU_Shared if not sequential else dace.dtypes.StorageType.Register, sequential_innermaps = sequential)
    else:
        fusion(sdfg, graph, sequential_innermaps = sequential)
    R3 = evaluate(sdfg, graph, view, compile)

    if compile:
        print(np.linalg.norm(R1))
        print(np.linalg.norm(R2))
        print(np.linalg.norm(R3))
        assert np.allclose(R1,R2)
        assert np.allclose(R1,R3)
        print('PASS')

if __name__ == '__main__':
    try:
        mode = sys.argv[1]
        assert mode == 'register' or mode == 'shared'
        tile1 = int(sys.argv[2])
        tile2 = int(sys.argv[3])
    except:
        print("Useage: mode tile1 tile2")
        raise RuntimeError
    
    tiles = (tile1, tile2)
    N.set(512)
    T.set(1)
    run(tiles, compile = True, gpu = True, view = False, sequential = mode == 'register', transient = True)
