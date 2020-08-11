import dace
from dace.transformation.dataflow.tiling import MapTiling
from dace.transformation.subgraph.pipeline import fusion
from dace.sdfg.propagation import propagate_memlets_sdfg

import dace.subsets as subsets

try:
    import polybench
except ImportError:
    polybench = None

N = dace.symbol('N')
tsteps = dace.symbol('tsteps')

#datatypes = [dace.float64, dace.int32, dace.float32]
datatype = dace.float64

# Dataset sizes
sizes = [{
    tsteps: 20,
    N: 30
}, {
    tsteps: 40,
    N: 90
}, {
    tsteps: 100,
    N: 250
}, {
    tsteps: 500,
    N: 1300
}, {
    tsteps: 1000,
    N: 2800
}]
args = [
    ([N, N], datatype),
    ([N, N], datatype)  #, N, tsteps
]


@dace.program(datatype[N, N], datatype[N, N])  #, dace.int32, dace.int32)
def jacobi2d(A, B):  #, N, tsteps):
    for t in range(tsteps):

        @dace.map
        def a(i: _[1:N - 1], j: _[1:N - 1]):
            a1 << A[i, j]
            a2 << A[i, j - 1]
            a3 << A[i, j + 1]
            a4 << A[i + 1, j]
            a5 << A[i - 1, j]
            b >> B[i, j]

            b = 0.2 * (a1 + a2 + a3 + a4 + a5)

        @dace.map
        def b(i: _[1:N - 1], j: _[1:N - 1]):
            a1 << B[i, j]
            a2 << B[i, j - 1]
            a3 << B[i, j + 1]
            a4 << B[i + 1, j]
            a5 << B[i - 1, j]
            b >> A[i, j]

            b = 0.2 * (a1 + a2 + a3 + a4 + a5)


def init_array(A, B):  #, N, tsteps):
    n = N.get()
    for i in range(n):
        for j in range(n):
            A[i, j] = datatype(i * (j + 2) + 2) / n
            B[i, j] = datatype(i * (j + 3) + 3) / n



def pre_tiling(sdfg, tile_size = 64):
    graph = None
    for g in sdfg.nodes():
        if len(g.nodes()) == 9:
            graph = g
            break

    for node in graph.nodes():
        if isinstance(node, dace.sdfg.nodes.MapEntry):
            if node.label == 'a':
                entry1 = node
            if node.label == 'b':
                entry2 = node

    # hack: modify B[i,j] memlet to B[i-1,j-1]
    for node in graph.nodes():
        if isinstance(node, dace.sdfg.nodes.CodeNode):
            if node.label == 'a':
                edge = graph.out_edges(node)[0]
                edge.data.subset.offset(subsets.Indices.from_string('1,1'), negative = True)
    propagate_memlets_sdfg(sdfg)

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

    t2.strides    = (tile_size, tile_size)
    t2.tile_sizes = (tile_size, tile_size)

    t1.apply(sdfg)
    t2.apply(sdfg)


if __name__ == '__main__':
    #polybench.main(sizes, args, [(0, 'A')], init_array, jacobi2d)
    sdfg = jacobi2d.to_sdfg()
    sdfg.apply_strict_transformations()
    sdfg.view()
    pre_tiling(sdfg)
    sdfg.view()
