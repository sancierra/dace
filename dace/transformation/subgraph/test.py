import dace
from dace.transformation.dataflow import MapTiling, StripMining
from dace.transformation.subgraph.stencil_tiling import StencilTiling

N = dace.symbol('N')
@dace.program
def test(A: dace.float64[N,N]):
    A[:] = 1.0
    return A


if __name__ == '__main__':
    sdfg = test.to_sdfg()
    sdfg.apply_strict_transformations()
    #sdfg.view()
    graph = sdfg.nodes()[0]
    subgraph = {StencilTiling._map_entry: next(graph.nodes().index(node) \
                for node in graph.nodes() \
                if isinstance(node, dace.sdfg.nodes.MapEntry))}
    tiling = StencilTiling(0,0,subgraph, 0)
    tiling.tile_sizes = (1,1)
    tiling.stencil_size = ((-1,2),(-1,2))
    tiling.reference_range = (dace.subsets.Range.from_string('1:N-1, 1:N-1'))
    tiling.apply(sdfg)
    #sdfg.view()
