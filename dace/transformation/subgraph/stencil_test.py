import dace
from dace.sdfg import SDFG
from dace.transformation.subgraph.stencil_tiling_new import StencilTiling
from dace.sdfg.graph import SubgraphView

def test():
    sdfg = SDFG.from_file('vadv/hdiff_merged.sdfg')
    graph = sdfg.nodes()[0]
    subgraph = SubgraphView(graph, graph.nodes())

    st = StencilTiling(subgraph)
    return_value = st.match(sdfg, subgraph)
    assert return_value == True
    print("OK")
    st.apply(sdfg)
    print("OK")


if __name__ == '__main__':
    test()
