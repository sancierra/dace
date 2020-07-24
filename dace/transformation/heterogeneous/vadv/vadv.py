import dace
from dace.sdfg import SDFG
import dace.sdfg

from dace.transformation.heterogeneous.pipeline import fusion, expand_maps, expand_reduce
from dace.transformation.interstate import StateFusion

import dace.transformation.heterogeneous as heterogeneous

vadv_unfused = SDFG.from_file('vadv-input.sdfg')
# apply state fusion exhaustively
vadv_unfused.apply_transformations_repeated(StateFusion)
vadv_unfused.view()

vadv_fused_partial = SDFG.from_file('vadv-2part.sdfg')
vadv_fused_partial.view()

vadv_fused_full = SDFG.from_file('vadv-fused.sdfg')
vadv_fused_full.view()


def test_matching():
    graph = vadv_unfused.nodes()[0]
    nodes = [node for node in graph.nodes()]
    subgraph1 = dace.sdfg.graph.SubgraphView(graph, nodes)
    subgraph2 = dace.sdfg.graph.SubgraphView(graph, [graph.nodes()[0],graph.nodes()[1]])
    expansion = heterogeneous.MultiExpansion()
    print("True  ==",expansion.match(vadv_unfused,subgraph1))
    print("False ==",expansion.match(vadv_unfused,subgraph2))


test_matching()
