import dace
from dace.sdfg import SDFG
import dace.sdfg

from dace.transformation.heterogeneous.pipeline import fusion, expand_maps, expand_reduce
from dace.transformation.interstate import StateFusion

import dace.transformation.heterogeneous as heterogeneous

vadv_unfused = SDFG.from_file('vadv-input.sdfg')
# apply state fusion exhaustively
vadv_unfused.apply_transformations_repeated(StateFusion)
vadv_fused_partial = SDFG.from_file('vadv-2part.sdfg')
vadv_fused_full = SDFG.from_file('vadv-fused.sdfg')

def view_all():
    vadv_unfused.view()
    vadv_fused_partial.view()
    vadv_fused_full.view()


def test_matching():
    graph = vadv_unfused.nodes()[0]
    # should yield True
    subgraph = dace.sdfg.graph.SubgraphView(graph, [node for node in graph.nodes()])
    expansion = heterogeneous.MultiExpansion()
    print("True  ==", heterogeneous.MultiExpansion.match(vadv_unfused,subgraph))
    fusion = heterogeneous.SubgraphFusion()
    print("True  ==", heterogeneous.SubgraphFusion.match(vadv_unfused,subgraph))

def test_fuse_all():
    graph = vadv_unfused.nodes()[0]
    subgraph = dace.sdfg.graph.SubgraphView(graph, [node for node in graph.nodes()])
    fusion = heterogeneous.SubgraphFusion()
    fusion.apply(vadv_unfused, subgraph)
    vadv_unfused.view()
view_all()
#test_matching()
test_fuse_all()
