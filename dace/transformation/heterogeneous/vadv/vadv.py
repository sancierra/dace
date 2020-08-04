import dace
from dace.sdfg import SDFG
import dace.sdfg

from dace.transformation.heterogeneous.pipeline import fusion, expand_maps, expand_reduce
from dace.transformation.interstate import StateFusion, NestSDFG
from dace.transformation.helpers import nest_state_subgraph

import dace.transformation.heterogeneous as heterogeneous

from dace.measure import Runner

import itertools

vadv_unfused = SDFG.from_file('vadv-input-fixed.sdfg')
# apply state fusion exhaustively
vadv_unfused.apply_transformations_repeated(StateFusion)
for node in vadv_unfused.nodes()[0].nodes():
    if isinstance(node, dace.sdfg.nodes.MapEntry):
        node.map.schedule = dace.dtypes.ScheduleType.Default
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

def test_fuse_all(view = True):
    if view:
        vadv_unfused.view()
    graph = vadv_unfused.nodes()[0]
    subgraph = dace.sdfg.graph.SubgraphView(graph, [node for node in graph.nodes()])
    fusion = heterogeneous.SubgraphFusion()
    fusion.apply(vadv_unfused, subgraph)
    if view:
        vadv_unfused.view()

def test_fuse_partial(view = False):
    if view:
        vadv_unfused.view()
    set1 = set()
    set2 = set()
    graph = vadv_unfused.nodes()[0]
    for node in graph.nodes():
        for edge in itertools.chain(graph.out_edges(node), graph.in_edges(node)):
            if isinstance(edge.dst, (dace.sdfg.nodes.MapEntry, dace.sdfg.nodes.MapExit)):
                if '0' in edge.dst.map.label or '1' in edge.dst.map.label or '2' in edge.dst.map.label:
                    set1.add(node)
                if '3' in edge.dst.map.label or '4' in edge.dst.map.label:
                    set2.add(node)
            if isinstance(edge.src, (dace.sdfg.nodes.MapEntry, dace.sdfg.nodes.MapExit)):
                if '0' in edge.src.map.label or '1' in edge.src.map.label or '2' in edge.src.map.label:
                    set1.add(node)
                if '3' in edge.src.map.label or '4' in edge.src.map.label:
                    set2.add(node)
            if isinstance(node, (dace.sdfg.nodes.MapEntry, dace.sdfg.nodes.MapExit)):
                if '0' in node.map.label or '1' in node.map.label or '2' in node.map.label:
                    set1.add(node)
                if '3' in node.map.label or '4' in node.map.label:
                    set2.add(node)

    subgraph1 = dace.sdfg.graph.SubgraphView(graph, list(set1))
    subgraph2 = dace.sdfg.graph.SubgraphView(graph, list(set2))
    print(list(set1))
    print(list(set2))
    fusion = heterogeneous.SubgraphFusion()
    fusion.apply(vadv_unfused, subgraph1)
    fusion.apply(vadv_unfused, subgraph2)
    vadv_unfused.view()

def test_fuse_all_numerically():
    graph = vadv_unfused.nodes()[0]
    runner = Runner(sequential = True)
    I = dace.symbol('I')
    J = dace.symbol('J')
    K = dace.symbol('K')
    I.set(50); J.set(55); K.set(60)
    runner.go(vadv_unfused, graph, None,
              I,J,K,
              pipeline = [fusion],
              output = ['utens_stage', 'data_col'])

#view_all()
#test_matching()
#test_fuse_all()
test_fuse_all_numerically()
#test_fuse_partial()





#test_fuse_all_state_push()
def test_fuse_all_state_push(view = True):
    # NOTE: this does not work as expected with
    # current StateFusion -- too weak
    # can just inline back
    if view:
        vadv_unfused.view()
    graph = vadv_unfused.nodes()[0]
    # in every map, push everything into states
    scope_dict = graph.scope_dict(node_to_children=True)
    print("SCOPE_DICT", scope_dict)

    map_entries = heterogeneous.helpers.get_lowest_scope_maps(vadv_unfused, graph)

    for parent in scope_dict.keys():
        if parent in map_entries:
            print("PARENT", parent)
            node_list = scope_dict[parent].copy()
            print("NODE LIST", node_list)
            node_list.remove(graph.exit_node(parent))
            print("NODE LIST", node_list)
            test = graph.scope_tree()
            subgraph = dace.sdfg.graph.SubgraphView(graph, node_list)
            nest_state_subgraph(vadv_unfused, graph, subgraph)

    subgraph = dace.sdfg.graph.SubgraphView(graph, [node for node in graph.nodes()])
    fusion = heterogeneous.SubgraphFusion()
    fusion.apply(vadv_unfused, subgraph)
    vadv_unfused.apply_transformations_repeated(NestSDFG)
    if view:
        vadv_unfused.view()
