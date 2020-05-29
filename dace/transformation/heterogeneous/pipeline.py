import dace
import numpy as np

from dace.perf.roofline import Roofline
from dace.perf.specs import *
from dace.perf.optimizer import SDFGRooflineOptimizer

from dace.transformation.heterogeneous import ReduceMap
from dace.transformation.heterogeneous import SubgraphFusion
from dace.transformation.heterogeneous import MultiExpansion

import dace.libraries.standard as stdlib

import dace.sdfg.nodes as nodes

import timeit


def expand_reduce(sdfg, graph, subgraph = None):
    subgraph = graph if not subgraph else subgraph
    reduce_nodes = []
    for node in subgraph.nodes():
        if isinstance(node, stdlib.Reduce):
            reduce_nodes.append(node)

    trafo_reduce = ReduceMap(0,0,{},0)
    start = timeit.default_timer()
    for reduce_node in reduce_nodes:
        trafo_reduce.expand(sdfg,graph,reduce_node)
    end = timeit.default_timer()
    print("**** Pipeline::Reduction timer =",end-start,"s")

def expand_maps(sdfg, graph, subgraph = None):
    subgraph = graph if not subgraph else subgraph
    map_entries = get_highest_scope_maps(sdfg, graph, subgraph)

    trafo_expansion = MultiExpansion()
    start = timeit.default_timer()
    trafo_expansion.expand(sdfg, graph, map_entries)
    end = timeit.default_timer()
    print("***** Pipeline::Expansion timer =",end-start,"s")

def fusion(sdfg, graph, subgraph = None):
    subgraph = graph if not subgraph else subgraph
    map_entries = get_highest_scope_maps(sdfg, graph, subgraph)
    print(map_entries)

    map_fusion = SubgraphFusion()
    start = timeit.default_timer()
    map_fusion.fuse(sdfg, graph, map_entries)
    end = timeit.default_timer()
    print("***** Pipeline::MapFusion timer =",end-start,"s")


def combo_fuse(sdfg, graph, subgraph = None):
    # does everything at once
    # no need to recombine maps
    # TODO
    pass

def get_highest_scope_maps(sdfg, graph, subgraph = None):
    subgraph = graph if not subgraph else subgraph
    scope_dict = graph.scope_dict()
    maps = [node for node in subgraph.nodes()            \
                 if isinstance(node, nodes.MapEntry) and \
                    (not scope_dict[node] or scope_dict[node] not in subgraph.nodes())]
    return maps
