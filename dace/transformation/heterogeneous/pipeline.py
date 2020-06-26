import dace
import numpy as np


from dace.transformation.heterogeneous import ReduceMap
from dace.transformation.heterogeneous import SubgraphFusion
from dace.transformation.heterogeneous import MultiExpansion

from typing import Union, List

import dace.libraries.standard as stdlib

import dace.sdfg.nodes as nodes
from dace.sdfg.graph import SubgraphView

import timeit

TRANSFORMATION_TIMER = True

"""
#################
Usual Pipeline:
    - Expand all desired Reduce Nodes using ReduceMap transformation
      (source found in reduce_map.py)
    - Expand all maps (including previously expanded Reduces)
      into outer and inner maps using MultiExpansion
      (source found in expansion.py)
    - Perform SubgraphFusion and fuse into one global outer map
      whose params and ranges have been determined by MultiExpansion
      (source found in subgraph_fusion.py)
#################
"""



def expand_reduce(sdfg: dace.SDFG,
                  graph: dace.SDFGState,
                  subgraph: Union[SubgraphView, List[SubgraphView]] = None,
                  **kwargs):
    """
    Perform a ReduceMap transformation of all the Reduce Nodes specified in the
    subgraph. If for a reduce node transformation cannot be done, a warning is omitted.
    After a successful transformation and if subgraph(s) is/are specified, the outer map and
    reduce are added back

    :param sdfg: SDFG
    :param graph: SDFGState of interest
    :param subgraph: If None, the whole state gets transformed.
                     If SubgraphView, all the Reduces therein are considered
                     If List of SubgraphViews, all the Reduces in all the
                     SubgraphViews are considered
    :param kwargs: Property setters for ReduceMap class

    """
    subgraph = graph if not subgraph else subgraph
    if not isinstance(subgraph, List):
        subgraph = [subgraph]

    if TRANSFORMATION_TIMER:
        start = timeit.default_timer()

    for sg in subgraph:
        reduce_nodes = []
        for node in sg.nodes():
            if isinstance(node, stdlib.Reduce):
                if not ReduceMap.can_be_applied(graph = graph,
                                                candidate = {ReduceMap._reduce: node},
                                                expr_index = 0,
                                                sdfg = sdfg):
                    print(f"WARNING: Cannot expand reduce node {node}: \
                            Can_be_applied() failed.")
                    continue
                reduce_nodes.append(node)

        trafo_reduce = ReduceMap(0,0,{},0)
        for (property, val) in kwargs.items():
            setattr(trafo_reduce, property, val)

        start = timeit.default_timer()
        for reduce_node in reduce_nodes:
            trafo_reduce.expand(sdfg,graph,reduce_node)
            if isinstance(sg, SubgraphView):
                sg.nodes().remove(reduce_node)
                sg.nodes().append(trafo_reduce._new_reduce)
                sg.nodes().append(trafo_reduce._outer_entry)

    if TRANSFORMATION_TIMER:
        end = timeit.default_timer()
        print("**** Pipeline::Reduction timer =",end-start,"s")

def expand_maps(sdfg: dace.SDFG,
                graph: dace.SDFGState,
                subgraph: Union[SubgraphView, List[SubgraphView]] = None,
                **kwargs):

    """
    Perform MultiExpansion on all the Map nodes specified.

    :param sdfg: SDFG
    :param graph: SDFGState of interest
    :param subgraph: If None, all top-level maps in the graph get expanded
                     If SubgraphView, all top-level maps in the SubgraphView get expanded
                     (the corresponding MapEntry has to be in the subgraph!)
                     If a list of SubgraphViews, all top-level maps in their SubgraphViews
                     get expanded (corresponding MapEntry has to be in one of the subgraphs)
    :param kwargs: Property setters for MultiExpansion

    """
    subgraph = graph if not subgraph else subgraph
    if not isinstance(subgraph, List):
        subgraph = [subgraph]

    trafo_expansion = MultiExpansion()
    for (property, val) in kwargs.items():
        setattr(trafo_expansion, property, val)

    if TRANSFORMATION_TIMER:
        start = timeit.default_timer()

    for sg in subgraph:
        map_entries = get_highest_scope_maps(sdfg, graph, sg)
        start = timeit.default_timer()
        trafo_expansion.expand(sdfg, graph, map_entries)

    if TRANSFORMATION_TIMER:
        end = timeit.default_timer()
        print("***** Pipeline::Expansion timer =",end-start,"s")

def fusion(sdfg: dace.SDFG,
           graph: dace.SDFGState,
           subgraph: Union[SubgraphView, List[SubgraphView]] = None,
           **kwargs):
    """
    Perform MapFusion on the graph/subgraph/subgraphs specified

    :param sdfg: SDFG
    :param graph: SDFGState of interest
    :param subgraph: if None, performs SubgraphFusion on
                     all the top-level maps in the graph
                     if SubgraphView, performs SubgraphFusion on
                     all the top-level maps in the SubgraphView
                     if List of SubgraphViews, performs SubgraphFusion
                     on each of these Subgraphs, using the respective
                     top-level maps for fusion
    :param kwargs: Property setters for SubgraphFusion
    """

    subgraph = graph if not subgraph else subgraph
    if not isinstance(subgraph, List):
        subgraph = [subgraph]

    map_fusion = SubgraphFusion()
    for (property, val) in kwargs.items():
        setattr(map_fusion, property, val)

    if TRANSFORMATION_TIMER:
        start = timeit.default_timer()

    for sg in subgraph:
        map_entries = get_highest_scope_maps(sdfg, graph, sg)
        # remove map_entries and their corresponding exits from the subgraph
        # already before applying transformation
        if isinstance(sg, SubgraphView):
            for map_entry in map_entries:
                sg.nodes().remove(map_entry)
                if graph.exit_node(map_entry) in sg.nodes():
                    sg.nodes().remove(graph.exit_node(map_entry))
        print(f"Subgraph Fusion on map entries {map_entries}")
        map_fusion.fuse(sdfg, graph, map_entries)
        if isinstance(sg, SubgraphView):
            sg.nodes().append(map_fusion._global_map_entry)
            #TODO: also add all created in_between transients...

    if TRANSFORMATION_TIMER:
        end = timeit.default_timer()
        print("***** Pipeline::MapFusion timer =",end-start,"s")

def go(sdfg: dace.SDFG,
       graph: dace.SDFGState,
       subgraph: Union[SubgraphView, List[SubgraphView]] = None):
    """ Does it all at once.
        Pipelines ReduceExpansion -> MultiExpansion -> SubgraphFusion

        :param sdfg: SDFG
        :param graph: SDFGState of interest
        :param subgraph: If None, performs pipeline on graph.
                         If SubgraphView, expands and fuses on this subgraph
                         If List of SubgraphViews, expands and fuses on
                         each of those subgraphs individually
    """
    expand_reduce(sdfg, graph, subgraph)
    expand_maps(sdfg, graph, subgraph)
    fusion(sdfg, graph, subgraph)

def get_highest_scope_maps(sdfg, graph, subgraph = None):
    subgraph = graph if not subgraph else subgraph
    scope_dict = graph.scope_dict()

    def is_highest_scope(node):
        while scope_dict[node]:
            if scope_dict[node] in subgraph.nodes():
                return False
            node = scope_dict[node]

        return True

    maps = [node for node in subgraph.nodes() if isinstance(node, nodes.MapEntry)
                                              and is_highest_scope(node)]

    return maps
