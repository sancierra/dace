from dace import dtypes, registry, symbolic, subsets
from dace.graph import nodes, nxutil
from dace.memlet import Memlet
from dace.sdfg import replace, SDFG, SDFGState
from dace.transformation import pattern_matching
from dace.properties import make_properties, Property
from dace.symbolic import symstr
from dace.graph.labeling import propagate_labels_sdfg

from copy import deepcopy as dcpy
from typing import List, Union, Dict, Tuple

import dace.libraries.standard as stdlib



# ****************
# Helper functions
def find_max_permuted_outer(maps: List[nodes.Map]) \
            -> Tuple[List[Ranges], Dict[nodes.Map, List[int]]]:
    """ Finds maximum permuted map base
        Input: all maps in the subgraphs
        Output: For every map, returns an array
        indicating which position

        FORNOW: Individual Ranges have to be equal
    """

def common_map_base_ranges(maps: List[nodes.Map]) -> List:
    """ Finds maximally extended common map base.
    Map base = set of ranges that every map in the list has
    """
    if len(maps) == 0:
        return None
    # first pass: find maximal set
    map_base = [rng for rng in map[0].range]
    for map in maps:
        tmp = [rng for rng in map.range]

        map_base_new = []
        for element in tmp:
            if element in map_base:
                map_base_new.append(element)
                map_base.remove(element)

        map_base = map_base_new


    return map_base


def find_reassignment(maps: List[nodes.Map], map_base_ranges) -> Dict[nodes.Map, List]:
    """ Provided an outer map base,
    finds a reassignment so that ranges get properly mapped to
    a respective base index. If there is none, we put -1 as index
    """
    result = {map: None for map in maps}
    outer_ranges_dict = enumerate(map_base)
    # 0: 0:N, 1: 0:N, 2: 0:M     |    0: 0:K, 1: 0:M, 2: 0:N, 3: 0:F, 4: 0:N
    for map in maps:
        result_map = []
        for i, current_range in enumerate(map.range):
            found = False
            for j, outer_range in outer_ranges_dict:
                if current_range == outer_range and j not in result_map:
                    result_map.append(j)
                    found = True
                    break
        if not found:
            result_map.append(-1)
        result[map] = result_map

    return result

def dependency_dict(graph: SDFGState, maps: List[nodes.MapEntry]):
    """ from a list of top-level map entries
        returns a dict of which map depends on data from which others
        assuming that all maps in list can be fused together

        All maps in the list should be on the same scope
        in order for this function to yields something useful
    """
    # TODO
    # graph.source_nodes(), entry magic

def non_connected_graph(*args):
    """ Generate non-connected graph part
        To test whether this leads to the desired behavior in pattern matching
    """
    path = gr.OrderedDiGraph()
    if len(args) == 1 and isinstance(args[0], list):
        input_nodes = args[0]

    else:
        input_nodes = list(args)

    path.add_nodes_from(input_nodes)
    return path


def path_from_to(node1, node2, graph):
    # BFS
    queue = [node1]
    while len(queue) > 0:
        current = queue.pop(0)
        if current == node2:
            return True
        else:
            queue.extend([edge.dst for edge in graph.out_edges(current))

    return False

def toplevel_scope_subgraph(graph, subgraph, scope_dict = None):
    """ returns the toplevel scope of the subgraph"""
    if not scope_dict:
        scope_dict = graph.scope_dict()
    scopes = set()
    for element in subgraph:
        scopes.add(scope_dict[element])
    for scope in scopes:
        # search the one whose parent is not in scopes
        # that must be the top level one
        if scope_dict[scope] not in scopes:
            return scope

    raise RuntimeError("Subgraph is not sound (must be connected)")

def toplevel_scope_maps(graph, maps, scope_dict = None):
    if not scope_dict:
        scope_dict = graph.scope_dict()
    scopes = set()
    for map in maps:
        scopes.add(scope_dict[map])
    for scope in scopes:
        if scope_dict[scope] not in scopes:
            return scope

    raise RuntimeError("Map structure is not sound (underlying subgraph must be complete and connected")
