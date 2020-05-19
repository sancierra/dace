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

    # first pass: find maximal set

    map_range = [rng for rng in map[0].range]
    for map in maps:
        tmp = [rng for rng in map.range]

        map_range_new = []
        for element in tmp:
            if element in map_range:
                map_range_new.append(element)
                map_range.remove(element)

        map_range = map_range_new

    # second pass: assign permutation
    # if range doesn't belong to outer indices,
    # put "-1" as index
    result = {map: None for map in maps}
    outer_ranges_dict = enumerate(map_range)
    for map in maps:
        result_map = []
        for i, current_range in enumerate(map.range):
            map_range_copy = map_range.copy()
            found = False
            for j, outer_range in outer_ranges_dict:
                if current_range == outer_range and j not in result_map:
                    result_map.append(j)
                    found = True
                    break
        if not found:
            result_map.append(-1)
        result[map] = result_map

    return (map_range, result)

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
