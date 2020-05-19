from dace import dtypes, registry, symbolic, subsets
from dace.graph import nodes, nxutil
from dace.memlet import Memlet
from dace.sdfg import replace, SDFG, SDFGState
from dace.transformation import pattern_matching
from dace.properties import make_properties, Property
from dace.symbolic import symstr
from dace.graph.labeling import propagate_labels_sdfg

from copy import deepcopy as dcpy
from typing import List, Union, Dict

import dace.libraries.standard as stdlib



# ****************
# Helper functions
def find_max_permuted_outer(maps: List[nodes.Map]) \
            -> Dict[nodes.Map, List[int]]:
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
    outer_ranges = enumerate(map_range)
    for map in maps:
        result_map = []
        for i, outer_range in outer_ranges:
            found = False
            for j, current_range in enumerate(map.range):
                if current_range == outer_range and j not in result:
                    result_map.append(j)
                    found = True
            if not found:
                result_map.append(-1)

        result[map] = result_map

    return result

def dependency_dict(graph: SDFGState, maps: List[nodes.MapEntry]):
    """ from a list of top-level map entries
        returns a dict of which map depends on data from which others
        assuming that all maps in list can be fused together
    """
    # TODO

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
