""" This module contains classes that implement the expansion transformation.
"""

from dace import dtypes, registry, symbolic, subsets
from dace.graph import nodes, nxutil
from dace.memlet import Memlet
from dace.sdfg import replace, SDFG
from dace.transformation import pattern_matching
from dace.properties import make_properties, Property
from dace.symbolic import symstr
from dace.graph.labeling import propagate_labels_sdfg

from copy import deepcopy as dcpy
from typing import List, Union

import dace.libraries.standard as stdlib

import helpers


@make_properties
class MultiExpansion():
    def __init__(self, debug = True):
        self.debug = debug
        pass



    def expand(self, sdfg, graph, subgraph = None):
        if not subgraph:
            subgraph = graph
        maps = []
        map_entries = []
        scope_dict = graph.scope_dict()
        toplevel_scope = helpers.toplevel_scope(graph, subgraph, scope_dict)
        # gather all top-level maps
        for node in subgraph:
            if isinstance(node, nodes.MapEntry) \
            and scope_dict[node] == toplevel_scope:
                    maps.append(node.map)
                    map_entries.append(node)

        map_base_ranges = helpers.common_map_base_ranges(maps)
        reassignments = helpers.find_reassignment(maps, map_base_ranges)

        ##### first, regroup and reassign
        # create params_dict for every map
        # first, let us define base variables, just take the first map for that
        map_base_variables = []
        for i in range(len(map[0].params)):
            if map[0].range[i] in map_base_ranges:
                map_base_variables.append(map[0].params[i])
        map_base = {item[0]:item[1] for item in zip(map_base_variables, map_base_ranges)}

        params_dict = {}
        for map in maps:
            # for each map create param dict, first assign identity
            params_dict_map = {param: param for param in map.params}
            # now look for the correct reassignment
            # for every element neq -1, need to change param to map_base_variables[]
            # if param already appears in own dict, just do a swap
            # else we just replace it
            # TODO
            for i, reassignment in enumerate(reassignments[map]):
                # 0:-1, 1:-1, 2:1, 3:0, 4:-1
                if reassignment == -1:
                    # nothing to do
                    pass
                else:
                    current_var = map.params[i]
                    current_assignment = params_dict_map[current_var]
                    target_assignment = map_base_variables[reassignment]
                    if current_assignment != target_assignment:
                        if target_assignment in params_dict_map.values():
                            # do a swap
                            key1 = current_var
                            # get the corresponding key, cumbersome
                            for key, value in params_dict_map.items():
                                if value == target_assignment:
                                    key2 = key

                            value1 = params_dict_map[key1]
                            value2 = params_dict_map[key2]
                            params_dict_map[key1] = key2
                            params_dict_map[key2] = key1
                        else:
                            # just reassign - noone cares
                            params_dict_map[current_var] = target_assignment

            # done, assign params_dict_map to the global one
            params_dict[map] = params_dict_map

        for map, map_entry in zip(maps, map_entries):
            map_scope = graph.scope_subgraph(map_entry)
            params_dict_map = params_dict[map]
            for firstp, secondp in params_dict_map.items():
                if firstp != secondp:
                    replace(map_scope, secondp, '__' + secondp + '_fused')
            for firstp, secondp in params_dict_map.items():
                if firstp != secondp:
                    replace(map_scope, '__' + secondp + '_fused', firstp)

        # then expand all the maps
        for map, map_entry in zip(maps, map_entries):
            if map.get_param_num() == len(map_base_variables):
                # nothing to expand, continue
                continue

            map_exit = graph.exit_node(map_entry)
            # create two new maps, outer and inner
            
