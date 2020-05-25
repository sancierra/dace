""" This module contains classes that implement the expansion transformation.
"""

from dace import dtypes, registry, symbolic, subsets
from dace.graph import nodes, nxutil
from dace.memlet import Memlet
from dace.sdfg import replace, SDFG, dynamic_map_inputs
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

    sequential_innermaps = Property(dtype = bool,
                                    desc = "Sequential innermaps",
                                    default = False)

    def expand(self, sdfg, graph, map_entries):

        """
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
        """

        maps = [entry.map for entry in map_entries]

        map_base_ranges = helpers.common_map_base_ranges(maps)
        reassignments = helpers.find_reassignment(maps, map_base_ranges)

        ##### first, regroup and reassign
        # create params_dict for every map
        # first, let us define base variables, just take the first map for that
        map_base_variables = []
        for rng in map_base_ranges:
            for i in range(len(maps[0].params)):
                if maps[0].range[i] == rng and maps[0].params[i] not in map_base_variables:
                    map_base_variables.append(maps[0].params[i])
                    break

        #map_base = {item[0]:item[1] for item in zip(map_base_variables, map_base_ranges)}

        params_dict = {}
        print("Map_base_variables", map_base_variables)
        print("Map_base_ranges", map_base_ranges)
        for map in maps:
            # for each map create param dict, first assign identity
            params_dict_map = {param: param for param in map.params}
            # now look for the correct reassignment
            # for every element neq -1, need to change param to map_base_variables[]
            # if param already appears in own dict, just do a swap
            # else we just replace it
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
            print(hex(id(map_entry)))
            params_dict_map = params_dict[map]
            for firstp, secondp in params_dict_map.items():
                if firstp != secondp:
                    replace(map_scope, firstp, '__' + firstp + '_fused')
            for firstp, secondp in params_dict_map.items():
                if firstp != secondp:
                    replace(map_scope, '__' + firstp + '_fused', secondp)

            # now also replace the map variables inside maps
            for i in range(len(map.params)):
                map.params[i] = params_dict_map[map.params[i]]

        print("PARAMS REPLACED")


        # then expand all the maps
        for map, map_entry in zip(maps, map_entries):
            if map.get_param_num() == len(map_base_variables):
                # nothing to expand, continue
                continue

            map_exit = graph.exit_node(map_entry)
            # create two new maps, outer and inner
            params_outer = map_base_variables
            ranges_outer = map_base_ranges

            init_params_inner = []
            init_ranges_inner = []
            for param, rng in zip(map.params, map.range):
                if param in map_base_variables:
                    continue
                else:
                    init_params_inner.append(param)
                    init_ranges_inner.append(rng)

            params_inner = init_params_inner
            ranges_inner = subsets.Range(init_ranges_inner)
            inner_map = nodes.Map(label = map.label + '_inner',
                                  params = params_inner,
                                  ndrange = ranges_inner,
                                  schedule = dtypes.ScheduleType.Sequential \
                                             if MultiExpansion.sequential_innermaps \
                                             else dtypes.ScheduleType.Default)

            map.label = map.label + '_outer'
            map.params = params_outer
            map.range = ranges_outer

            # create new map entries and exits
            map_entry_inner = nodes.MapEntry(inner_map)
            map_exit_inner = nodes.MapExit(inner_map)

            # analogously to Map_Expansion
            for edge in graph.out_edges(map_entry):
                graph.remove_edge(edge)
                graph.add_memlet_path(map_entry,
                                      map_entry_inner,
                                      edge.dst,
                                      src_conn = edge.src_conn,
                                      memlet = edge.data,
                                      dst_conn=edge.dst_conn)

            dynamic_edges = dynamic_map_inputs(graph, map_entry)
            for edge in dynamic_edges:
                # Remove old edge and connector
                graph.remove_edge(edge)
                edge.dst._in_connectors.remove(edge.dst_conn)

                # Propagate to each range it belongs to
                path = []
                for mapnode in [map_entry, map_entry_inner]:
                    path.append(mapnode)
                    if any(edge.dst_conn in map(str, symbolic.symlist(r))
                           for r in mapnode.map.range):
                        graph.add_memlet_path(edge.src,
                                              *path,
                                              memlet = edge.data,
                                              src_conn = edge.src_conn,
                                              dst_conn = edge.dst_conn)

            for edge in graph.in_edges(map_exit):
                graph.remove_edge(edge)
                graph.add_memlet_path(edge.src,
                                      map_exit_inner,
                                      map_exit,
                                      memlet = edge.data,
                                      src_conn = edge.src_conn,
                                      dst_conn = edge.dst_conn)

        return
