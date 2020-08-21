""" This module contains classes that implement the expansion transformation.
"""

from dace import dtypes, registry, symbolic, subsets
from dace.sdfg import nodes
from dace.memlet import Memlet
from dace.sdfg import replace, SDFG, dynamic_map_inputs
from dace.transformation import pattern_matching
from dace.properties import make_properties, Property
from dace.symbolic import symstr
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.transformation.subgraph import helpers

from copy import deepcopy as dcpy
from typing import List, Union

import dace.libraries.standard as stdlib


@make_properties
class MultiExpansion(pattern_matching.SubgraphTransformation):
    ''' Implements the MultiExpansion transformation.
    Takes all the lowest scope maps in a given subgraph,
    for each of these maps splits it into an outer and inner map,
    where the outer map contains the common ranges of all maps,
    and the inner map the rest.
    Map access variables and memlets are changed accordingly
    '''

    debug = Property(dtype=bool, desc="Debug Mode", default=True)
    sequential_innermaps = Property(dtype=bool,
                                    desc="Sequential innermaps",
                                    default=False)

    @staticmethod
    def match(sdfg, subgraph) -> bool:
        ### get lowest scope maps of subgraph
        # grab first node and see whether all nodes are in the same graph
        # (or nested sdfgs therein)

        graph = subgraph.graph

        for node in subgraph.nodes():
            if node not in graph.nodes():
                return False

        # next, get all the maps
        maps = helpers.get_lowest_scope_maps(sdfg, graph, subgraph)
        brng = helpers.common_map_base_ranges(maps)

        # if leq than one map found -> fail
        if len(maps) <= 1:
            return False

        # see whether they have common parameters; if not -> fail
        if len(brng) == 0:
            return False

        return True

    def apply(self, sdfg, subgraph, map_base_variables=None):
        # get lowest scope map entries and expand

        graph = subgraph.graph

        # next, get all the base maps and expand
        maps = helpers.get_lowest_scope_maps(sdfg, graph, subgraph)
        self.expand(sdfg, graph, maps, map_base_variables=map_base_variables)

    def expand(self, sdfg, graph, map_entries, map_base_variables=None):
        """
        Expansion into outer and inner maps for each map in a specified set.
        The resulting outer maps all have same range and indices, corresponding
        variables and memlets get changed accordingly. The inner map contains
        the leftover dimensions
        :param sdfg: Underlying SDFG
        :param graph: Graph in which we expand
        :param map_entries: List of Map Entries(Type MapEntry) that we want to expand
        :param map_base_variables: Optional parameter. List of strings
                                   If None, then expand() searches for the maximal amount
                                   of equal map ranges and pushes those and their corresponding
                                   loop variables into the outer loop.
                                   If specified, then expand() pushes the ranges belonging
                                   to the loop iteration variables specified into the outer loop
                                   (For instance map_base_variables = ['i','j'] assumes that
                                   all maps have common iteration indices i and j with corresponding
                                   correct ranges)
        """

        maps = [entry.map for entry in map_entries]

        if not map_base_variables:
            # find the maximal subset of variables to expand
            # greedy if there exist multiple ranges that are equal in a map

            map_base_ranges = helpers.common_map_base_ranges(maps)
            reassignments = helpers.find_reassignment(maps, map_base_ranges)

            ##### first, regroup and reassign
            # create params_dict for every map
            # first, let us define the outer iteration variable names,
            # just take the first map and their indices at common ranges
            map_base_variables = []
            for rng in map_base_ranges:
                for i in range(len(maps[0].params)):
                    if maps[0].range[i] == rng and maps[0].params[
                            i] not in map_base_variables:
                        map_base_variables.append(maps[0].params[i])
                        break

            params_dict = {}
            if self.debug:
                print("MultiExpansion::Map_base_variables:", map_base_variables)
                print("MultiExpansion::Map_base_ranges:", map_base_ranges)
            for map in maps:
                # for each map create param dict, first assign identity
                params_dict_map = {param: param for param in map.params}
                # now look for the correct reassignment
                # for every element neq -1, need to change param to map_base_variables[]
                # if param already appears in own dict, do a swap
                # else we just replace it
                for i, reassignment in enumerate(reassignments[map]):
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
                                for key, value in params_dict_map.items():
                                    if value == target_assignment:
                                        key2 = key

                                value1 = params_dict_map[key1]
                                value2 = params_dict_map[key2]
                                params_dict_map[key1] = key2
                                params_dict_map[key2] = key1
                            else:
                                # just reassign
                                params_dict_map[current_var] = target_assignment

                # done, assign params_dict_map to the global one
                params_dict[map] = params_dict_map

            for map, map_entry in zip(maps, map_entries):
                map_scope = graph.scope_subgraph(map_entry)
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

            if self.debug:
                print("MultiExpansion::Params replaced")

        else:
            # just calculate map_base_ranges
            # do a check whether all maps correct
            map_base_ranges = []

            map0 = maps[0]
            for var in map_base_variables:
                index = map0.params.index(var)
                map_base_ranges.append(map0.range[index])

            for map in maps:
                for var, rng in zip(map_base_variables, map_base_ranges):
                    assert map.range[map.params.index(var)] == rng

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
                                             if self.sequential_innermaps \
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
                                      src_conn=edge.src_conn,
                                      memlet=edge.data,
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
                                              memlet=edge.data,
                                              src_conn=edge.src_conn,
                                              dst_conn=edge.dst_conn)

            for edge in graph.in_edges(map_exit):
                graph.remove_edge(edge)
                graph.add_memlet_path(edge.src,
                                      map_exit_inner,
                                      map_exit,
                                      memlet=edge.data,
                                      src_conn=edge.src_conn,
                                      dst_conn=edge.dst_conn)

        return
