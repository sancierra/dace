""" This module contains classes and functions that implement the orthogonal
    stencil tiling transformation. """
"""
TODO:
    - Stencil detection improvement                       [OK]
        - write function to detect from memlets           [OK]
        - write map offset function [Not Necessary]
    - Fix ignore cases and prettify strip_mining section  [OK]
      [UGLY]
    - Make ready for PR                                   [OK]
    - Write a unittest                                    [OK]
    - Extend can_be_applied by OUTER RANGE CHECK          [TODO]
    - Check whether sink map condition is enough          [TODO]

"""
import math

import dace
from dace import registry, symbolic
from dace.properties import make_properties, Property, ShapeProperty
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import pattern_matching
from dace.symbolic import pystr_to_symbolic, simplify_ext
from dace.subsets import Range
from dace.sdfg.propagation import _propagate_node

from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.dataflow.map_expansion import MapExpansion
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace.transformation.interstate.loop_detection import DetectLoop

from copy import deepcopy as dcpy

import dace.subsets as subsets
import dace.symbolic as symbolic

import itertools
import warnings

from collections import defaultdict

from dace.transformation.subgraph import helpers


@registry.autoregister_params(singlestate=True)
@make_properties
class StencilTiling(pattern_matching.SubgraphTransformation):
    """ Implements the stencil tiling transformation.

        Operates on top level maps of the given subgraph.
        Applies orthogonal tiling to each of the maps with
        the given strides and extends the newly created
        inner tiles to account for data dependencies
        due to stencil patterns. For each map all outgoing
        memlets to an array must cover the memlets that
        are incoming into a following child map.

        All maps must have the same map parameters in
        the same order.
    """

    # Properties
    debug = Property(desc="Debug mode", dtype=bool, default=True)

    prefix = Property(dtype=str,
                      default="stencil",
                      desc="Prefix for new inner tiled range symbols")

    strides = ShapeProperty(dtype=tuple, default=(1, ), desc="Tile stride")

    schedule = Property(dtype=dace.dtypes.ScheduleType,
                        default=dace.dtypes.ScheduleType.Default,
                        desc="Dace.Dtypes.ScheduleType of Inner Maps")

    unroll_loops = Property(desc="Unroll Inner Loops if they have Size > 1",
                            dtype=bool,
                            default=False)

    @staticmethod
    def coverage_dicts(sdfg, graph, map_entry, outer_range=True):
        '''
        returns a tuple of two dicts:
        the first dict has as a key all data entering the map
        and its associated access range
        the second dict has as a key all data exiting the map
        and its associated access range
        if outer_range = True, substitutes outer ranges
        into min/max of inner access range
        '''
        map_exit = graph.exit_node(map_entry)
        map = map_entry.map

        entry_coverage = {}
        exit_coverage = {}
        # create dicts with which we can replace all iteration
        # variable_mapping by their range
        map_min = {
            dace.symbol(param): e
            for param, e in zip(map.params, map.range.min_element())
        }
        map_max = {
            dace.symbol(param): e
            for param, e in zip(map.params, map.range.max_element())
        }

        # look at inner memlets at map entry
        for e in graph.out_edges(map_entry):
            if outer_range:
                # get subset
                min_element = [
                    m.subs(map_min) for m in e.data.subset.min_element()
                ]
                max_element = [
                    m.subs(map_max) for m in e.data.subset.max_element()
                ]
                # create range
                rng = subsets.Range(
                    (min_e, max_e, 1)
                    for min_e, max_e in zip(min_element, max_element))
            else:
                rng = dcpy(e.data.subset)

            if e.data.data not in entry_coverage:
                entry_coverage[e.data.data] = rng
            else:
                old_coverage = entry_coverage[e.data.data]
                entry_coverage[e.data.data] = subsets.union(old_coverage, rng)

        # look at inner memlets at map exit
        for e in graph.in_edges(map_exit):
            if outer_range:
                # get subset
                min_element = [
                    m.subs(map_min) for m in e.data.subset.min_element()
                ]
                max_element = [
                    m.subs(map_max) for m in e.data.subset.max_element()
                ]
                # craete range
                rng = subsets.Range(
                    (min_e, max_e, 1)
                    for min_e, max_e in zip(min_element, max_element))
            else:
                rng = dcpy(e.data.subset)

            if e.data.data not in exit_coverage:
                exit_coverage[e.data.data] = rng
            else:
                old_coverage = exit_coverage[e.data]
                exit_coverage[e.data.data] = subsets.union(old_coverage, rng)

        # return both coverages as a tuple
        return (entry_coverage, exit_coverage)

    @staticmethod
    def can_be_applied(sdfg: SDFG, subgraph: SubgraphView) -> bool:
        # get highest scope maps
        graph = subgraph.graph
        map_entries = set(helpers.get_highest_scope_maps(sdfg, graph, subgraph))
        if len(map_entries) < 1:
            return False

        # all parameters have to be the same
        # we do not want any permutations here as this gets too messy
        params = dcpy(next(iter(map_entries)).map.params)
        for map_entry in map_entries:
            if map_entry.map.params != params:
                return False

        # check whether all map entries only differ by a const amount
        first_entry = next(iter(map_entries))
        for map_entry in map_entries:
            for r1, r2 in zip(map_entry.map.range, first_entry.map.range):
                if len((r1[0] - r2[0]).free_symbols) > 0:
                    return False
                if len((r1[1] - r2[1]).free_symbols) > 0:
                    return False


        # get source maps as a starting point for BFS
        source_nodes = set(sdutil.find_source_nodes(graph))
        maps_reachable_source = set()
        sink_maps = set()
        while len(source_nodes) > 0:
            # traverse and find source maps
            node = source_nodes.pop()
            if isinstance(node, nodes.MapEntry) and node in map_entries:
                maps_reachable_source.add(node)
            else:
                for e in graph.out_edges(node):
                    source_nodes.add(e.dst)

        # main loop: traverse graph and check whether topologically
        # connected maps cover each other in terms of memlets
        maps_queued = set(maps_reachable_source)
        maps_processed = set()
        coverages = {}
        for map_entry in map_entries:
            coverages[map_entry] = StencilTiling.coverage_dicts(
                sdfg, graph, map_entry)

        while len(maps_queued) > 0:
            maps_added = 0
            current_map = maps_queued.pop()
            if current_map in maps_processed:
                continue

            nodes_following = set([e.dst for e in graph.out_edges(current_map)])
            while len(nodes_following) > 0:
                current_node = nodes_following.pop()
                if isinstance(current_node,
                              nodes.MapEntry) and current_node in map_entries:
                    if current_node not in (maps_processed | maps_queued):
                        maps_queued.add(current_node)
                    maps_added += 1
                    # now check whether subsets cover
                    # get dict of incoming edges of this inner loop map
                    in_dict = coverages[current_node][0]
                    # get dict of outgoing edges of map in the
                    # preceding outer loop map
                    out_dict = coverages[current_map][1]
                    for data_name in in_dict:
                        # check if data is in out_dict
                        if data_name in out_dict:
                            # check coverage: if not covered, we can't continue
                            if not out_dict[data_name].covers(
                                    in_dict[data_name]):
                                return False

                else:
                    for e in graph.out_edges(current_node):
                        nodes_following.add(e.dst)

            if maps_added == 0:
                sink_maps.add(current_map)

            maps_processed.add(current_map)

        # last condition: we want all sink maps to have the same
        # range, if not we cannot form a common map range for fusion later
        # last condition: we want all incoming memlets into sink maps
        # to have the same ranges

        assert len(sink_maps) > 0
        first_sink_map = next(iter(sink_maps))
        if not all([
                map.range.size() == first_sink_map.range.size()
                for map in sink_maps
        ]):
            return False

        return True

    def calculate_stripmining_parameters(self, current_range, reference_range,
                                         stride):
        """ Calculates parameters for stripmining and offsetting
            based on the current range and a reference range
            (corresponding range in one of the )
            Iff successful (transformation posible), returns True
            Saves all parameters as class variable
        """
        map_stride = reference_range.strides()[0]
        if current_range.strides()[0] != reference_range.strides()[0]:
            return False
        min_diff = symbolic.SymExpr(reference_range.min_element()[0] \
                        - current_range.min_element()[0])
        max_diff = symbolic.SymExpr(current_range.max_element()[0] \
                        - reference_range.max_element()[0])

        try:
            min_diff = symbolic.evaluate(min_diff, {})
            max_diff = symbolic.evaluate(max_diff, {})
        except TypeError:
            # cannot evaluate to integer
            # abort mission
            return False

        if min_diff < 0 or max_diff < 0:
            return False

        self.tile_sizes.append(stride + max_diff + min_diff)
        self.tile_offset_lower.append(pystr_to_symbolic(str(min_diff)))
        self.tile_offset_upper.append(pystr_to_symbolic(str(max_diff)))

        return True

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        subgraph = self.subgraph_view(sdfg)
        map_entries = helpers.get_highest_scope_maps(sdfg, graph, subgraph)

        # first get dicts of parents and children for each map_entry
        # get source maps as a starting point for BFS
        # these are all map entries reachable from source nodes
        source_nodes = set(sdutil.find_source_nodes(graph))
        maps_reachable_source = set()
        sink_maps = set()
        while len(source_nodes) > 0:
            # traverse and find source maps
            node = source_nodes.pop()
            if isinstance(node, nodes.MapEntry) and node in map_entries:
                maps_reachable_source.add(node)
            else:
                for e in graph.out_edges(node):
                    source_nodes.add(e.dst)

        # traverse graph
        maps_queued = set(maps_reachable_source)
        maps_processed = set()

        children_dict = defaultdict(set)
        parent_dict = defaultdict(set)

        # get sink nodes, children_dict, parent_dict using BFS/DFS
        while len(maps_queued) > 0:
            maps_added = 0
            current_map = maps_queued.pop()
            if current_map in maps_processed:
                continue

            nodes_following = set([e.dst for e in graph.out_edges(current_map)])
            while len(nodes_following) > 0:
                current_node = nodes_following.pop()
                if isinstance(current_node,
                              nodes.MapEntry) and current_node in map_entries:
                    if current_node not in maps_processed or maps_queued:
                        maps_queued.add(current_node)
                    maps_added += 1
                    children_dict[current_map].add(current_node)
                    parent_dict[current_node].add(current_map)
                else:
                    for e in graph.out_edges(current_node):
                        nodes_following.add(e.dst)

            if maps_added == 0:
                sink_maps.add(current_map)

            maps_processed.add(current_map)

        # next up, calculate inferred ranges for each map
        # for each map entry, this contains a tuple of dicts:
        # each of those maps from data_name of the array to
        # inferred outer ranges. An inferred outer range is created
        # by taking the union of ranges of inner subsets corresponding
        # to that data and substituting this subset by the min / max of the
        # parametrized map boundaries
        # finally, from these outer ranges we can easily calculate
        # strides and tile sizes required for every map
        inferred_ranges = defaultdict(dict)

        # create array of reverse topologically sorted map entries
        # to iterate over
        topo_reversed = []
        queue = set(sink_maps.copy())
        while len(queue) > 0:
            element = next(e for e in queue
                           if not children_dict[e] - set(topo_reversed))
            topo_reversed.append(element)
            queue.remove(element)
            for parent in parent_dict[element]:
                queue.add(parent)

        # main loop
        # first get coverage dicts for each map entry
        # for each map, contains a tuple of two dicts
        # each of those two maps from data name to outer range
        coverage = {}
        for map_entry in map_entries:
            coverage[map_entry] = StencilTiling.coverage_dicts(sdfg,
                                                               graph,
                                                               map_entry,
                                                               outer_range=True)

        # we have a mapping from data name to outer range
        # however we want a mapping from map parameters to outer ranges
        # for this we need to find out how all array dimensions map to
        # outer ranges

        variable_mapping = defaultdict(list)
        for map_entry in topo_reversed:
            map = map_entry.map

            # first find out variable mapping
            for e in itertools.chain(graph.out_edges(map_entry),
                                     graph.in_edges(
                                         graph.exit_node(map_entry))):
                mapping = []
                for dim in e.data.subset:
                    syms = set()
                    for d in dim:
                        syms |= symbolic.symlist(d).keys()
                    if len(syms) > 1:
                        raise NotImplementedError(
                            "One incoming or outgoing stencil subset is indexed "
                            "by multiple map parameters. "
                            "This is not supported yet."
                        )
                    try:
                        mapping.append(syms.pop())
                    except KeyError:
                        # just append None if there is no map symbol in it.
                        # we don't care for now.
                        mapping.append(None)

                if e.data in variable_mapping:
                    # assert that this is the same everywhere.
                    # else we might run into problems
                    assert variable_mapping[e.data.data] == mapping
                else:
                    variable_mapping[e.data.data] = mapping

            # now do mapping data -> indent
            # and from that infer mapping variable -> indent
            local_ranges = {dn: None for dn in coverage[map_entry][1].keys()}
            for data_name, cov in coverage[map_entry][1].items():
                local_ranges[data_name] = subsets.union(local_ranges[data_name],
                                                        cov)
                # now look at proceeding maps
                # and union those subsets -> could be larger with stencil indent
                for child_map in children_dict[map_entry]:
                    if data_name in coverage[child_map][0]:
                        local_ranges[data_name] = subsets.union(
                            local_ranges[data_name],
                            coverage[child_map][0][data_name])

            # final assignent: combine local_ranges and variable_mapping
            # together into inferred_ranges
            inferred_ranges[map_entry] = {p: None for p in map.params}
            for data_name, ranges in local_ranges.items():
                for param, r in zip(variable_mapping[data_name], ranges):
                    # create new range from this subset and assign
                    rng = subsets.Range((r, ))
                    if param:
                        inferred_ranges[map_entry][param] = subsets.union(
                            inferred_ranges[map_entry][param], rng)

        # outer range of one of the sink maps TODO comment better
        params = next(iter(map_entries)).map.params.copy()
        self.reference_range = inferred_ranges[next(iter(sink_maps))]
        if self.debug:
            print("StencilTiling::Reference Range", self.reference_range)
        # next up, search for the ranges that don't change
        invariant_dims = []
        for idx, p in enumerate(params):
            different = False
            if self.reference_range[p] is None:
                invariant_dims.append(idx)
                warnings.warn(
                    f"StencilTiling::No Tiling Indent for parameter {p}")
                continue
            for m in map_entries:
                if inferred_ranges[m][p] != self.reference_range[p]:
                    different = True
                    break
            if not different:
                invariant_dims.append(idx)
                warnings.warn(
                    f"StencilTiling::No Tiling Indent for parameter {p}")

        # with inferred_ranges constructed, we can begin to strip mine
        from dace.transformation.dataflow.map_collapse import MapCollapse
        from dace.transformation.dataflow.strip_mining import StripMining
        for map_entry in map_entries:
            # Retrieve map entry and exit nodes.
            map = map_entry.map

            stripmine_subgraph = {
                StripMining._map_entry: graph.nodes().index(map_entry)
            }

            sdfg_id = sdfg.sdfg_id
            last_map_entry = None
            original_schedule = map_entry.schedule
            self.tile_sizes = []
            self.tile_offset_lower = []
            self.tile_offset_upper = []

            # strip mining each dimension where necessary
            removed_maps = 0
            all_trivial = True
            for dim_idx, param in enumerate(map_entry.map.params):
                # get current_node tile size
                if dim_idx >= len(self.strides):
                    tile_stride = symbolic.pystr_to_symbolic(self.strides[-1])
                else:
                    tile_stride = symbolic.pystr_to_symbolic(
                        self.strides[dim_idx])

                trivial = False

                if dim_idx in invariant_dims:
                    self.tile_sizes.append(tile_stride)
                    self.tile_offset_lower.append(0)
                    self.tile_offset_upper.append(0)
                else:
                    target_range_current = inferred_ranges[map_entry][param]
                    reference_range_current = self.reference_range[param]
                    # calculate parameters for this dimension
                    success = self.calculate_stripmining_parameters(
                        reference_range=reference_range_current,
                        current_range=target_range_current,
                        stride=tile_stride)

                    assert success, "Please check your parameters and run match()"

                # get calculated parameters
                tile_size = self.tile_sizes[-1]

                dim_idx -= removed_maps
                # If map or tile sizes are trivial, skip strip-mining map dimension
                # special cases:
                # if tile size is trivial AND we have an invariant dimension, skip
                if tile_size == map.range.size()[dim_idx] and (
                        dim_idx + removed_maps) in invariant_dims:
                    continue

                # trivial map: we just continue
                if map.range.size()[dim_idx] in [0, 1]:
                    continue

                if tile_size == 1 and tile_stride == 1 and (
                        dim_idx + removed_maps) in invariant_dims:
                    trivial = True
                    removed_maps += 1
                else:
                    all_trivial = False

                # indent all map ranges accordingly and then perform
                # strip mining on these. Offset inner maps accordingly afterwards

                range_tuple = (map.range[dim_idx][0] +
                               self.tile_offset_lower[-1],
                               map.range[dim_idx][1] -
                               self.tile_offset_upper[-1],
                               map.range[dim_idx][2])
                map.range[dim_idx] = range_tuple
                stripmine = StripMining(sdfg_id, self.state_id,
                                        stripmine_subgraph, 0)

                stripmine.dim_idx = dim_idx
                stripmine.new_dim_prefix = self.prefix if not trivial else ''
                # use tile_stride for both -- we will extend
                # the inner tiles later
                stripmine.tile_size = str(tile_stride)
                stripmine.tile_stride = str(tile_stride)
                outer_map = stripmine.apply(sdfg)

                # apply to the new map the schedule of the original one
                map_entry.schedule = self.schedule

                # if tile stride is 1, we can make a nice simplification by just
                # taking the overapproximated inner range as inner range
                # this eliminates the min/max in the range which
                # enables loop unrolling
                if tile_stride == 1:
                    map_entry.range[dim_idx] = tuple(
                        symbolic.SymExpr(el._approx_expr) if isinstance(
                            el, symbolic.SymExpr) else el
                        for el in map_entry.range[dim_idx])

                # in map_entry: enlarge tiles by upper and lower offset
                # doing it this way and not via stripmine strides ensures
                # that the max gets changed as well
                old_range = map_entry.range[dim_idx]
                map_entry.range[dim_idx] = ((old_range[0] -
                                             self.tile_offset_lower[-1]),
                                            (old_range[1] +
                                             self.tile_offset_upper[-1]),
                                            old_range[2])

                # We have to propagate here - else nasty nasty
                _propagate_node(graph, map_entry)
                _propagate_node(graph, graph.exit_node(map_entry))

                # usual tiling pipeline
                if last_map_entry:
                    new_map_entry = graph.in_edges(map_entry)[0].src
                    mapcollapse_subgraph = {
                        MapCollapse._outer_map_entry:
                        graph.node_id(last_map_entry),
                        MapCollapse._inner_map_entry:
                        graph.node_id(new_map_entry)
                    }
                    mapcollapse = MapCollapse(sdfg_id, self.state_id,
                                              mapcollapse_subgraph, 0)
                    mapcollapse.apply(sdfg)
                last_map_entry = graph.in_edges(map_entry)[0].src

            # Loop Unroll Feature: only unroll if it makes sense
            if self.unroll_loops and all(s == 1 for s in self.strides) and any(
                    s not in [0, 1] for s in map_entry.range.size()):
                l = len(map_entry.params)
                subgraph = {
                    MapExpansion._map_entry: graph.nodes().index(map_entry)
                }
                trafo_expansion = MapExpansion(sdfg.sdfg_id,
                                               sdfg.nodes().index(graph),
                                               subgraph, 0)
                trafo_expansion.apply(sdfg)
                maps = [map_entry]
                for _ in range(l - 1):
                    map_entry = graph.out_edges(map_entry)[0].dst
                    maps.append(map_entry)

                for map in reversed(maps):
                    # MapToForLoop
                    subgraph = {
                        MapToForLoop._map_entry: graph.nodes().index(map)
                    }
                    trafo_for_loop = MapToForLoop(sdfg.sdfg_id,
                                                  sdfg.nodes().index(graph),
                                                  subgraph, 0)
                    trafo_for_loop.apply(sdfg)
                    nsdfg = trafo_for_loop._nsdfg

                    # LoopUnroll
                    guard = trafo_for_loop.guard
                    begin = trafo_for_loop.before_state
                    end = trafo_for_loop.after_state

                    subgraph = {
                        DetectLoop._loop_guard: nsdfg.nodes().index(guard),
                        DetectLoop._loop_begin: nsdfg.nodes().index(begin),
                        DetectLoop._exit_state: nsdfg.nodes().index(end)
                    }
                    transformation = LoopUnroll(0, 0, subgraph, 0)
                    transformation.apply(nsdfg)
            elif self.unroll_loops:
                warnings.warn(
                    "Could not unroll loops. Loop size has to be known"
                    "(and thus strides have to be of size 1)")
