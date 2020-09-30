""" This module contains classes and functions that implement the orthogonal
    stencil tiling transformation. """
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

from dace.transformation.subgraph import helpers

@registry.autoregister_params(singlestate=True)
@make_properties
class StencilTiling(pattern_matching.SubgraphTransformation):
    """ Implements the orthogonal tiling transformation.

        Orthogonal tiling is a type of nested map fission that creates tiles
        in every dimension of the matched Map.
    """

    # Properties
    debug = Property(desc = "Debug mode",
                     dtype = bool,
                     default = True)

    reference_range = Property(desc = "Reference Range",
                          dtype = Range,
                          default = None,
                          allow_none = True)

    prefix = Property(dtype=str,
                      default="stencil",
                      desc="Prefix for new range symbols")

    strides = ShapeProperty(dtype=tuple,
                            default=(1,),
                            desc="Tile stride")

    schedule = Property(dtype=dace.dtypes.ScheduleType,
                        default = dace.dtypes.ScheduleType.Default,
                        desc = "Inner Schedule Type")

    unroll_loops = Property(desc = "Loop unroll",
                            dtype = bool,
                            default = False)

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    @staticmethod
    def match(sdfg, subgraph):
        graph = subgraph.graph
        map_entries = set(helpers.get_highest_scope_maps(sdfg, graph, subgraph))
        if len(map_entries) < 1:
            return False
        source_nodes = set(sdutil.find_source_nodes(graph))
        source_maps = set()
        sink_maps = set()
        while len(source_nodes) > 0:
            # traverse and find source maps
            node = next(iter(source_nodes))
            if isinstance(node, nodes.MapEntry) and node in map_entries:
                source_maps.add(node)
            else:
                for e in graph.out_edges(node):
                    source_nodes.add(e.dst)
            source_nodes.remove(node)

        maps_queued = set(source_maps)
        maps_processed = set()
        while len(maps_queued) > 0:
            maps_added = 0
            current_map = next(iter(maps_queued))
            if current_map in maps_processed:
                maps_queued.remove(current_map)
                continue
            nodes_current = set([e.dst for e in graph.out_edges(current_map)])
            while len(nodes_current) > 0:
                current = next(iter(nodes_current))
                if isinstance(current, nodes.MapEntry) and current in map_entries:
                    if current not in maps_processed or maps_queued:
                        maps_queued.add(current)
                    maps_added += 1
                    # now check whether subsets cover
                    if not current_map.range.covers(current.range):
                        return False
                else:
                    for e in graph.out_edges(current):
                        nodes_current.add(e.dst)
                nodes_current.remove(current)

            if maps_added == 0:
                sink_maps.add(current_map)

            maps_queued.remove(current_map)
            maps_processed.add(current_map)

        # last condition: we want all sink maps to have the same
        # range, if not we cannot form a common map range for fusion later
        assert len(sink_maps) > 0
        map0 = next(iter(sink_maps))
        if not all([map.range == map0.range for map in sink_maps]):
            return False

        return True

    def calculate_stripmining_parameters(self,
                                         reference_range,
                                         target_range,
                                         stride):
        """ Calculates parameters for stripmining and offsetting
            Iff successful (transformation posible), returns True
            Saves all parameters as class variable
        """
        map_stride = target_range.strides()[0]
        if reference_range.strides()[0] != target_range.strides()[0]:
            # different strides
            return False
        min_diff = symbolic.SymExpr(target_range.min_element()[0] \
                        - reference_range.min_element()[0])
        #                / reference_range.strides()[0]
        max_diff = symbolic.SymExpr(reference_range.max_element()[0] \
                        - target_range.max_element()[0])
        #                / reference_range.strides()[0]


        try:
            min_diff = symbolic.evaluate(min_diff, {})
            max_diff = symbolic.evaluate(max_diff, {})
        except TypeError:
            print("TypeError in Symbolic Evaluation")
            print("min_diff =", min_diff)
            print("max_diff =", max_diff)
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

        # first, get the reference range (smallest range covered by all nodes)
        # has to exist uniquely, else match should return False
        # O(n^2) slow, TODO: improve?
        for map_entry in map_entries:
            if all([m.range.covers(map_entry.range) for m in map_entries]):
                self.reference_range = dcpy(map_entry.range)
                break
        if self.debug:
            print("Operating with reference range", self.reference_range)

        # next up, search for the ranges that don't change
        invariant_ranges = []
        for idx, rng in enumerate(self.reference_range):
            different = False
            for m in map_entries:
                if m.range[idx] != rng:
                    print(m.range[idx], '!=', rng)
                    different = True
                    break
            if not different:
                invariant_ranges.append(idx)
        print("INVARIANT RANGES", invariant_ranges)

        assert self.reference_range is not None, "Wrong input, please use match()"


        from dace.transformation.dataflow.map_collapse import MapCollapse
        from dace.transformation.dataflow.strip_mining import StripMining
        print("MAP_ENTRIES", map_entries)
        for map_entry in map_entries:
            # Retrieve map entry and exit nodes.
            map = map_entry.map

            stripmine_subgraph = {
                StripMining._map_entry: graph.nodes().index(map_entry)
            }
            sdfg_id = sdfg.sdfg_id
            last_map_entry = None

            original_schedule = map_entry.schedule

            # TBD arrays
            self.tile_sizes = []
            self.tile_offset_lower = []
            self.tile_offset_upper = []

            # strip mining each dimension where necessary
            removed_maps = 0
            all_trivial = True

            for dim_idx in range(len(map_entry.map.params)):
                # get current tile size
                if dim_idx >= len(self.strides):
                    tile_stride = symbolic.pystr_to_symbolic(self.strides[-1])
                else:
                    tile_stride = symbolic.pystr_to_symbolic(self.strides[dim_idx])

                reference_range_current = subsets.Range((map.range[dim_idx],))
                target_range_current = subsets.Range((self.reference_range[dim_idx-removed_maps],))

                inner_trivial = False

                # calculate parameters for this dimension
                print("*********************************************")
                print(map_entry)
                print(self.reference_range)
                success = self.calculate_stripmining_parameters(
                    reference_range = reference_range_current,
                    target_range = target_range_current,
                    stride = tile_stride)

                assert success, "Please check your parameters and run match()"

                # get calculated parameters
                tile_size = self.tile_sizes[-1]
                offset = self.tile_offset_lower[-1]

                # If tile size is trivial, skip strip-mining map dimension
                if map_entry.map.range.size()[dim_idx-removed_maps] in [0,1]:
                    continue

                #if tile_size == map_entry.map.range.size()[dim_idx]:
                #    continue


                # change map range to target reference.
                # then perform strip mining on this and
                # offset inner maps accordingly.
                dim_idx -= removed_maps
                if tile_size == 1 and tile_stride == 1:
                    inner_trivial = True
                else:
                    all_trivial = False

                if inner_trivial and dim_idx+removed_maps in invariant_ranges and 0==1: # TODO
                    print("XXXX inner_trivial invariant activated")
                    map.range[dim_idx] = dcpy(self.reference_range[dim_idx+removed_maps])

                    stripmine = StripMining(sdfg_id, self.state_id, stripmine_subgraph,
                                            0)
                    stripmine.dim_idx = dim_idx
                    stripmine.new_dim_prefix = self.prefix
                    stripmine.tile_size = str(tile_stride)
                    stripmine.tile_stride = str(tile_stride)
                    outer_map = stripmine.apply(sdfg)
                    removed_maps += 1

                else:

                    map.range[dim_idx] = dcpy(self.reference_range[dim_idx+removed_maps])
                    stripmine = StripMining(sdfg_id, self.state_id, stripmine_subgraph,
                                            0)
                    stripmine.skip_trivial_dims = False

                    stripmine.dim_idx = dim_idx
                    stripmine.new_dim_prefix = self.prefix
                    # use tile_stride for both -- we will extend
                    # the inner tiles later
                    stripmine.tile_size = str(tile_stride)
                    stripmine.tile_stride = str(tile_stride)
                    outer_map = stripmine.apply(sdfg)

                # apply to the new map the schedule of the original one
                map_entry.schedule = self.schedule

                # if inner_trivial:
                # just take overapproximation - strip the rest from outer
                if tile_stride == 1:
                    map_entry.range[dim_idx] =  tuple(symbolic.SymExpr(el._approx_expr)  \
                                                if isinstance(el, symbolic.SymExpr) else el \
                                                for el in map_entry.range[dim_idx])

                # in map_entry: enlarge tiles by upper and lower offset
                # doing it this way and not via stripmine strides ensures
                # that the max gets changed as well
                old_range = map_entry.range[dim_idx]
                map_entry.range[dim_idx] = ((old_range[0] - self.tile_offset_lower[-1]), \
                                            (old_range[1] + self.tile_offset_upper[-1]), \
                                            old_range[2])

                # We have to propagate here - else nasty nasty
                _propagate_node(graph, map_entry)
                _propagate_node(graph, graph.exit_node(map_entry))

                #sdfg.view()
                # usual tiling pipeline
                if last_map_entry:
                    new_map_entry = graph.in_edges(map_entry)[0].src
                    mapcollapse_subgraph = {
                        MapCollapse._outer_map_entry:
                        graph.node_id(last_map_entry),
                        MapCollapse._inner_map_entry: graph.node_id(new_map_entry)
                    }
                    mapcollapse = MapCollapse(sdfg_id, self.state_id,
                                              mapcollapse_subgraph, 0)
                    mapcollapse.apply(sdfg)
                last_map_entry = graph.in_edges(map_entry)[0].src

            print(map_entry)
            if self.unroll_loops and all(s == 1 for s in self.strides) and not all_trivial:
                l = len(map_entry.params)
                subgraph = {MapExpansion._map_entry : graph.nodes().index(map_entry)}
                trafo_expansion = MapExpansion(sdfg.sdfg_id,
                    sdfg.nodes().index(graph), subgraph, 0)
                trafo_expansion.apply(sdfg)
                maps = [map_entry]
                for _ in range(l-1):
                    map_entry = graph.out_edges(map_entry)[0].dst
                    maps.append(map_entry)

                print(maps)
                for map in reversed(maps):
                    # MapToForLoop
                    subgraph = {MapToForLoop._map_entry: graph.nodes().index(map)}
                    trafo_for_loop = MapToForLoop(sdfg.sdfg_id,
                        sdfg.nodes().index(graph), subgraph, 0)
                    trafo_for_loop.apply(sdfg)
                    nsdfg = trafo_for_loop._nsdfg

                    # LoopUnroll
                    # TODO: improve
                    # lol
                    guard, begin, end = None, None, None
                    for ngraph in nsdfg.nodes():
                        if ngraph.label == 'state_0':
                            begin = ngraph
                        if ngraph.label == 'state_2':
                            end = ngraph
                        if ngraph.label == 'guard':
                            guard = ngraph
                    assert guard is not None
                    assert begin is not None
                    assert end is not None
                    subgraph = {DetectLoop._loop_guard: nsdfg.nodes().index(guard),
                                DetectLoop._loop_begin: nsdfg.nodes().index(begin),
                                DetectLoop._exit_state: nsdfg.nodes().index(end)}
                    transformation = LoopUnroll(0, 0, subgraph, 0)
                    transformation.apply(nsdfg)

                #sdfg.view()
