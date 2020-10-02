""" This module contains classes and functions that implement the orthogonal
    stencil tiling transformation. """

"""
TODO:
    - Stencil detection improvement
        - write function to detect from memlets
        - write map offset function
    - Fix ignore cases
    - Make ready for PR
    - Write a unittest






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

from collections import defaultdict

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
    def coverage_dicts(sdfg, graph, map_entry, outer_range = True):
        '''
        returns a two dicts:
        one dict that has as a key all data entering the map
        and its associated access range
        one dict that has as a key all data exiting the map
        and its associated access range
        '''
        map_exit = graph.exit_node(map_entry)
        map = map_entry.map

        entry_coverage = {}
        exit_coverage = {}
        # create dicts with which we can replace all iteration
        # variables by their range
        map_min = {dace.symbol(param): e
                   for param, e in zip(map.params, map.range.min_element())}
        map_max = {dace.symbol(param): e
                   for param, e in zip(map.params, map.range.max_element())}

        # look at inner memlets at map entry
        for e in graph.out_edges(map_entry):
            if outer_range:
                # get subset
                min_element = [m.subs(map_min) for m in e.data.subset.min_element()]
                max_element = [m.subs(map_max) for m in e.data.subset.max_element()]
                # create range
                rng = subsets.Range((min_e, max_e+1, 1)
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
                min_element = [m.subs(map_min) for m in e.data.subset.min_element()]
                max_element = [m.subs(map_max) for m in e.data.subset.max_element()]
                # craete range
                rng = subsets.Range((min_e, max_e+1, 1)
                       for min_e, max_e in zip(min_element, max_element))
            else:
                rng = dcpy(e.data.subset)

            if e.data.data not in exit_coverage:
                exit_coverage[e.data.data] = rng
            else:
                old_coverage = exit_coverage[e.data]
                exit_coverage[e.data.data] = subsets.union(old_coverage, rng)
        return (entry_coverage, exit_coverage)


    @staticmethod
    def match(sdfg, subgraph):
        # TODO: all parameters have to be the same
        # get highest scope maps
        graph = subgraph.graph
        map_entries = set(helpers.get_highest_scope_maps(sdfg, graph, subgraph))
        if len(map_entries) < 1:
            return False

        params = dcpy(next(iter(map_entries)).map.params)
        for map_entry in map_entries:
            if map_entry.map.params != params:
                return False

        # get source maps as a starting point for BFS
        source_nodes = set(sdutil.find_source_nodes(graph))
        maps_reachable_source = set()
        sink_maps = set()
        while len(source_nodes) > 0:
            # traverse and find source maps
            node = next(iter(source_nodes))
            if isinstance(node, nodes.MapEntry) and node in map_entries:
                maps_reachable_source.add(node)
            else:
                for e in graph.out_edges(node):
                    source_nodes.add(e.dst)
            source_nodes.remove(node)

        # main loop: traverse graph and check whether topologically
        # connected maps cover each other in terms of memlets
        maps_queued = set(maps_reachable_source)
        maps_processed = set()
        coverages = {}
        for map_entry in map_entries:
            coverages[map_entry] = StencilTiling.coverage_dicts(sdfg, graph, map_entry)
            for data, cov in coverages[map_entry][0].items():
                print(cov[0])
                if data in coverages[map_entry][1] and not cov.covers(coverages[map_entry][1][data]):
                    print("WRONG COVERAGE INTERNAL")
                    print(data)
                    print(cov)
                    print(coverages[map_entry[1][data]])
                    return False
        while len(maps_queued) > 0:
            maps_added = 0
            current_map = next(iter(maps_queued))
            if current_map in maps_processed:
                # this should not occur in a DAG
                maps_queued.remove(current_map)
                continue

            nodes_following = set([e.dst for e in graph.out_edges(current_map)])
            while len(nodes_following) > 0:
                current = next(iter(nodes_following))
                if isinstance(current, nodes.MapEntry) and current in map_entries:
                    if current not in maps_processed or maps_queued:
                        maps_queued.add(current)
                    maps_added += 1
                    # now check whether subsets cover
                    # get dict of incoming edges of this inner loop map
                    in_dict = coverages[current][0]
                    # get dict of outgoing edges of map in the
                    # preceding outer loop map
                    out_dict = coverages[current_map][1]
                    for data_name in in_dict:
                        # check if data is in out_dict
                        if data_name in out_dict:
                            # check coverage
                            if not out_dict[data_name].covers(in_dict[data_name]):
                                print("WRONG COVERAGE CHILD")
                                print("Parent:", current_map)
                                print("Child:", current)
                                print(out_dict[data_name])
                                print(in_dict[data_name])
                                #sdfg.view()
                                return False
                            else:
                                print("PASS")

                else:
                    for e in graph.out_edges(current):
                        nodes_following.add(e.dst)
                nodes_following.remove(current)

            if maps_added == 0:
                sink_maps.add(current_map)

            maps_queued.remove(current_map)
            maps_processed.add(current_map)

        # last condition: we want all sink maps to have the same
        # range, if not we cannot form a common map range for fusion later
        # last condition: we want all incoming memlets into sink maps
        # to have the same ranges

        assert len(sink_maps) > 0
        first_sink_map = next(iter(sink_maps))
        # TODO: need to check whether this is strong enough
        if not all([map.range.size() == first_sink_map.range.size() for map in sink_maps]):
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

        # first get dicts of parents and children for each map_entry
        # get source maps as a starting point for BFS
        source_nodes = set(sdutil.find_source_nodes(graph))
        maps_reachable_source = set()
        sink_maps = set()
        while len(source_nodes) > 0:
            # traverse and find source maps
            node = next(iter(source_nodes))
            if isinstance(node, nodes.MapEntry) and node in map_entries:
                maps_reachable_source.add(node)
            else:
                for e in graph.out_edges(node):
                    source_nodes.add(e.dst)
            source_nodes.remove(node)

        # traverse graph
        maps_queued = set(maps_reachable_source)
        maps_processed = set()

        children_dict = defaultdict(set)
        parent_dict = defaultdict(set)

        # get sink nodes, children_dict, parent_dict using BFS/DFS
        while len(maps_queued) > 0:
            maps_added = 0
            current_map = next(iter(maps_queued))
            if current_map in maps_processed:
                # this should not occur in a DAG
                maps_queued.remove(current_map)
                raise RuntimeError("DAG")

            nodes_following = set([e.dst for e in graph.out_edges(current_map)])
            while len(nodes_following) > 0:
                current = next(iter(nodes_following))
                if isinstance(current, nodes.MapEntry) and current in map_entries:
                    if current not in maps_processed or maps_queued:
                        maps_queued.add(current)
                    maps_added += 1
                    children_dict[current_map].add(current)
                    parent_dict[current].add(current_map)
                else:
                    for e in graph.out_edges(current):
                        nodes_following.add(e.dst)
                nodes_following.remove(current)

            if maps_added == 0:
                sink_maps.add(current_map)

            maps_queued.remove(current_map)
            maps_processed.add(current_map)

        ######################## OPTION 1
        # next up, for each map, calculate the indent of each parameter that
        # we have to perform later during the tiling step

        # FORMAT:
        # parameter_indent[map_entry] =
        # {data_name: ((lower_indent, upper_indent),(lower......)) }
        parameter_indent = defaultdict(dict)

        # create array of reverse topologically sorted map entries
        # to iterate over
        topo = []
        topo_set = set()
        queue = set(sink_maps.copy())
        while len(queue) > 0:
            current = None
            for element in queue:
                # if all parents are already in topo
                if not children_dict[element] - topo_set:
                    topo.append(element)
                    topo_set.add(element)
                queue.remove(element)
                for parent in parent_dict[element]:
                    queue |= parent

        # main loop
        # first get inner coverage
        coverage = {}
        for map_entry in map_entries:
            coverage[map_entry] = StencilTiling.coverage_dicts(sdfg, graph, map_entry, outer_range = True)

        for map_entry in topo:
            map = map_entry.map
            if map_entry in sink_maps:
                # easy, indent is 0 everywhere
                # this is our base range
                parameter_indent[map_entry] = {p: (0,1) for p in map.params}
            else:
                # not a sink map, we potentially have to add an indent
                # look at all outgoing and
                # first find out variable mapping

                local_variables = {}
                for e in chain(graph.out_edges(map_entry), graph.in_edges(graph.exit_node(map_entry))):
                    mapping = []
                    for dim in e.data.subset.ranges:
                        syms = set()
                        for d in dim:
                            syms |= symbolic.symlist(d).keys()
                        assert len(syms) <= 2, "Not supported" # TODO: error message
                        local_variables[e.data].append(next(syms) if len(syms) == 1 else None)

                    if e.data in local_variables:
                        assert local_variables[e.data] == mapping
                    else:
                        local_variables[e.data] = mapping

                # now do mapping data -> indent
                # and from that infer mapping variable -> indent
                local_baserange = {}
                for child_map in children_dict[map_entry]:

                    for data_name, coverage in coverages[child_map].values():

                        # below is for outer == False:
                        # get coverage diff
                        #diff_max = [pre_cov[0] + c2 - c1 for pre_cov, c1, c2 in zip(parameter_indent[map_entry][data_name], coverage[0].min_element(), coverage[1].min_element())]
                        #diff_min = [pre_cov[1] + c1 - c2 for pre_cov, c1, c2 in zip(parameter_indent[map_entry][data_name], coverage[0].max_element(), coverage[1].max_element())]
                        #indent = tuple((dmin, dmax) for dmin, dmax in zip(diff_min, diff_max))

                        if data_name in coverages[map_entry][1]:

                            # update parameters
                            if data_name in local_baserange:
                                # extend
                                local_baserange[data_name] = subsets.union(local_baserange[data_name], coverage)
                                #
                                #local_baserange[data_name] = tuple(min(pimin, dmin), max(pimax, dmax) for ((pimin, pimax), (dmin, dmax))
                                #                                            in zip(indent_local[map_entry[data_name]], zip(diff_max, diff_min)))
                            else:
                                local_baserange[data_name] = coverage
                                #local_baserange[data_name] = indent
                        else:
                            # data element not present in child map
                            pass


                # final assignent
                for (data_name, indents) in local_baserange.values():
                    for (param, indent) in zip(local_variables[data_name], indents):
                        '''
                        if param in parameter_indent[map_entry]:
                            assert parameter_indent[map_entry][param] == indent
                        else:
                            parameter_indent[map_entry][param] = indent
                        '''
                        if param in parameter_indent[map_entry]:
                            parameter_indent[map_entry] = subsets.union(parameter_indent[map_entry], subsets.Range(indent))
                        else:
                            parameter_indent[map_entry] = subsets.Range(indent)


        '''
        ###################################### OPTION 2
        # get coverage dicts
        coverages = {}
        for map_entry in map_entries:
            coverages[map_entry] = StencilTiling.coverage_dicts(sdfg, graph, current_map)

        # get reference range of sink node, all of them have to be the same
        self.reference_range = coverages[next(sink_maps)]
        assert all(self.reference_range == coverages[sm]for sm in sink_maps)
        print("Operating with reference range", self.reference_range)

        # tricky/dodgy part: create a mapping from array coverages
        # to map params and see whether everyting is safe and sound
        # first see which variables the data ranges map to
        variables = {}
        # create a new dict range_coverages to map
        # from map parameter to reference range instead of
        # map data_name to reference range
        range_coverages = {}
        # example for 3-dim data array:
        # variable[map_entry][data_name] = [i,j,k]
        for map_entry in map_entries:
            local_variables = {}
            for e in chain(graph.out_edges(map_entry), graph.in_edges(graph.exit_node(map_entry))):
                mapping = []
                for dim in e.data.subset.ranges:
                    syms = set()
                    for d in dim:
                        syms |= symbolic.symlist(d).keys()
                    assert len(syms) <= 2, "Not supported" # TODO: error message
                    local_variables[e.data].append(next(syms) if len(syms) == 1 else None)

                if e.data in local_variables:
                    assert local_variables[e.data] == mapping
                else:
                    local_variables[e.data] = mapping
            variables[map_entry] = local_variables

            # TODO: ????? PARAMETER CHANGE ?????

            rcov = {}
            coverages[map_entry][0] # entry
            coverages[map_entry][1] # exit

        '''
        print("***************************************")
        print("END OF PART 1")
        print(parameter_indent)


        params = next(map_entries).map.params.copy()
        # next up, search for the ranges that don't change
        invariant_ranges = []
        for idx, p in enumerate(params):
            different = False
            for m in map_entries:
                if parameter_indent[m][p] != (0,1):
                    print(parameter_indent[m][p])
                    different = True
                    break
            if not different:
                invariant_ranges.append(idx)
        print("INVARIANT RANGES", invariant_ranges)

        # finally, we strip mine

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
                ######tile_size = self.tile_sizes[-1]
                tile_size = sum(parameter_indent[map_entry][dim_idx])

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
                map_entry.range[dim_idx] = ((old_range[0] - self.parameter_indent[map_entry][params[dim_idx + removed_maps]][0]), \
                                            (old_range[1] + self.parameter_indent[map_entry][params[dim_idx + removed_maps]][1]), \
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
