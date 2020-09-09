""" This module contains classes and functions that implement the orthogonal
    stencil tiling transformation. """
import math

from dace import registry, symbolic
from dace.properties import make_properties, Property, ShapeProperty
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.transformation import pattern_matching
from dace.symbolic import pystr_to_symbolic
from dace.subsets import Range

from copy import deepcopy as dcpy

import dace.subsets as subsets
import dace.symbolic as symbolic

@registry.autoregister_params(singlestate=True)
@make_properties
class StencilTiling(pattern_matching.Transformation):
    """ Implements the orthogonal tiling transformation.

        Orthogonal tiling is a type of nested map fission that creates tiles
        in every dimension of the matched Map.
    """

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))

    # Properties
    reference_range = Property(desc = "Reference Range",
                          dtype = Range,
                          default = None,
                          allow_none = True)

    stencil_size = Property(desc = "Stencil size. For trivial dimension put"
                                   "None or (0,1)",
                            dtype = tuple,
                            default = ((0,1),))

    prefix = Property(dtype=str,
                      default="s",
                      desc="Prefix for new range symbols")

    strides = ShapeProperty(dtype=tuple,
                            default=(1,),
                            desc="Tile stride")
    '''
    #not needed any more

    tile_sizes = ShapeProperty(
        dtype=tuple,
        default=tuple(),
        desc="Tile size")

    tile_offset = ShapeProperty(
        dtype=tuple,
        default=(0,0,0),
        desc="Negative Tile offset")
    '''

    divides_evenly = Property(dtype=bool,
                              default=False,
                              desc="Tile size divides dimension length evenly")

    @staticmethod
    def annotates_memlets():
        return True

    @staticmethod
    def expressions():
        return [sdutil.node_path_graph(StencilTiling._map_entry)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        map_entry = graph.nodes()[candidate[StencilTiling._map_entry]]
        return map_entry.map.label + ': ' + str(map_entry.map.params)


    def calculate_stripmining_parameters(self,
                                         reference_range,
                                         target_range,
                                         stencil_size,
                                         stride):
        """ Calculates parameters for stripmining and offsetting
            Iff successful (transformation posible), returns True
            Saves all parameters as class variable
        """
        print(str(reference_range.strides()[0]))
        print(target_range.strides())
        if reference_range.strides()[0] != target_range.strides()[0]:
            # different strides
            return False
        min_diff = symbolic.SymExpr(target_range.min_element()[0] \
                        - reference_range.min_element()[0]) \
                        / reference_range.strides()[0]
        max_diff = symbolic.SymExpr(reference_range.max_element()[0] \
                        - target_range.max_element()[0]) \
                        / reference_range.strides()[0]

        stencil_lo = stencil_size[0]
        stencil_hi = stencil_size[1]

        assert stencil_lo <=0 and stencil_hi >=1

        min_diff = symbolic.evaluate(min_diff, {})
        max_diff = symbolic.evaluate(max_diff, {})
        if min_diff < 0 or max_diff < 0:
            return False
        if stencil_lo == 0 and min_diff > 0:
            return False
        if stencil_hi == 1 and max_diff > 0:
            return False

        window_lower = abs(min_diff / stencil_lo) if stencil_lo < 0 else 0
        window_higher = abs(max_diff / (stencil_hi-1)) if stencil_hi > 1 else 0

        if window_lower != window_higher \
                            or math.floor(window_lower) != window_lower \
                            or math.floor(window_higher) != window_higher:
            return False

        window = int(window_lower)
        print("WINDOW", window)
        print(f"(stencil_hi, stencil lo) = ({stencil_hi}, {stencil_lo})")

        self.tile_sizes.append(stride + window * (stencil_hi - stencil_lo - 1))
        self.tile_offset.append(pystr_to_symbolic(str(abs(window * stencil_lo))))

        return True

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]

        # Retrieve map entry and exit nodes.
        map_entry = graph.nodes()[self.subgraph[StencilTiling._map_entry]]
        map = map_entry.map
        from dace.transformation.dataflow.map_collapse import MapCollapse
        from dace.transformation.dataflow.strip_mining import StripMining

        stripmine_subgraph = {
            StripMining._map_entry: self.subgraph[StencilTiling._map_entry]
        }
        sdfg_id = sdfg.sdfg_id
        last_map_entry = None
        removed_maps = 0

        original_schedule = map_entry.schedule

        # TBD arrays
        self.tile_sizes = []
        self.tile_offset = []

        # strip mining each dimension where necessary

        for dim_idx in range(len(map_entry.map.params)):
            # get current tile size
            if dim_idx >= len(self.strides):
                tile_stride = symbolic.pystr_to_symbolic(self.strides[-1])
            else:
                tile_stride = symbolic.pystr_to_symbolic(self.strides[dim_idx])

            # get current stencil size
            stencil_size_current = self.stencil_size[dim_idx] \
                                   if dim_idx < len(self.stencil_size) \
                                   else self.stencil_size[-1]
            stencil_size_current = (0,1) if stencil_size_current is None \
                                   else stencil_size_current

            reference_range_current = subsets.Range((map.range[dim_idx],))
            target_range_current = subsets.Range((self.reference_range[dim_idx],))

            inner_trivial = False
            '''
            if not stencil_size_current \
                or stencil_size_current == (0,1) \
                or target_range_current == reference_range_current:
                continue
            '''

            # calculate parameters for this dimension
            success = self.calculate_stripmining_parameters(
                reference_range = reference_range_current,
                target_range = target_range_current,
                stencil_size = stencil_size_current,
                stride = tile_stride)

            assert success, "Please check your parameters and run match()"

            # get calculated parameters
            tile_size = self.tile_sizes[-1]
            offset = self.tile_offset[-1]

            dim_idx -= removed_maps
            # If tile size is trivial, skip strip-mining map dimension
            if tile_size == map_entry.map.range.size()[dim_idx]:
                continue

            stripmine = StripMining(sdfg_id, self.state_id, stripmine_subgraph,
                                    self.expr_index)
            # Special case: Tile size of 1 should be omitted from inner map
            if tile_size == 1 and tile_stride == 1:
                print("SC")
                print(str(offset))
                stripmine.dim_idx = dim_idx
                stripmine.new_dim_prefix = ''
                stripmine.tile_size = str(tile_size)
                stripmine.tile_stride = str(tile_stride)
                stripmine.tile_offset = str(offset)
                outer_map = stripmine.apply(sdfg)
                removed_maps += 1
                inner_trivial = True
            else:
                stripmine.dim_idx = dim_idx
                stripmine.new_dim_prefix = self.prefix
                stripmine.tile_size = str(tile_size)
                stripmine.tile_stride = str(tile_stride)
                stripmine.tile_offset = str(offset)
                outer_map = stripmine.apply(sdfg)

            ## apply to the new map the schedule of the original one
            #map_entry.schedule = original_schedule
            #return
            # correct the ranges of outer_map
            print("****")
            print(outer_map.range)
            outer_map.range = dcpy(self.reference_range)
            print(outer_map.range)

            # just take overapproximation
            if not inner_trivial:
                print(map_entry.range[dim_idx])
                map_entry.range[dim_idx] =  tuple(symbolic.SymExpr(el._approx_expr)  \
                                            if isinstance(el, symbolic.SymExpr) else el \
                                            for el in map_entry.range[dim_idx])
                print(map_entry.range[dim_idx])
                print(type(map_entry.range[dim_idx]))

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
