""" This module contains classes that implement a composite
    Subgraph Fusion - Stencil Tiling Transformation
"""


import dace

import dace.transformation.transformation as transformation
from dace.transformation.subgraph import SubgraphFusion, StencilTiling
import dace.transformation.subgraph.helpers as helpers

from dace import dtypes, registry, symbolic, subsets, data
from dace.properties import make_properties, Property, ShapeProperty
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

import warnings

@registry.autoregister_params(singlestate=True)
@make_properties
class CompositeFusion(transformation.SubgraphTransformation):
    """ StencilTiling + SubgraphFusion in one Transformation
        This is needed for Estimation purposes

        Checks and applies SubgraphFusion if possible. If not,
        it tries to resorts to StencilTiling followed by SubgraphFusion.
    """

    debug = Property(desc="Debug mode", dtype=bool, default = True)

    allow_tiling = Property(desc="Allow StencilTiling before",
                            dtype = bool,
                            default = True)

    gpu_fusion_mode = Property(desc="Fusion local register memory architecture "
                                    "or shared memory architecture. If set to "
                                    "None, all is left at default.",
                               default = None,
                               dtype = str,
                               choices = ['register', 'shared'],
                               allow_none = True)

    transient_allocation = Property(
        desc="Storage Location to push transients to that are "
              "fully contained within the subgraph.",
        dtype=dtypes.StorageType,
        default=dtypes.StorageType.Default)

    schedule_innermaps = Property(desc="Schedule of inner maps",
                                  dtype=dtypes.ScheduleType,
                                  default=dtypes.ScheduleType.Default)

    stencil_unroll_loops = Property(desc="Unroll Inner Loops if they have Size > 1",
                                    dtype=bool,
                                    default=False)
    stencil_strides = ShapeProperty(dtype=tuple,
                                    default=(1, ),
                                    desc="Tile stride")


    @staticmethod
    def can_be_applied(sdfg: SDFG, subgraph: SubgraphView) -> bool:
        if SubgraphFusion.can_be_applied(sdfg, subgraph):
            return True
        if CompositeFusion.allow_tiling:
            if StencilTiling.can_be_applied(sdfg, subgraph):
                return True

        return False

    def apply(self, sdfg):
        # first do some grunt work
        # get graph and subgraph
        subgraph = self.subgraph_view(sdfg)
        graph = subgraph.graph
        scope_dict = graph.scope_dict()
        first_entry = next(iter(helpers.get_outermost_scope_maps(sdfg, graph, subgraph, scope_dict)))

        # check gpu related issues:
        is_gpu = False
        if first_entry.schedule in [dtypes.ScheduleType.GPU_Device]:
            # just take GPU_Device fornow
            is_gpu = True

        # assign reduce node implementation
        if is_gpu:
            for node in subgraph.nodes():
                if isinstance(node, dace.libraries.standard.nodes.Reduce) and scope_dict[node] is not None:
                    if self.transient_allocation ==dtypes.StorageType.Default:
                        warnings.warn("Warning: GPU Reduce node is going to "
                                      "be expanded with default - behaviour.")
                    elif self.transient_allocation == dtypes.StorageType.Register:
                        node.implementation = 'pure'
                    elif self.transient_allocation == dtypes.StorageType.GPU_Shared:
                        node.implementation = 'CUDA (block allreduce)'
                    else:
                        raise RuntimeError("For GPU useage, transient allocation has to be "
                                           "either register or shared memory.")

        # do a few safety checks
        if is_gpu:
            if self.transient_allocation not in [dtypes.StorageType.Register, dtypes.StorageType.GPU_Shared]:
                warnings.warn("Ambiguous transient allocation "
                              "in a GPU State for Fusion")
            if self.schedule_innermaps not in [dtypes.ScheduleType.Sequential, dtypes.ScheduleType.GPU_ThreadBlock]:
                warnings.warn("Ambiguous innermap scheduling "
                              "in a GPU State for Fusion")


        if SubgraphFusion.can_be_applied(sdfg, self.subgraph_view(sdfg)):
            sf = SubgraphFusion(subgraph, self.sdfg_id, self.state_id)
            # set SubgraphFusion properties
            sf.debug = self.debug
            sf.transient_allocation = self.transient_allocation
            sf.schedule_innermaps = self.schedule_innermaps
            sf.apply(sdfg)

        elif self.allow_tiling and StencilTiling.can_be_applied(sdfg, self.subgraph_view(sdfg)):
            print("******** TILING")
            st = StencilTiling(self.subgraph_view(sdfg), self.sdfg_id, self.state_id)
            # set StencilTiling properties
            st.debug = self.debug
            st.unroll_loops = self.stencil_unroll_loops
            st.strides= self.stencil_strides
            st.apply(sdfg)

            # StencilTiling: add to nodes
            for map_entry in map_entries_copy:
                outer_entry = graph_copy.in_edges(map_entry)[0].src
                outer_exit = graph.exit_node(outer_entry)
                subgraph._subgraph_nodes += [outer_entry, outer_exit]

            sf = SubgraphFusion(self.subgraph_view(sdfg), self.sdfg_id, self.state_id)
            # set SubgraphFusion properties
            sf.debug = self.debug
            sf.transient_allocation = self.transient_allocation
            sf.schedule_innermaps = self.schedule_innermaps

            sf.apply(sdfg)

        else:
            raise NotImplementedError("Error")
