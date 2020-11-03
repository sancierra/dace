""" This module contains classes that implement a composite
    Subgraph Fusion - Stencil Tiling Transformation
"""


import dace

import dace.transformation.transformation as transformation
from dace.transformation.subgraph import SubgraphFusion, StencilTiling

from dace import dtypes, registry, symbolic, subsets, data
from dace.properties import make_properties, Property, ShapeProperty
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView

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

    transient_allocation = Property(
        desc="Storage Location to push transients to that are "
              "fully contained within the subgraph.",
        dtype=dtypes.StorageType,
        default=dtypes.StorageType.Default)

    schedule_innermaps = Property(desc="Schedule of inner maps",
                                  dtype=dtypes.ScheduleType,
                                  default=dtypes.ScheduleType.Default,
                                  allow_none=True)

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
        sf = SubgraphFusion(self.subgraph_view(sdfg), self.sdfg_id, self.state_id)
        # set SubgraphFusion properties
        sf.debug = self.debug
        sf.transient_allocation = self.transient_allocation
        sf.schedule_innermaps = self.schedule_innermaps

        if SubgraphFusion.can_be_applied(sdfg, self.subgraph_view(sdfg)):
            print("******** SGF")
            sf.apply(sdfg)

        elif self.allow_tiling and StencilTiling.can_be_applied(sdfg, self.subgraph_view(sdfg)):
            print("******** TILING")
            st = StencilTiling(self.subgraph_view(sdfg), self.sdfg_id, self.state_id)
            # set StencilTiling properties
            st.debug = self.debug
            st.unroll_loops = self.stencil_unroll_loops
            st.strides= self.stencil_strides
            st.apply(sdfg)
            sf.apply(sdfg)

        else:
            raise NotImplementedError("Error")
