""" This module contains classes that implement a composite
    Subgraph Fusion - Stencil Tiling Transformation
"""


import dace

import dace.sdfg.transformation as transformation
from dace.sdfg.transformation.subgraph import SubgraphFusion, StencilTiling
from dace import dtypes, registry, symbolic, subsets, data


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


    @staticmethod
    def can_be_applied(sdfg: SDFG, subgraph: SubgraphView) -> bool:
        if SubgraphFusion.can_be_applied(sdfg, subgraph):
            return True
        if self.allow_tiling:
            if StencilTiling.can_be_applied(sdfg, subgraph):
                return True

        return False

    def apply(self, sdfg):
        st = StencilTiling(self.subgraph, self.sdfg_id, self.state_id)
        sf = SubgraphFUsion(self.subgraph, self.sdfg_id, self.state_id)

        # set all the properties
