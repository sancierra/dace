""" This module contains classes that implement
    the fusion transformation
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

@registry.autoregister_params(singlestate=True)
@make_properties
class Fusion(pattern_matching.Transformation):
    """ Implements the Fusion transformation.
        Similar to MapFusion, it fuses two maps A and B together.
        Input data of B might be dependent of output data of A.
        AccessNodes in between the two maps are deleted
        if there no out connections of that AccessNode apart from
        the Memlet to the other map. Appropriate transients
        are created automatically
    """

    # we use this just for pattern recognition in graph

    _first_map_entry = nodes.ExitNode()
    _second_map_entry = nodes.EntryNode()

    outer_range = Property(
        dtype = List, # list of subsets, cannot do nested type validation
        desc = "List of outer Ranges the transformation should look for. \
                If none, automatically call find_max_permuted_outer",
        default = None,
        allow_none = True
    )

    @staticmethod
    def expressions():
        return [
            helpers.non_connected_graph(
                Fusion._first_map_entry,
                Fusion._second_map_entry
            )
        ]

    # only for transformation API use
    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict = False):
        # parts of these are copied over from map_fusion.py
        first_map_entry = graph.nodes()[candidate[MapFusion._first_map_entry]]
        second_map_entry = graph.nodes()[candidate[MapFusion._second_map_entry]]
        # when can we fuse map?
        # 1. They have at least one range in common
        # 2. Second map does not lead to first map (no duplicates)
        # 3. If there are any outgoing edges from the first map that lead to
        #    the second map, then there can only be an access node in between
        #    (no tasklet, no reduction, etc.).
        #    Also there must be no WCR on that edge.
        #    Else the maps are sequentially non-fusable

        # TODO: Next

    def apply(self,sdfg):
        """
        Fuse the two maps by splitting both maps in outer and inner maps,
        where the outer maps have the same permuted ranges. Connect inner
        graphs appropriately. If there are any in-between transients in case
        of sequential fusion, they either get discarded or copied if there
        are other dependencies.
        """

        #TODO: Next




if __name__ == "__main__":
    transformation = Fusion(0, 0, {}, 0)
