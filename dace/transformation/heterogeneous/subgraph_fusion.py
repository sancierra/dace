""" This module contains classes that implement
    subgraph fusion
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




@make_properties
class SubgraphFusion():
    """ Implements the SubgraphFusion transformation.
        Fuses the maps specified in the subgraph into
        one subgraph, creating transients and new connections
        where necessary.
        The subgraph to be inputted has to have been verified
        to be transformable into one map. This module is just
        responsible for the transformation.
        If the corresponding maps have different map ranges / permutations,
        choose the permuted range of maximum size


        This is currently not implemented as a transformation template,
        as we want to input an arbitrary subgraph / collection of maps.

    """


    @staticmethod
    def can_be_applied(sdfg, graph, subgraph):
        # TODO
        """ dummy, do this later in the prematching phase? """
        """ could check:
            - on path between any maps there must not be
              any tasklets / reductions
            - maps must have common permuted base
            - no outgoing wcr on any map that is not at the end
        """
        return True

    def fuse():
        # TODO
        """ do the fusion part here """
        """
            1. Calculate common base
            2. Expand all maps according to base into outer, inner
            3. Get dependency_dict of the maps
            4. Fuse them in valid order
        """
        pass
