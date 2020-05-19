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
        The two input maps have to have the same outer access sets and ranges
    """

    # we use this just for pattern recognition in graph

    _first_entry = nodes.EntryNode()
    _second_entry = nodes.EntryNode()

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
                Fusion._first_entry,
                Fusion._second_map_entry
            )
        ]

    # only for transformation API use
    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict = False):
        # parts of these are copied over from map_fusion.py
        first_entry = graph.nodes()[candidate[MapFusion._first_entry]]
        second_entry = graph.nodes()[candidate[MapFusion._second_entry]]
        # when can we fuse map?
        # 1. They have the same access sets and ranges in order
        # 2. Second map does not lead to first map (no duplicates)
        # 3. If there are any outgoing edges from the first map that lead to
        #    the second map, then there can only be an access node in between
        #    (no tasklet, no reduction, etc.).
        #    Also there must be no WCR on that edge.
        #    Else the maps are sequentially non-fusable

        # TODO
        # 1. They have the same access sets and ranges in order
        if len(first_entry.map.range) != len(second_entry.map.range):
            return False
        if not all(element1 == element2      \
                    for (element1, element2) \
                    in zip(first_entry.map.range.rng, second_entry.map.range.rng)):
            return False
        if not all(element1 == element2
                    for (element1, element2)
                    in zip(first_entry.map.params, second_entry.map.params)):
            return False

        """
        # old code for max_permuted_outer -> TODO: refactor find_max_permuted_outer
        (map_range, result) = helpers.find_max_permuted_outer(
                                      maps = [first_map, second_map])
        if len(map_range) == 0:
            return False
        """


        # 2. Second map does not lead to first map (no duplicates)
        # Do it the easy but inefficient way here -> DFS
        queue = [graph.exit_node(second_entry)]
        while len(queue) > 0:
            current = queue.pop(0)
            if current == first_entry:
                return False
            else:
                queue.extend(graph.out_edges(current))

        # 3. If there are any outgoing edges from the first map that lead to
        #    the second map, then there can only be an access node in between
        #    (no tasklet, no reduction, etc.).
        #    Also there must be no WCR on that edge.
        #    Else the maps are sequentially non-fusable

        first_exit = graph.exit_node(first_entry)

        first_exit_nodes = set([edge.dst for edge in graph.out_edges(first_exit)])
        for node in first_exit_nodes:
            if isinstance(node, nodes.AccessNode):
                for _, _, dst, _, _ in graph.out_edges(node):
                    if dst != first_entry:
                        if helpers.path_from_to(dst, second_entry):
                            return False
                    else:
                        # see that we don't have wcr on node in_edge
                        if graph.in_edges(node)[0].wcr is not None:
                            return False

            else:
                if helpers.path_from_to(node, second_entry, graph):
                    return False


        # It's a match
        return True

    def apply(self,sdfg):
        """
        Fuse the two maps by splitting both maps in outer and inner maps,
        where the outer maps have the same permuted ranges. Connect inner
        graphs appropriately. If there are any in-between transients in case
        of sequential fusion, they either get discarded or copied if there
        are other dependencies.
        """

        graph = sdfg.nodes()[self.state_id]
        first_entry = graph.nodes()[self.subgraph[Fusion._first_entry]]
        second_entry = graph.nodes()[self.subgraph[Fusion._second_entry]]
        first_exit = graph.exit_node(first_entry)
        second_exit = graph.exit_node(second_entry)

        # if not defined as class property
        outer_range = Fusion.outer_range
        if not outer_range:
            outer_range = helpers.

        # compute some useful sets:
        first_nodes = set()
        first_nodes_external = set()
        second_nodes = set()
        second_nodes_external = set()

        common_nodes = set()


        for edge in graph.out_edges(first_exit):
            first_nodes.add(edge.dst)
            for edge_dst in graph.out_edges(edge.dst)
                if edge_dst.dst == second_entry:
                    common_nodes.add(edge.dst)
                else:
                    first_nodes_external.add(edge.dst)

        for edge in graph.in_edges(second_entry):
            second_nodes.add(edge.src)
            for edge_src in graph.out_edges(edge.src):
                if edge_src.src == first_exit:
                    check = edge.src in common_nodes
                    if not check:
                        print("ERROR")
                else:
                    second_nodes_external.add(edge.src)

        # Requirements:

        # begin fusing the maps
        # we add map 2 to the first one
        # use covers for attaching
        # if it doesn't cover, we have to get
        # a new hook at the beginning of the map


        ### Transients:
        # transient in first_nodes_external will be left, but it is connected
        # to the new map exit and linked to the correct result. An additional
        # transient might have to be created inside if it is also in common_nodes!

        # transient in second_nodes_external will be left, but it is connected
        # to the new map entry and linked to the correct place.
        # there can't be an element in both second_nodes_external and common_nodes!
        # (else the state would not be valid)

        # transient only in common_nodes can be removed (TODO: always? what if there is only
        # a subset accessed second time and full save third time and then read at the end)
        # if scalar, directly connect, else replace by new transient inside


        ### Non - transients:
        # if not in common_nodes: easy, same as above
        # if in common_nodes: always need to look at second map exit
        # TODO: SUBSETS!!!
    """
    sketch:

      first_exit
    \____________/
    |   |   |
   /
  1     2   3    4
                /
        |   |   |
    /------------\
     second_entry

    first_nodes = {1,2,3}
    first_nodes_external = {1}

    second_nodes = {2,3,4}
    second_nodes_external = {4}

    common_nodes = {2,3}

    a node is in both common and external if
    there is both a connection to a next map and to somewhere else
    """


if __name__ == "__main__":
    transformation = Fusion(0, 0, {}, 0)
