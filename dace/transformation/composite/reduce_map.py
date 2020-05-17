""" This module contains classes that implement the reduce-map transformation.
"""

from copy import deepcopy as dcpy
from dace import dtypes, registry, symbolic, subsets
from dace.graph import nodes, nxutil
from dace.memlet import Memlet
from dace.sdfg import replace
from dace.transformation import pattern_matching
from typing import List, Union


@registry.autoregister_params(singlestate=True)
@make_properties
class ExpandReduce(pattern_matching.Transformation):
    """ Implements the Reduce-Map transformation.
        Expands a Reduce node into inner and outer map components,
        then introduces a transient for its intermediate output,
        deletes the WCR on the outer map exit
        and transforms the inner map back into a reduction.
    """

    import dace.libraries.standard as stdlib
    _reduce = stdlib.Reduce()

    innermaps_to_reduce = Property(
        dtype = bool,
        desc = "Convert inner Array back to Reduce Object",
        default = True,
        allow_none = False
    )

    @staticmethod
    def expressions():
        return[
            nxutil.node_path_graph(_reduce)
        ]
    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict = False):
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        reduce = candidate[ExpandReduce._reduce]
        return str(reduce)

    def apply(self, sdfg: SDFG, strict = False):
        """ Create two nested maps. The inner map ranges over the reduction
            axis, the outer map over the rest of the axes. An out-transient
            in between is created. The inner map memlets that flow out get
            assigned the reduce node's WCR.
        """
        graph = sdfg.nodes()[self.state_id]
        reduce_node = graph.nodes()[self.subgraph[ReduceExpansion._reduce]]
        out_storage_node = graph.out_edges(reduce_node)[0].dst
        in_storage_node = graph.in_edges(reduce_node)[0].src


        # check whether the new nested sdfg only has one state
        # if not, then there has been a WCR identity assigned
        # -> if not, we can't do anything further
        if reduce_node.identity is not None:
            print(f"Could not successfully fully transform {reduce_node}. \
                    Nested SDFG Expansion has multiple states (-> WCR Identity has been assigned)")
            return

        # assumption: there are accessNodes right before and after
        # the reduce nodes


        # expand the reduce node
        try:
            nsdfg = reduce_node.expand(sdfg)
        except Exception:
            print(f"Aborting: Could not execute expansion in {reduce_node}")

        nstate = nsdfg.sdfg.nodes()[0]
        for node, scope in nstate.scope_dict():
            if isinstance(node, nodes.MapEntry):
                if scope is None:
                    outer_entry = node
                else:
                    inner_entry = node

        inner_exit = nstate.exit_node(inner_entry)
        outer_exit = nstate.exit_node(outer_entry)


        ###### create an out transient between inner and outer map exit
        array = nstate.in_edges(outer_entry)[0].data.data
        print("Data being reduced:", array)

        from dace.transformation.dataflow.local_storage import LocalStorage
        local_storage_subgraph = {
            LocalStorage._node_a: inner_exit,
            LocalStorage._node_b: outer_exit
        }
        nsdfg_id = nsdfg.sdfg.sdfg_list.index(nsdfg.sdfg)
        state_id = graph.state_id

        local_storage = LocalStorage(nsdfg_id, state_id,
                                     local_storage_subgraph, 0)
        local_storage.array = array
        local_storage.apply(nsdfg.sdfg)
        transient_node_inner = local_storage._data_node



        # find earliest parent read-write occurrence of array_out
        # do BFS, best complexity
        # O(V + E)

        queue = [nsdfg]
        array_closest_ancestor = None
        while len(queue) > 0:
            current = queue.pop(0)
            if isinstance(current, nodes.AccessNode):
                if current.data == array:
                    # it suffices to find the first node
                    # no matter what access (ReadWrite or Read)
                    # can't be Read only, else state would be ambiguous
                    array_closest_ancestor = current
                    break
            queue.extend([in_edge.src for graph.in_edges(current)])



        # if it doesnt exist:
        #           if non-transient: create data node accessing it
        #           if transient: ancestor_node = none, set_zero on outer node

        set_zero_bypass = False
        if array_closest_ancestor is None:
            if array.transient:
                set_zero_bypass = True
                # TODO: also enable for other types of reductions
                storage_node.setzero = True
                nstate.out_edges(transient_node_inner)[0].data.wcr = None
                nstate.out_edges(outer_exit)[0].data.wcr = None
            else:
                array_closest_ancestor = nodes.AccessNode(storage_node.data,
                                            access = dtypes.AccessType.ReadOnly)
                graph.add_node(array_closest_ancestor)


        # array_closest_ancestor now points to the node we want to connect
        # to the map entry

        # first, inline fuse back our NSDFG
        from dace.transformations.interstate import InlineSDFG
        # debug
        if InlineSDFG.can_be_applied(graph, nsdfg, 0, sdfg) is False:
            print("ERROR: This should not appear")
        inline_sdfg = InlineSDFG(sdfg.id, graph.id, nsdfg, 0)
        inline_sdfg.apply(sdfg)


        # TODO: create memlet from that ancestor node (if not none) to outer_map entry
        if array_closest_ancestor:
            '''
            new_memlet = Memlet(data = out_storage_node.data,
                                num_accesses = out_storage_node.data.total_size,
                                subset = out_storage_node.data.shape,
                                vector_length = 1)
            graph.add_edge(array_closest_ancestor, None, outer_map_entry, new_memlet)
            '''
            # TODO: create tasklet
            new_tasklet = graph.add_tasklet(name = "reduction_transient_update",
                                            inputs = {"reduction_in", "array_in"},
                                            outputs = {"out"},
                                            code = """
                                            out = reduction_in + array_in
                                            """)
            # TODO: connect tasklet with transient and input ancestor path
            new_memlet_array = Memlet(data = out_storage_node.data,
                                      num_accesses = 1,
                                      subset = dace.subsets.Indices)
            graph.add_memlet_path(array_closest_ancestor,
                                  outer_map_entry,
                                  new_tasklet,
                                  memlet = XXXX
                                  src_conn = None,
                                  dst_conn = 'array_in')

            graph.add_memlet_path()


        # TODO: remove wcr on outer edges

        # TODO: re-create reduction node inside

        # TODO: create setzero identities for other reductions than +


        # fuse again
        if strict:
            sdfg.apply_strict_transformations()
