""" This module contains classes that implement the cuda-block transformation.
"""

from dace import dtypes, registry, symbolic, subsets
from dace.sdfg import nodes, utils
from dace.memlet import Memlet
from dace.sdfg import replace, SDFG
from dace.transformation import pattern_matching
from dace.properties import make_properties, Property
from dace.symbolic import symstr
from dace.sdfg.propagation import propagate_memlets_sdfg

from dace.frontend.operations import detect_reduction_type
from dace.transformation.heterogeneous import helpers

from dace.transformation.dataflow import LocalStorage

from copy import deepcopy as dcpy
from typing import List, Union

import dace.libraries.standard as stdlib

import timeit



@registry.autoregister_parmas(singlestate=True)
@make_properties
class CUDABlockAllReduce(pattern_matching.Transformation):
    """ Implements the CUDABlockReduce transformation.
        Takes a cuda block reduce node, transforms it to a block reduce node,
        warps it in outer maps and creates an if-output of thread0
        to a newly created shared memory container
    """

    _reduce = stdlib.Reduce()

    debug = Property(desc="Debug Info",
                     dtype = bool,
                     default = True)


    @staticmethod
    def expressions():
        return [
            utils.node_path_graph(CUDABlockReduce._reduce)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict = False):
        reduce_node = graph.nodes()[candidate[CUDABlockReduce._reduce]]
        inedge = graph.in_edges(reduce_node)[0]
        if reduce_node.implementation != 'CUDA (block)':
            return False
        if inedge.data.total_size == 1:
            return False
        if not reduce_node.axes:
            # full reduction, cannot do threadBlocks
            return False
        if len(reduce_node.axes) == len(inedge.data.subset):
            # full reduction, cannot do threadBlocks
            return False

        # good to go
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        reduce = candidate[ReduceMap._reduce]
        return str(reduce)

    def redirect_edge(self, graph, edge, new_src = None, new_src_conn = None ,
                                         new_dst = None, new_dst_conn = None, new_data = None ):
        if not(new_src or new_dst) or new_src and new_dst:
            raise RuntimeError("Redirect Edge has been used wrongly")
        data = new_data if new_data else edge.data
        if new_src:
            ret = graph.add_edge(new_src, new_src_conn, edge.dst, edge.dst_conn, data)
            graph.remove_edge(edge)
        if new_dst:
            ret = graph.add_edge(edge.src, edge.src_conn, new_dst, new_dst_conn, data)
            graph.remove_edge(edge)

        return ret

    def apply(self, sdfg, strict = False):
        """ Create a map around the BlockReduce node
            with in and out transients in registers
            and an if tasklet that redirects the output
            of thread 0 to a shared memory transient
        """

        graph = sdfg.nodes()[self.state_id]
        reduce_node = graph.nodes()[self.subgraph[ReduceMap._reduce]]
        in_edge = graph.in_edges(reduce_node)[0]
        out_edge = graph.out_edge(reduce_ndoe)[0]

        axes = reduce_node.axes

        # first, set reduce_node axes to all
        reduce_node.axes = None

        # add a map that encloses the reduce node
        (new_entry, new_exit) = graph.add_map(
                      name = str(reduce_node)+'_mapped',
                      ndrange = [rng for (i,rng) in enumerate(in_edge.data.subset)],
                      schedule = dtypes.ScheduleType.Default)

        self.redirect_edge(graph, in_edge, new_dst = new_entry)
        self.redirect_edge(graph, out_edge, new_src = new_exit)
        subset_in = subsets.Range([in_edge.data.subset[i] if i not in axes
                                   else new_entry.map.params[i]
                                   for i in range(len(in_edge.data.subset))])
        memlet_in = dace.memlet.Memlet(
                        data = in_edge.data.data,
                        num_accesses = 1,
                        subset = subset_in,
                        vector_length = 1
        )
        memlet_out = dcpy(out_edge.data)
        graph.add_edge(u = new_entry, v = reduce_node, memlet = memlet_in)
        graph.add_edge(u = reduce_node, v = new_exit, memlet_memlet_out)

        # add in and out local storage

        in_local_storage_subgraph = {
            LocalStorage._node_a: new_entry,
            LocalStorage._node_b: reduce_node
        }
        out_local_storage_subgraph = {
            LocalStorage._node_a: reduce_node,
            LocalStorage._node_b: new_exit
        }
        local_storage = LocalStorage(sdfg.sdfg_id,
                                     self.state_id,
                                     in_local_stoarge_subgraph,
                                     0)

        local_storage.array = array_out
        '''
        from dace.transformation.dataflow.local_storage import LocalStorage
        local_storage_subgraph = {
            LocalStorage._node_a: nsdfg.sdfg.nodes()[0].nodes().index(inner_exit),
            LocalStorage._node_b: nsdfg.sdfg.nodes()[0].nodes().index(outer_exit)
        }
        nsdfg_id = nsdfg.sdfg.sdfg_list.index(nsdfg.sdfg)
        nstate_id = 0
        local_storage = LocalStorage(nsdfg_id,
        '''

        # add an if tasket and diverge to it
