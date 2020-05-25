""" This module contains classes that implement
    subgraph fusion
"""

from dace import dtypes, registry, symbolic, subsets
from dace.graph import nodes, nxutil
from dace.memlet import Memlet, EmptyMemlet
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
        Use MultiExpansion first before fusing a graph with SubgraphFusion


        This is currently not implemented as a transformation template,
        as we want to input an arbitrary subgraph / collection of maps.

    """


    @staticmethod
    def can_be_applied(sdfg, graph, maps):

        # TODO: Not really a priority right now

        # Fusable if
        # 1. Maps have the same access sets and ranges in order
        # 2. Any nodes in between are AccessNodes only without WCR
        #    There is at least one AccessNode only between two maps
        # 3a Every array that is in between two maps must never appear
        #    in another write/read (practically write) node in the entire sdfg.
        #    The array has to be transient and transient only in this state
        # 3b Every array that is in between two maps can only appear once
        #    in a write node within the maps subgraph
        # 3c No restrictions

        return True

    def redirect_edge(self, graph, edge, new_src = None, new_src_conn = None ,
                                         new_dst = None, new_dst_conn = None, new_data = None ):
        if not(new_src or new_dst) or new_src and new_dst:
            raise RuntimeError("Redirect Edge has been used wrongly")
        data = new_data if new_data else edge.data
        if new_src:
            graph.add_edge(new_src, new_src_conn, edge.dst, edge.dst_conn, data)
            graph.remove_edge(edge)
        if new_dst:
            graph.add_edge(edge.src, edge.src_conn, new_dst, new_dst_conn, data)
            graph.remove_edge(edge)


    def _create_transients(self, sdfg, graph, in_nodes, out_nodes, intermediate_nodes, map_entries, do_not_delete = None):
        # handles arrays that are
        # and connect

        def redirect(redirect_node, original_node):
            # redirect all traffic to original node to redirect node
            # and then create a path from redirect to original
            nxutil.change_edge_dest(graph, original_node, redirect_node)
            graph.add_edge(redirect_node,
                           None,
                           original_node,
                           None,
                           EmptyMemlet())
            for edge in graph.out_edges(original_node):
                if edge.dst in map_entries:
                    #edge.src = redirect_node
                    self.redirect_edge(graph, edge, new_src = redirect_node)


        transient_dict = {}
        for node in (intermediate_nodes & out_nodes):
            data_ref = sdfg.data(node.data)
            trans_data_name = node.data + '_tmp'

            data_trans = sdfg.add_transient(name=trans_data_name,
                                            shape= data_ref.shape,
                                            dtype= data_ref.dtype,
                                            storage= data_ref.storage,
                                            offset= data_ref.offset)
            node_trans = graph.add_access(trans_data_name)
            redirect(node_trans, node)
            transient_dict[node_trans] = node

        return transient_dict

    def _create_transients_NOTRANS(sdfg, graph, in_nodes, out_nodes, intermediate_nodes, map_entries, do_not_delete = None):
        # better preprocessing which allows for non-transient in between nodes
        # handles arrays that are
        # and connect

        def redirect(redirect_node, original_node):
            # redirect all traffic to original node to redirect node
            # and then create a path from redirect to original
            # outgoing edges to other maps in our subset should be originated from the clone
            nxutil.change_edge_dest(graph, original_node, redirect_node)
            graph.add_edge(src = redirect_node,
                           src_conn = None,
                           dst = original_node,
                           dst_conn = None,
                           memlet = EmptyMemlet())
            for edge in graph.out_edges(original_node):
                if edge.dst in map_entries:
                    #edge.src = redirect_node
                    self.redirect_edge(graph, edge, new_src = redirect_node)


        transient_dict = {}
        for node in (intermediate_nodes):
            if node in out_nodes \
               or node in do_not_delete \
               or sdfg.data(node.data).is_transient == False:

                data_ref = sdfg.data(node.data)
                trans_data_name = data_ref.name + '_tmp'

                data_trans = sdfg.add_transient(name=trans_data_name,
                                                shape= data_ref.shape,
                                                dtype= data_ref.dtype,
                                                storage= data_ref.storage,
                                                offset= data_ref.offset)
                out_node_trans = graph.add_access(data_trans)
                redirect(out_node_trans, out_node)
                transient_dict[out_node_trans] = out_node

        return transient_dict

    def fuse(self, sdfg, graph, map_entries, **kwargs):
        # WORK IN PROGRESS
        maps = [map_entry.map for map_entry in map_entries]
        map_exits = [graph.exit_node(map_entry) for map_entry in map_entries]

        # Nodes that flow into one or several maps but no data is flowed to them from any map
        in_nodes = set()

        # Nodes into which data is flowed but that no data flows into any map from them
        out_nodes = set()

        # Nodes that act as intermediate node - data flows from a map into them and then there
        # is an outgoing path into another map
        intermediate_nodes = set()

        """ NOTE:
        - in_nodes, out_nodes, intermediate_nodes refer to the configuration of
          the final fused map
        - in_nodes and out_nodes are trivially disjoint
        - Intermediate_nodes and out_nodes are not necessarily disjoint
        - Intermediate_nodes and in_nodes SHOULD be disjoint in a valid sdfg.
          Else there could always be a race condition....
        """

        for map_entry, map_exit in zip(map_entries, map_exits):
            for edge in graph.in_edges(map_entry):
                in_nodes.add(edge.src)
            for edge in graph.out_edges(map_exit):
                current_node = edge.dst
                if len(graph.out_edges(current_node)) == 0:
                    out_nodes.add(current_node)
                else:
                    for dst_edge in graph.out_edges(current_node):
                        if isinstance(dst_edge.dst, nodes.MapEntry):
                            if dst_edge.dst in map_entries:
                                intermediate_nodes.add(current_node)
                        else:
                            out_nodes.add(current_node)

        # any intermediate_nodes currently in in_nodes shouldnt be there
        in_nodes -= intermediate_nodes


        print("In_nodes", in_nodes)
        print("Out_nodes", out_nodes)
        print("intermediate_nodes", intermediate_nodes)

        # all maps are assumed to have the same params and range in order
        global_map = nodes.Map(label = "outer_fused",
                               params = maps[0].params,
                               ndrange = maps[0].range)
        global_map_entry = nodes.MapEntry(global_map)
        global_map_exit  = nodes.MapExit(global_map)

        # if we make new transients of any objects, we are to save them
        # into this dict. This allows for easy redirection

        try:
            # in-between transients that should be duplicated nevertheless
            do_not_delete = kwargs['do_not_delete']
        except KeyError:
            do_not_delete = None

        transient_dict = self._create_transients(sdfg, graph, in_nodes, out_nodes, intermediate_nodes, map_entries, do_not_delete)
        inconnectors_dict = {}
        # {access_node: (edge, in_conn, out_conn)}
        print("Transient_dict", transient_dict)

        graph.add_node(global_map_entry)
        graph.add_node(global_map_exit)

        for map_entry, map_exit in zip(map, map_entries, map_exits):
            print("Current map:" map_entry)
            # handle inputs
            # TODO: dynamic map range -- this is fairly unrealistic in such a setting
            for edge in graph.in_edges(map_entry):
                src = edge.src
                mmt = graph.memlet_tree(edge)
                out_edges = [child.edge for child in mmt.root.traverse_children()]

                if src in in_nodes:

                    in_conn = None; out_conn = None

                    if src in inconnectors_dict:
                        if not subsets.covers(inconnectors_dict[src][0].data.subset, edge.data.subset):
                            print("Extend range")
                            inconnectors_dict[edge.data.data][0].subset = edge.data.subset

                        in_conn = inconnectors_dict[src][1]
                        out_conn = inconnectors_dict[src][2]

                    else:
                        next_conn = global_map.next_connector()
                        in_conn = 'IN_' + next_conn
                        out_conn = 'OUT_' + next_conn
                        global_map.add_in_connector(in_conn)
                        global_map.add_out_connector(out_conn)

                        inconnectors_dict[src] = (src, in_conn, out_conn)

                        # reroute in edge via global_map_entry
                        edge.dst = global_map_entry
                        edge.dst_conn = in_conn

                    # map out edges to new map
                    for out_edge in out_edges:
                        out_edge.src = global_map_entry
                        out_edge.src_conn = out_conn

                else:

                    # connect directly
                    for out_edge in out_edges:
                        out_edge.src = src

                    graph.remove_edge(edge)

            
            for edge in graph.in_edges(exit_node):
                mmt = graph.memlet_tree(edge)
                out_edges = [child.edge for child in mmt.root.traverse_children()]

                for out_edge in out_edges:
                    dst = out_edge.dst
                    transient_created = None
                    try:
                        dst_original = transient_dict[dst]
                        transient_created = True
                    except KeyError:
                        transient_created = False
                        dst_original = dst


                    if transient_created:
                        # transients only get created for itntermediate_nodes
                        # that are either in out_nodes or non-transient
                        next_conn = global_map.next_connector()
                        in_conn= 'IN_' + next_conn
                        out_conn = 'OUT_' + next_conn
                        global_map.add_in_connector(in_conn)
                        global_map.add_out_connector(out_conn)

                        graph.add_edge(dst, None,
                                       global_map_exit, in_conn,
                                       edge.data)
                        graph.add_edge(global_map_exit, out_conn,
                                       dst_original, None,
                                       outer_edge.data)

                        # additionally, change the shape of the transient data
                        memlet_subset = edge.data.subset
                        sizes = edge.data.subset
                        new_data_size = [s for (sz, s) in zip(sizes, memlet_subset) if sz > 1]
                        underlying_data = sdfg.data(dst.data)
                        underlying_data.shape = new_data_size
                        total_size = 1
                        for s in new_data_size:
                            total_size *= s
                        underlying_data.total_size = total_size


                    # handle separately: intermediate_nodes and pure out nodes
                    if dst_original in intermediate_nodes:
                        out_edge.src = edge.src
                        out_edge.src_conn = edge.src_conn
                        out_edge.data = edge.data

                    if dst_original in (out_nodes - intermediate_nodes):

                        if edge.dst != global_map_exit:
                            next_conn = global_map.next_connector()
                            in_conn= 'IN_' + next_conn
                            out_conn = 'OUT_' + next_conn
                            global_map.add_in_connector(in_conn)
                            global_map.add_out_connector(out_conn)
                            edge.dst = global_map_exit
                            edge.dst_conn = in_conn

                        else:
                            conn_nr = edge.dst_conn[3:]
                            in_conn = 'IN_' + conn_nr
                            out_conn = 'OUT_' + conn_nr


                        # map
                        out_edge.src = global_map_exit
                        out_edge.src_conn = out_conn
                        out_edge.dst = dst
                        out_edge.dst_conn = None


                # remove the edge if it has not been used by any pure out node
                if edge.dst != global_map_exit:
                    graph.remove_edge(edge)


            graph.remove_node(map_entry)
            graph.remove_node(map_exit)
