""" This module contains classes that implement subgraph fusion
"""
import dace

from dace import dtypes, registry, symbolic, subsets, data
from dace.sdfg import nodes, utils, replace, SDFG, scope_contains_scope
from dace.memlet import Memlet
from dace.transformation import pattern_matching
from dace.properties import make_properties, Property
from dace.symbolic import symstr
from dace.sdfg.propagation import propagate_memlets_sdfg, propagate_memlet

from copy import deepcopy as dcpy
from typing import List, Union

import dace.libraries.standard as stdlib

'''
TODO:

- revamp                        [OK]   TEST: failed
- other_subset                  [OK]   TEST: failed
- can_be_applied()              [In progress]
- StorageType Inference         [OK]   cancelled
- cover intermediate nodes with incoming edges from outside
- stencils

'''

@make_properties
class SubgraphFusion(pattern_matching.SubgraphTransformation):
    """ Implements the SubgraphFusion transformation.
        Fuses the maps specified together with their outer maps
        as a global outer map, creating transients and new connections
        where necessary.
        Use MultiExpansion first before fusing a graph with SubgraphFusion
        Applicability checks have not been implemented yet.

        This is currently not implemented as a transformation template,
        as we want to input an arbitrary subgraph / collection of maps.

    """

    debug = Property(desc = "Show debug info",
                     dtype = bool,
                     default = True)

    cuda_transient_allocation = Property(desc = "Storage Location to push"
                                                "transients to in GPU environment",
                                         dtype = str,
                                         default = "local",
                                         choices = ["auto", "shared", "local", "default"])
    @staticmethod
    def match(sdfg, subgraph):

        # Fusable if
        # 1. Maps have the same access sets and ranges in order
        # 2. Any nodes in between two maps are AccessNodes only, without WCR
        #    There is at most one AccessNode only on a path between two maps,
        #    no other nodes are allowed
        # 3. The exiting memlets' subsets to an intermediate edge must cover
        #    the respective incoming memlets' subset into the next map

        graph = subgraph.graph
        for node in subgraph.nodes():
            if node not in graph.nodes():
                return False

        # next, get all the maps
        map_entries = helpers.get_lowest_scope_maps(sdfg, graph, subgraph)
        map_exits = [graph.exit_node(map_entry) for map_entry in map_entries]

        # 1. check whether all map ranges and indices are the same
        if len(maps) <= 1:
            return False
        base_map = map[0]
        for map in maps:
            if map.get_param_num() != base_map.get_param_num():
                return False
            if not all([p1 == p2 for (p1,p2) in zip(map.params, base_map.params)]):
                return False
            if not map.range == base_map.range:
                return False

        # 2. check intermediate feasiblility
        # see map_fusion.py for similar checks
        # we are being more relaxed here

        # 2.1 do some preparation work first:
        # calculate all out_nodes and intermediate_nodes
        # definition see in apply()
        intermediate_nodes = set()
        out_nodes = set()
        for map_entry, map_exit in zip(map_entries, map_exits):
            for edge in graph.out_edges(map_exit):
                current_node = edge.dst
                if len(graph.out_edges(current_node)) == 0:
                    out_nodes.add(current_node)
                else:
                    for dst_edge in graph.out_edges(current_node):
                        if dst_edge.dst in map_entries:
                            intermediate_nodes.add(current_node)
                        else:
                            out_nodes.add(current_node)

        # 2.2 topological feasibility:
        # For each intermediate and out node: must never reach any map
        # entry if it is not connected to map entry immediately
        visited = set()
        # for memoization purposes
        def visit_descendants(graph, node, visited, map_entries):
            # if we have already been at this node
            if node in visited:
                return True
            # not necessary to add if there aren't any other in connections
            if graph.in_edges(node) > 1:
                visited.add(node)
            for oedge in graph.out_edges(node):
                if not visit_descendants(graph, oedge.dst, visited, map_entries):
                    return False
            return True

        for node in intermediate_nodes | out_nodes:
            # these nodes must not lead to a map entry
            nodes_to_check = set()
            for oedge in graph.out_edges(node):
                if oedge.dst not in map_entries:
                    nodes_to_check.add(oedge.dst)

            for forbidden_node in nodes_to_check:
                if not visit_descandants(graph, forbidden_node, visited, map_entries):
                    return False

        del visited

        # 2.3 memlet feasibility
        # For each intermediate node, look at whether incoming inner map memlets
        # cover outgoing inner map memlets
        # check for WCRs on the fly
        for node in intermediate_nodes:
            upper_subsets = set()
            lower_subsets = set()
            # find upper_subsetes
            for in_edge in graph.in_edges(node):
                # first check for WCRs
                if in_edge.wcr:
                    return False
                if in_edge.src in map_exits:
                    subset = graph.memlet_path(in_edge)[1].subset
                    upper_subsets.add(subset)
                else:
                    raise NotImplementedError("TODO")

            for out_edge in graph.out_edges(node):
                if out_edge.dst in map_exits:
                    for idx, memlet in enumerate(graph.memlet_tree(out_edge)):
                        if idx > 0 and memlet.data == node.data:
                            lower_subsets.add(memlet.subset)

            union_upper = subsets.Range()
            for subs in upper_subsets:
                union_upper = subsets.union(union_upper, subs)
            union_lower = subsets.Range()
            for subs in lower_subsets:
                union_lower = subsets.union(union_lower, subs)


            # finally check coverage
            if not union_upper.covers(union_lower):
                # TODO: Implement special case stencil smem.
                return False
            # TODO: The following restriction is too harsh,
            # but relaxing it is hard to do in an elegant way.
            for subs_lo in union_lower:
                covered = False
                for subs_hi in union_higher:
                    if subs_hi.covers(subs_lo):
                        covered = True
                        break
                if not covered:
                    print('[WIP] WARNING: Please check subsets manually')
                    return False


        '''
        #####################################################################
        first_map_exit = graph.nodes()[candidate[MapFusion._first_map_exit]]
        first_map_entry = graph.entry_node(first_map_exit)
        second_map_entry = graph.nodes()[candidate[
            MapFusion._second_map_entry]]

        # NOTE: check whether WCR in intermediate node

        for _in_e in graph.in_edges(first_map_exit):
            if _in_e.data.wcr is not None:
                for _out_e in graph.out_edges(second_map_entry):
                    if _out_e.data.data == _in_e.data.data:
                        # wcr is on a node that is used in the second map, quit
                        return False




        # NOTE: Check whether there is a pattern map -> access -> map.
        # NOTE: just create the sets. we don't need more.
        intermediate_nodes = set()
        intermediate_data = set()
        for _, _, dst, _, _ in graph.out_edges(first_map_exit):
            if isinstance(dst, nodes.AccessNode):
                intermediate_nodes.add(dst)
                intermediate_data.add(dst.data)

                # If array is used anywhere else in this state.
                num_occurrences = len([
                    n for n in graph.nodes()
                    if isinstance(n, nodes.AccessNode) and n.data == dst.data
                ])
                if num_occurrences > 1:
                    return False
            else:
                return False


        # NOTE: Need to do something more general here
        # Check if any intermediate transient is also going to another location
        second_inodes = set(e.src for e in graph.in_edges(second_map_entry)
                            if isinstance(e.src, nodes.AccessNode))
        transients_to_remove = intermediate_nodes & second_inodes
        # if any(e.dst != second_map_entry for n in transients_to_remove
        #        for e in graph.out_edges(n)):
        if any(graph.out_degree(n) > 1 for n in transients_to_remove):
            return False


        # NOTE: need something like these memlet checks
        # AFTER having done
        # Check that input set of second map is provided by the output set
        # of the first map, or other unrelated maps
        for second_edge in graph.out_edges(second_map_entry):
            # Memlets that do not come from one of the intermediate arrays
            if second_edge.data.data not in intermediate_data:
                # however, if intermediate_data eventually leads to
                # second_memlet.data, need to fail.
                for _n in intermediate_nodes:
                    source_node = _n
                    destination_node = graph.memlet_path(second_edge)[0].src
                    # NOTE: Assumes graph has networkx version
                    if destination_node in nx.descendants(
                            graph._nx, source_node):
                        return False
                continue

            provided = False

            # Compute second subset with respect to first subset's symbols
            sbs_permuted = dcpy(second_edge.data.subset)
            sbs_permuted.replace({
                symbolic.pystr_to_symbolic(k): symbolic.pystr_to_symbolic(v)
                for k, v in params_dict.items()
            })

            for first_memlet in out_memlets:
                if first_memlet.data != second_edge.data.data:
                    continue

                # If there is a covered subset, it is provided
                if first_memlet.subset.covers(sbs_permuted):
                    provided = True
                    break

            # If none of the output memlets of the first map provide the info,
            # fail.
            if provided is False:
                return False
        '''
        return True

    def storage_type_inference(node):
        ''' on a fused graph, looks in which memory intermediate node
            needs to be pushed to
        '''
        raise NotImplementedError("WIP")

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

    def preapare_intermediate_nodes(self, sdfg, graph, in_nodes, out_nodes,
                                    intermediate_nodes, map_entries, map_exits,
                                    intermediate_data_counter):

        def redirect(redirect_node, original_node):
            # redirect all outgoing traffic which
            # does not enter fusion scope again
            # from original_node to redirect_node
            # and then create a path from original_node to redirect_node.

            edges = list(graph.out_edges(original_node))
            for edge in edges:
                if edge.dst not in map_entries:
                    self.redirect_edge(graph, edge, new_src = redirect_node)

            graph.add_edge(original_node, None,
                           redirect_node, None,
                           Memlet())

        # first search whether intermediate_nodes appear outside of subgraph
        # and store it in dict
        data_counter = defaultdict(int)
        # do a full global search and count each data from each intermediate node
        scope_dict = graph.scope_dict()
        data_intermediate = set([node.data for node in intermediate_nodes])
        def search(sdfg, graph, intermediate_nodes, data_counter):
            for state in sdfg.nodes():
                for node in state.nodes():
                    if node.data in data_intermediate:
                        data_counter[node.data] += 1
                    if isinstance(node, nodes.NestedSDFG):
                        # check whether NestedSDFG is part of our subgraph
                        # if so, we can pass
                        nestedSearch = False
                        if state != graph:
                            nestedSearch = True
                        else:
                            if self.subgraph and node not in self.subgraph:
                                # if called via apply() just use self.subgraph
                                nestedSearch = True
                            elif not self.subgraph and \
                                 any([scope_contains_scope(scope_dict, map_entry, node) \
                                      for map_entry in map_entries]):
                                nestedSearch = True
                        if nestedSearch:
                            search(node.sdfg, graph, intermediate_nodes, data_counter)

        # finally: If intermediate_counter and global counter match and if the array
        # is declared transient, it is fully contained by the subgraph
        subgraph_contains_data = {data: data_counter[data] == intermediate_data_counter[data] \
                                        and sdfg.data(data).transient \
                                  for data in data_intermediate}

        transients_created = {}
        for node in intermediate_nodes & out_nodes:
            # create new transient at exit replacing the array
            # and redirect all traffic
            data_ref = sdfg.data(node.data)
            out_trans_data_name = node.data + '_OUT'
            data_trans = sdfg.add_transient(name = out_trans_data_name,
                                            shape = data_ref.shape,
                                            dtype = data_ref.dtype,
                                            storage= data_ref.storage,
                                            offset = data_ref.offset)
            node_trans = graph.add_access(trans_data_name)
            redirect(node_trans, node)
            transients_created[node] = node_trans

        return (subgraph_contains_data, transients_created)


    def _create_transients(self,sdfg, graph, in_nodes, out_nodes, intermediate_nodes, \
                           map_entries, map_exits, do_not_override = []):
        # TODO: ready to delete
        # better preprocessing which allows for non-transient in between nodes
        ''' creates transients for every in-between node that has exit connections or that is non-transient (or both)
            the resulting transient can then later be pushed into the map
        '''

        def redirect(redirect_node, original_node):
            # redirect all traffic to original node to redirect node
            # and then create a path from redirect to original
            # outgoing edges to other maps in our subset should be originated from the clone
            # similar to utils.change_edge_dest(graph, original_node, redirect_node)
            # but only when edge comes from a map exit
            edges = list(graph.in_edges(original_node))
            for e in edges:
                if e.src in map_exits:
                    graph.remove_edge(e)
                    if isinstance(e, dace.sdfg.graph.MultiConnectorEdge):
                        graph.add_edge(e.src, e.src_conn, redirect_node, e.dst_conn, e.data)
                    else:
                        graph.add_edge(e.src, redirect_node, e.data)


            graph.add_edge(redirect_node,
                           None,
                           original_node,
                           None,
                           Memlet())
            for edge in graph.out_edges(original_node):
                if edge.dst in map_entries:
                    #edge.src = redirect_node
                    self.redirect_edge(graph, edge, new_src = redirect_node)


        transient_dict = {}
        for node in (intermediate_nodes):
            if node in out_nodes \
               or node in do_not_override \
               or not sdfg.data(node.data).transient:

                data_ref = sdfg.data(node.data)
                trans_data_name = node.data + '__trans'

                data_trans = sdfg.add_transient(name=trans_data_name,
                                                shape= data_ref.shape,
                                                dtype= data_ref.dtype,
                                                storage= data_ref.storage,
                                                offset= data_ref.offset)
                node_trans = graph.add_access(trans_data_name)
                redirect(node_trans, node)
                transient_dict[node_trans] = node

        return transient_dict


    def apply(self, sdfg, subgraph, **kwargs):
        self.subgraph = subgraph

        graph = subgraph.graph

        maps = helpers.get_lowest_scope_maps(sdfg, graph, subgraph)
        self.fuse(sdfg, graph, map_entries, **kwargs)

    def fuse(self, sdfg, graph, map_entries, **kwargs):
        """ takes the map_entries specified and tries to fuse maps.

            all maps have to be extended into outer and inner map
            (use MapExpansion as a pre-pass)

            Arrays that don't exist outside the subgraph get pushed
            into the map and their data dimension gets cropped.
            Otherwise the original array is taken.

            For every output respective connections are crated automatically.

            See can_be_applied for requirements.

            [Work in Progress] Features and corner cases not supported yet:
            - border memlets with subset changes (memlet.other_subset)
              are not supported currently
            - Transients that get pushed into the global map
              always persist, even if they have size one (unlike MapFusion)

            :param sdfg: SDFG
            :param graph: State
            :param map_entries: Map Entries (class MapEntry) of the outer maps
                                which we want to fuse
            :param do_not_override: List of AccessNodes that are transient but
                                  should not be directly modified when pushed
                                  into the global map. Instead a transient copy
                                  is created and linked to it.
        """

        # if there are no maps, return immediately
        if len(map_entries) == 0:
            return

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
        - in_nodes, out_nodes, intermediate_nodes refer to the configuration of the final fused map
        - in_nodes and out_nodes are trivially disjoint
        - Intermediate_nodes and out_nodes are not necessarily disjoint
        - Intermediate_nodes and in_nodes are disjoint by design.
          There could be a node that has both incoming edges from a map exit
          and from outside, but it is just treated as intermediate_node and handled
          automatically.
        """
        # needed later for determining whether data is contained in
        # subgraph
        intermediate_data_counter = defaultdict(int)

        for map_entry, map_exit in zip(map_entries, map_exits):
            for edge in graph.in_edges(map_entry):
                in_nodes.add(edge.src)
            for edge in graph.out_edges(map_exit):
                current_node = edge.dst
                if len(graph.out_edges(current_node)) == 0:
                    out_nodes.add(current_node)
                else:
                    for dst_edge in graph.out_edges(current_node):
                        if dst_edge.dst in map_entries:
                            intermediate_nodes.add(current_node)
                            # add to data counter
                            intermediate_data_counter[current_node.data] += 1
                        else:
                            out_nodes.add(current_node)
                for e in graph.in_edges(current_node):
                    if e.src not in map_exits:
                        # TODO: cover this last special case as well
                        raise NotImplementedError("Not implemented yet")

        # any intermediate_nodes currently in in_nodes shouldnt be there
        in_nodes -= intermediate_nodes

        if self.debug:
            print("SubgraphFusion::In_nodes", in_nodes)
            print("SubgraphFusion::Out_nodes", out_nodes)
            print("SubgraphFusion::Intermediate_nodes", intermediate_nodes)

        # all maps are assumed to have the same params and range in order
        global_map = nodes.Map(label = "outer_fused",
                               params = maps[0].params,
                               ndrange = maps[0].range)
        global_map_entry = nodes.MapEntry(global_map)
        global_map_exit  = nodes.MapExit(global_map)

        # assign correct schedule to global_map_entry
        # TODO: move to can_be_applied
        schedule = map_entries[0].schedule
        if not all([entry.schedule == schedule for entry in map_entries]):
            raise RuntimeError("Not all the maps have the same schedule. Cannot fuse.")

        global_map_entry.schedule = schedule
        graph.add_node(global_map_entry)
        graph.add_node(global_map_exit)


        try:
            # in-between transients whose data should not be modified
            do_not_override = kwargs['do_not_override']
        except KeyError:
            do_not_override = []

        # next up, for any intermediate node, find whether it only appears
        # in the subgraph or also somewhere else / as an input
        # create new transients for nodes that are in out_nodes and
        # intermediate_nodes simultaneously
        node_info = self.prepare_intermediate_nodes(sdfg, graph, in_nodes, out_nodes, \
                                                    intermediate_nodes,\
                                                    map_entries, map_exits, \
                                                    intermediate_data_counter, \
                                                    do_not_override)

        (subgraph_contains_data, transients_created) = node_info

        inconnectors_dict = {}
        # Dict for saving incoming nodes and their assigned connectors
        # Format: {access_node: (edge, in_conn, out_conn)}

        for map_entry, map_exit in zip(map_entries, map_exits):
            # handle inputs
            # TODO: dynamic map range -- this is fairly unrealistic in such a setting
            for edge in graph.in_edges(map_entry):
                src = edge.src
                mmt = graph.memlet_tree(edge)
                out_edges = [child.edge for child in mmt.root().children]

                if src in in_nodes:
                    in_conn = None; out_conn = None
                    if src in inconnectors_dict:
                        if not inconnectors_dict[src][0].data.subset.covers(edge.data.subset):
                            if debug:
                                print("SubgraphFusion::Extend range")
                            inconnectors_dict[edge.data.data][0].subset = edge.data.subset

                        in_conn = inconnectors_dict[src][1]
                        out_conn = inconnectors_dict[src][2]
                        graph.remove_edge(edge)

                    else:
                        next_conn = global_map_entry.next_connector()
                        in_conn = 'IN_' + next_conn
                        out_conn = 'OUT_' + next_conn
                        global_map_entry.add_in_connector(in_conn)
                        global_map_entry.add_out_connector(out_conn)

                        inconnectors_dict[src] = (edge, in_conn, out_conn)

                        # reroute in edge via global_map_entry
                        self.redirect_edge(graph, edge, new_dst = global_map_entry, \
                                                        new_dst_conn = in_conn)
                        #edge.dst = global_map_entry, edge.dst_conn = in_conn

                    # map out edges to new map
                    for out_edge in out_edges:
                        self.redirect_edge(graph, out_edge, new_src = global_map_entry, \
                                                            new_src_conn = out_conn)
                        #out_edge.src = global_map_entry
                        #out_edge.src_conn = out_conn

                else:
                    # connect directly
                    for out_edge in out_edges:
                        mm = dcpy(out_edge.data)
                        self.redirect_edge(graph, out_edge, new_src = src, new_data = mm)

                    graph.remove_edge(edge)


            for edge in graph.out_edges(map_entry):
                # special case: for nodes that have no data connections
                if not edge.src_conn:
                    self.redirect_edge(graph, edge, new_src = global_map_entry)

            ######################################

            for edge in graph.in_edges(map_exit):
                if not edge.dst_conn:
                    # no destination connector, path ends here.
                    self.redirect_edge(graph, edge, new_dst = global_map_exit)
                    continue
                # find corresponding out_edges for current edge, cannot use mmt anymore
                out_edges = [oedge for oedge in graph.out_edges(map_exit)
                                      if oedge.src_conn[3:] == edge.dst_conn[2:]]

                # Tuple to store in/out connector port that might be created
                port_created = None

                for out_edge in out_edges:
                    dst = out_edge.dst

                    if dst in intermediate_nodes & out_nodes:
                        # create connection thru global map from
                        # dst to dst_transient that was created
                        dst_transient = transients_created[dst]
                        next_conn = global_map_exit.next_connector()
                        in_conn= 'IN_' + next_conn
                        out_conn = 'OUT_' + next_conn
                        global_map_exit.add_in_connector(in_conn)
                        global_map_exit.add_out_connector(out_conn)

                        graph.add_edge(dst, None,
                                       global_map_exit, in_conn,
                                       dcpy(edge.data))
                        graph.add_edge(global_map_exit, out_conn,
                                       dst_transient, None,
                                       dcpy(out_edge.data))

                        # remove edge from dst to dst_transient that was created
                        # in intermediate preparation.
                        for e in graph.out_edges(dst):
                            if e.dst == dst_transient:
                                graph.remove_edge(e)
                                removed = True
                                break

                        if self.debug:
                            assert removed == True

                    # handle separately: intermediate_nodes and pure out nodes
                    # case 1: intermediate_nodes: can just redirect edge
                    if dst in intermediate_nodes:
                        self.redirect_edge(graph, out_edge, new_src = edge.src,
                                                            new_src_conn = edge.src_conn,
                                                            new_data = dcpy(edge.data))

                    # case 2: pure out node: connect to outer array node
                    if dst in (out_nodes - intermediate_nodes):
                        if edge.dst != global_map_exit:
                            next_conn = global_map_exit.next_connector()
                            in_conn= 'IN_' + next_conn
                            out_conn = 'OUT_' + next_conn
                            global_map_exit.add_in_connector(in_conn)
                            global_map_exit.add_out_connector(out_conn)
                            self.redirect_edge(graph, edge, new_dst = global_map_exit,
                                                                   new_dst_conn = in_conn)
                            port_created = (in_conn, out_conn)
                            #edge.dst = global_map_exit
                            #edge.dst_conn = in_conn

                        else:
                            conn_nr = edge.dst_conn[3:]
                            in_conn = port_created.st
                            out_conn = port_created.nd

                        # map
                        graph.add_edge(global_map_exit,
                                       out_conn,
                                       dst,
                                       None,
                                       dcpy(out_edge.data))
                        graph.remove_edge(out_edge)

                # remove the edge if it has not been used by any pure out node
                if not port_created:
                    graph.remove_edge(edge)

            # maps are now ready to be discarded
            graph.remove_node(map_entry)
            graph.remove_node(map_exit)


        # do one pass to adjust in-transients and their corresponding memlets
        for node in intermediate_nodes:
            # all incoming edges to node
            in_edges = graph.in_edges(node)
            # outgoing edges going to another fused part
            inter_edges = []
            # outgoing edges that exit global map
            out_edges = []
            for e in graph.out_edges(node):
                if e.dst == global_map_exit:
                    out_edges.append(e)
                else:
                    inter_edges.append(e)

            # general case: multiple in_edges per in-between transient
            # e.g different subsets written to it
            in_edges_iter = iter(in_edges)
            in_edge = next(in_edges_iter)
            target_subset = dcpy(in_edge.data.subset)
            base_offset = in_edge.data.subset.min_element()
            ######
            while True:
                try: # executed if there are multiple in_edges
                    in_edge = next(in_edges_iter)
                    target_subset_curr = dcpy(in_edge.data.subset)
                    base_offset_curr = in_edge.data.subset.min_element()

                    target_subset = subsets.bounding_box_union(target_subset, \
                                                               target_subset_curr)
                    base_offset = [min(base_offset[i], base_offset_curr[i]) for i in range(len(base_offset))]
                except StopIteration:
                    break
            ######

            # Transient augmentation
            # check whether array data has to be modified, if not, keep OG
            if subgraph_contains_data[node.data]:
                sizes = target_subset.bounding_box_size()
                new_data_shape = [sz for (sz, s) in zip(sizes, target_subset)]
                new_data_strides = [data._prod(new_data_shape[i+1:])
                                    for i in range(len(new_data_shape))]

                new_data_totalsize = data._prod(new_data_shape)
                new_data_offset = [0]*len(new_data_shape)

                transient_to_transform = sdfg.data(node.data)
                transient_to_transform.shape   = new_data_shape
                transient_to_transform.strides = new_data_strides
                transient_to_transform.total_size = new_data_totalsize
                transient_to_transform.offset  = new_data_offset

                if schedule == dtypes.ScheduleType.GPU_Device:
                    if self.cuda_transient_allocation == 'local':
                        transient_to_transform.storage = dtypes.StorageType.Register
                    if self.cuda_transient_allocation == 'shared':
                        transient_to_transform.storage = dtypes.StorageType.GPU_Shared
                    if self.cuda_transient_allocation == 'default':
                        transient_to_transform.storage = dtypes.StorageType.Default
                    if self.cuda_transient_allocation == 'auto':
                        transient_to_transform.storage = self.storage_type_inference(node)

            else:
                # don't modify data container - array is needed outside
                # of subgraph.

                # hack: set lifetime to State | TODO: verify
                sdfg.data(node.data).lifetime = dtypes.AllocationLifetime.State

            # offset every memlet where necessary
            for iedge in in_edges:
                for edge in graph.memlet_tree(iedge):
                    if edge.data.data == node.data:
                        edge.data.subset.offset(base_offset, True)
                    elif edge.data.other_subset:
                        edge.data.other_subset.offset(base_offset, True)

            for cedge in cont_edges:
                for edge in graph.memlet_tree(cedge):
                    if edge.data.data == node.data:
                        edge.data.subset.offset(base_offset, True)
                    elif edge.data.other_subset:
                        edge.data.other_subset.offset(base_offset, True)


            # if in_edges has several entries:
            # put other_subset into out_edges for correctness
            if len(in_edges) > 1:
                for oedge in out_edges:
                    oedge.data.other_subset = dcpy(oedge.data.subset)
                    oedge.data.other_subset.offset(base_offset, True)



        # do one last pass to correct outside memlets adjacent to global map
        for out_connector in global_map_entry.out_connectors:
            # find corresponding in_connector
            in_connector = 'IN' + out_connector[3:]
            for iedge in graph.in_edges(global_map_entry):
                if iedge.dst_conn == in_connector:
                    in_edge = iedge
            for oedge in graph.out_edges(global_map_entry):
                if oedge.src_conn == out_connector:
                    out_edge = oedge
            # do memlet propagation
            memlet_out = propagate_memlet(dfg_state = graph,
                                          memlet = out_edge.data,
                                          scope_node = global_map_entry,
                                          union_inner_edges = True)

            # override number of accesses
            in_edge.data.volume = memlet_out.volume

        # create a hook for outside access to global_map
        self._global_map_entry = global_map_entry
