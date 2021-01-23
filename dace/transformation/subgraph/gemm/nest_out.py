# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

from copy import deepcopy as dcpy
from dace import dtypes, registry, symbolic, subsets
from dace.sdfg import nodes
from dace.memlet import Memlet
from dace.sdfg import replace
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from typing import List, Union
import networkx as nx
from dace.sdfg import state as dace_state
from dace.sdfg import sdfg as dace_sdfg
from dace import memlet
from dace.sdfg import graph as dace_graph
from dace.sdfg.nodes import Map
from dace.transformation.helpers import nest_state_subgraph
import itertools
from dace import data


def rename_map_parameter(state: dace_state.SDFGState, map_entry: nodes.MapEntry, old_name: str, new_name: str):
    for i, p in enumerate(map_entry.map.params):
        if p == old_name:
            map_entry.map.params[i] = new_name
            subgraph = dace_graph.SubgraphView(state, state.all_nodes_between(map_entry, state.exit_node(map_entry)))
            replace(subgraph, old_name, new_name)


def add_connector_prefixes(state: dace_state.SDFGState, nsdfg: nodes.NestedSDFG, prefix: str):
    old_in_connectors = nsdfg.in_connectors
    nsdfg.in_connectors = {}
    for ic in old_in_connectors:
        new_name = prefix + ic
        nsdfg.in_connectors[new_name] = old_in_connectors[ic]
        nsdfg.sdfg.replace(ic, new_name)
    for e in state.in_edges(nsdfg):
        e.dst_conn = prefix + e.dst_conn
    old_out_connectors = nsdfg.out_connectors
    nsdfg.out_connectors = {}
    for ic in old_out_connectors:
        new_name = prefix + ic
        nsdfg.out_connectors[new_name] = old_out_connectors[ic]
        nsdfg.sdfg.replace(ic, new_name)
    for e in state.out_edges(nsdfg):
        e.src_conn = prefix + e.src_conn


def add_transient_prefixes(sdfg: dace_sdfg.SDFG, prefix: str):
    transients = []
    for name, array in sdfg.arrays.items():
        if array.transient:
            transients.append(name)
    for name in transients:
        sdfg.replace(name, prefix + name)


def add_state_prefixes(sdfg: dace_sdfg.SDFG, prefix: str):
    for node in sdfg.nodes():
        node._label = prefix + node._label


@registry.autoregister_params()
class NestedMapFusion(transformation.Transformation):

    first_state = transformation.PatternNode(dace_sdfg.SDFGState)
    second_state = transformation.PatternNode(dace_sdfg.SDFGState)

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                NestedMapFusion.first_state,
                NestedMapFusion.second_state
            )
        ]

    @staticmethod
    def can_be_applied(sdfg: dace_sdfg.SDFG, candidate, expr_index, _sdfg, strict=False):

        first_state: dace_state.SDFGState = sdfg.nodes()[candidate[NestedMapFusion.first_state]]
        second_state: dace_state.SDFGState = sdfg.nodes()[candidate[NestedMapFusion.second_state]]

        sources1 = first_state.source_nodes()
        sinks1 = first_state.sink_nodes()

        sources2 = second_state.source_nodes()
        sinks2 = second_state.sink_nodes()

        map1 = first_state.out_edges(sources1[0])
        map2 = second_state.out_edges(sources2[0])
        if not isinstance(map1, nodes.MapEntry) or not isinstance(map2, nodes.MapEntry):
            return False 

        for e in first_state.out_edges(s1) for s1 in sources1:
            if e.dst != map1:
                return False 
        
        for e in second_state.out_edges(s2) for s2 in sources2:
            if e.dst != map2:
                return False 

        for e in first_state.in_edges(s1) for s1 in sinks1:
            if e.src != first_state.exit_node(map1):
                return False 
        
        for e in second_state.out_edges(s2) for s2 in sinks2:
            if e.src != second_state.exit_node(map2):
                return False 
        
        return True
        # check whether there are maps that are 


    @staticmethod
    def match_to_str(sdfg, candidate):
        first_state = sdfg.nodes()[candidate[NestedMapFusion.first_state]]
        second_state = sdfg.nodes()[candidate[NestedMapFusion.second_state]]

        return " -> ".join(state.label for state in [first_state, second_state])

    def apply(self, sdfg: dace_sdfg.SDFG):
        first_state: dace_state.SDFGState = sdfg.nodes()[self.subgraph[NestedMapFusion.first_state]]
        second_state: dace_state.SDFGState = sdfg.nodes()[self.subgraph[NestedMapFusion.second_state]]

        sources1 = first_state.source_nodes()
        sinks1 = first_state.sink_nodes()

        sources2 = second_state.source_nodes()
        sinks2 = second_state.sink_nodes()

        map_entry1: nodes.MapEntry = first_state.out_edges(sources1[0])[0].dst
        map_exit1: nodes.MapExit = first_state.in_edges(sinks1[0])[0].src

        map_entry2: nodes.MapEntry = second_state.out_edges(sources2[0])[0].dst
        map_exit2: nodes.MapExit = second_state.in_edges(sinks2[0])[0].src

        common_params = dict()
        for i, ((p1, r1), (p2, r2)) in enumerate(zip(zip(map_entry1.map.params,map_entry1.map.range), zip(map_entry2.map.params, map_entry2.map.range))):
            if r1 == r2:
                new_name = f'p{i}'
                common_params[f'p{i}'] = r1 
                rename_map_parameter(state1, map_entry1, p1, new_name)
                rename_map_parameter(state2, map_entry2, p2, new_name)

            else:
                break 
        
        sdfg.apply_transformations_repeated(MapExpansion)

        for _ in range(len(common_params) - 1):
            t = MapCollapse(sdfg_id = sdfg.sdfg_id, sdfg.nodes().index(first_state),
                            {MapCollapse._outer_map_entry: first_state.node_id(map_entry1),
                             MapCollapse._inner_map_entry: first_state.node_id(first_state.out_edges(map_entry1)[0].dst)}
            t.apply()

            t = MapCollapse(sdfg_id = sdfg.sdfg_id, sdfg.nodes().index(second_state),
                            {MapCollapse._outer_map_entry: second_state.node_id(map_entry2),
                             MapCollapse._inner_map_entry: second_state.node_id(second.out_edges(map_entry2)[0].dst)}
            t.apply()
        
        # next delete maps 
        # reconnect first and then delete  
        
        graph.remove_node(map_entry1)
        graph.remove_node(map_entry2)

            




        inputs1: List[nodes.AccessNode] = [e.dst for e in first_state.out_edges(map_entry1)]
        outputs1: List[nodes.AccessNode] = [e.src for e in first_state.in_edges(map_exit1)]

        inputs2: List[nodes.AccessNode] = [e.dst for e in second_state.out_edges(map_entry2)]
        outputs2: List[nodes.AccessNode] = [e.src for e in second_state.in_edges(map_exit2)]

        map1: Map = map_entry1.map
        map2: Map = map_entry2.map

        # get nested sdfgs
        nsdfg1: nodes.NestedSDFG = first_state.out_edges(inputs1[0])[0].dst
        nsdfg2: nodes.NestedSDFG = second_state.out_edges(inputs2[0])[0].dst

        # rename parameter name in the second state to match the first state
        for p1, p2 in zip(map1.params, map2.params):
            rename_map_parameter(second_state, map_entry2, p2, p1)

        # rename all connectors in nested sdfgs to avoid name collisions
        add_connector_prefixes(first_state, nsdfg1, 'n1_')
        add_connector_prefixes(second_state, nsdfg2, 'n2_')

        add_transient_prefixes(nsdfg1.sdfg, 'n1_')
        add_transient_prefixes(nsdfg2.sdfg, 'n2_')

        add_state_prefixes(nsdfg1.sdfg, 'n1_')
        add_state_prefixes(nsdfg2.sdfg, 'n2_')

        # rename connectors to make names the same in both nested sdfgs
        new_state = sdfg.add_state('state')
        for e in sdfg.in_edges(first_state):
            sdfg.add_edge(e.src, new_state, dace_sdfg.InterstateEdge(e.data.condition, e.data.assignments))
        for e in sdfg.out_edges(second_state):
            sdfg.add_edge(new_state, e.dst, dace_sdfg.InterstateEdge(e.data.condition, e.data.assignments))

        new_map = nodes.Map("map", map1.params, map1.range)
        new_map_entry = nodes.MapEntry(new_map)
        new_map_exit = nodes.MapExit(new_map)
        new_state.add_nodes_from([new_map_entry, new_map_exit])

        new_sdfg: dace_sdfg.SDFG = dace_sdfg.SDFG(
            name='sdfg',
            arg_types={**nsdfg1.sdfg.arg_types, **nsdfg2.sdfg.arg_types},
            constants={**nsdfg1.sdfg.constants, **nsdfg2.sdfg.constants},
        )
        new_nsdfg: nodes.NestedSDFG = new_state.add_nested_sdfg(
            sdfg=new_sdfg,
            parent='unused_field',
            inputs={**nsdfg1.in_connectors, **nsdfg2.in_connectors},
            outputs={**nsdfg1.out_connectors, **nsdfg2.out_connectors},
            symbol_mapping={**nsdfg1.symbol_mapping, **nsdfg2.symbol_mapping}
        )

        for i in inputs1 + inputs2:
            state = first_state if i in inputs1 else second_state
            new_access = new_state.add_access(i.data)
            oe = state.out_edges(i)
            assert len(oe) == 1
            e: dace_graph.MultiConnectorEdge = oe[0]
            new_state.add_nedge(new_map_entry, new_access, memlet.Memlet())
            new_state.add_edge(new_access, None, new_nsdfg, e.dst_conn, e.data)

        for o in outputs1 + outputs2:
            state = first_state if o in outputs1 else second_state
            new_access = new_state.add_access(o.data)
            ie = state.in_edges(o)
            assert len(ie) == 1
            e: dace_graph.MultiConnectorEdge = ie[0]
            new_state.add_edge(new_nsdfg, e.src_conn, new_access, None, e.data)
            new_state.add_nedge(new_access, new_map_exit, memlet.Memlet())

        for node in itertools.chain(nsdfg1.sdfg.nodes(), nsdfg2.sdfg.nodes()):
            node.parent = new_sdfg

            # ugly fix of nested sdfg field that points to parent
            for n in node.nodes():
                if isinstance(n, nodes.NestedSDFG):
                    s: dace_sdfg.SDFG = n.sdfg
                    s.parent_sdfg = new_sdfg

            new_sdfg.add_node(node)

        for edge in itertools.chain(nsdfg1.sdfg.edges(), nsdfg2.sdfg.edges()):
            new_sdfg.add_edge(edge.src, edge.dst, edge.data)

        # copy arrays
        for name, array in itertools.chain(nsdfg1.sdfg.arrays.items(), nsdfg2.sdfg.arrays.items()):
            new_sdfg.add_datadesc(name, array)

        # create barrier state
        barrier_state = new_sdfg.add_state('barrier_state')

        # detect sdfg1 exit states (states without outgoing edges)
        sdfg1_exit_states = [
            n for n in nsdfg1.sdfg.nodes()
            if not nsdfg1.sdfg.out_edges(n)
        ]

        # create edges from sdfg1 exit states to barrier state
        for exit1_node in sdfg1_exit_states:
            new_sdfg.add_edge(exit1_node, barrier_state, dace_sdfg.InterstateEdge())

        # create edge from barrier state to the start of sdfg2
        new_sdfg.add_edge(barrier_state, nsdfg2.sdfg.start_state, dace_sdfg.InterstateEdge())

        # remove old states
        sdfg.remove_node(first_state)
        sdfg.remove_node(second_state)