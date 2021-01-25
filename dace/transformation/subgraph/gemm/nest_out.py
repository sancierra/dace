# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

from copy import deepcopy as dcpy
from dace import dtypes, registry, symbolic, subsets
from dace.sdfg import nodes
from dace.memlet import Memlet
from dace.sdfg import replace
from dace.sdfg import utils as sdutil
from dace.transformation import transformation
from typing import List, Union
from dace.sdfg import state as dace_state
from dace.sdfg import sdfg as dace_sdfg
from dace import memlet
from dace.sdfg import graph as dace_graph
from dace.sdfg.nodes import Map
from dace.transformation.helpers import nest_state_subgraph
import itertools
from dace import data

import dace
from dace.transformation.dataflow import MapExpansion, MapCollapse
from copy import deepcopy 

def rename_map_parameter(state: dace_state.SDFGState, map_entry: nodes.MapEntry, old_name: str, new_name: str):
    for i, p in enumerate(map_entry.map.params):
        if p == old_name:
            map_entry.map.params[i] = new_name
            subgraph = dace_graph.SubgraphView(state, state.nodes())
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
class NestOut(transformation.Transformation):

    first_state = transformation.PatternNode(dace_sdfg.SDFGState)
    second_state = transformation.PatternNode(dace.sdfg.SDFGState)
    #nsdfg = transformation.PatternNode(dace.sdfg.nodes.NestedSDFG)

    @staticmethod
    def annotates_memlets():
        return False

    @staticmethod
    def expressions():
        return [
            sdutil.node_path_graph(
                NestOut.first_state,
                NestOut.second_state
            )
        ]

    @staticmethod
    def can_be_applied(sdfg: dace_sdfg.SDFG, candidate, expr_index, _sdfg, strict=False):
        return True 
        '''
        first_state: dace_state.SDFGState = sdfg.nodes()[candidate[NestOut.first_state]]
        second_state: dace_state.SDFGState = sdfg.nodes()[candidate[NestOut.second_state]]

        sources1 = first_state.source_nodes()
        sinks1 = first_state.sink_nodes()

        sources2 = second_state.source_nodes()
        sinks2 = second_state.sink_nodes()

        map1 = sources1[0]
        map2 = second_state.out_edges(sources2[0])

        if not isinstance(map1, nodes.MapEntry) or not isinstance(map2, nodes.MapEntry):
            return False 

        
        return True
        # check whether there are maps that are 
        '''


    @staticmethod
    def match_to_str(sdfg, candidate):
        first_state = sdfg.nodes()[candidate[NestOut.first_state]]
        second_state = sdfg.nodes()[candidate[NestOut.second_state]]

        return " -> ".join(state.label for state in [first_state, second_state])

    def apply(self, sdfg: dace_sdfg.SDFG):
        first_state: dace_state.SDFGState = sdfg.nodes()[self.subgraph[NestOut.first_state]]
        second_state: dace_state.SDFGState = sdfg.nodes()[self.subgraph[NestOut.second_state]]

        sources1 = first_state.source_nodes()
        sinks1 = first_state.sink_nodes()

        sources2 = second_state.source_nodes()
        sinks2 = second_state.sink_nodes()

        map_entry1: nodes.MapEntry = sources1[0]
        map_exit1: nodes.MapExit = first_state.in_edges(sinks1[0])[0].src

        map_entry2: nodes.MapEntry = second_state.out_edges(sources2[0])[0].dst
        map_exit2: nodes.MapExit = second_state.in_edges(sinks2[0])[0].src

        print("map_entry1", map_entry1)
        print("map_entry2", map_entry2)
        common_params = dict()

        for i, ((p1, r1), (p2, r2)) in enumerate(zip(zip(map_entry1.map.params,map_entry1.map.range), zip(map_entry2.map.params, map_entry2.map.range))):
            if r1 == r2:
                new_name = f'p{i}'
                common_params[f'p{i}'] = r1 
                rename_map_parameter(first_state, map_entry1, str(p1), new_name)
                rename_map_parameter(second_state, map_entry2, str(p2), new_name)

            else:
                break 
        
        sdfg.save('after_replace.sdfg')
        sdfg.apply_transformations_repeated(MapExpansion)

        for _ in range(len(common_params) - 1):
            t = MapCollapse(sdfg.sdfg_id, sdfg.nodes().index(first_state),
                            {MapCollapse._outer_map_entry: first_state.node_id(map_entry1),
                             MapCollapse._inner_map_entry: first_state.node_id(first_state.out_edges(map_entry1)[0].dst)},
                             0)
            (new_entry, new_exit) = t.apply(sdfg)
            map_entry1 = new_entry 
            map_exit1 = new_exit

            t = MapCollapse(sdfg.sdfg_id, sdfg.nodes().index(second_state),
                            {MapCollapse._outer_map_entry: second_state.node_id(map_entry2),
                             MapCollapse._inner_map_entry: second_state.node_id(second_state.out_edges(map_entry2)[0].dst)},
                             0)
            (new_entry, new_exit) = t.apply(sdfg)
            map_entry2 = new_entry 
            map_exit2 = new_exit
        
        # next delete maps 
        # reconnect first and then delete  
        sdfg.save('inspect_me.sdfg')
        print("Part 1: DONE")
        print(map_entry1)
        print(map_entry2)
        print(map_exit1)
        print(map_exit2)
        print(type(map_entry1))
        print(type(map_entry2))
        print(type(map_exit1))
        print(type(map_exit2))
        

        e_dict = {} 
        for ie in first_state.in_edges(map_entry1):
            for oe in first_state.out_edges(map_entry1):
                if ie.dst_conn[2:] == oe.src_conn[3:]:
                    first_state.add_edge(ie.src, ie.src_conn, oe.dst, oe.dst_conn, oe.data)

        for ie in first_state.in_edges(map_exit1):
            for oe in first_state.out_edges(map_exit1):
                if ie.dst_conn[2:] == oe.src_conn[3:]:
                    first_state.add_edge(ie.src, ie.src_conn, oe.dst, oe.dst_conn, ie.data)

        for ie in second_state.in_edges(map_entry2):
            for oe in second_state.out_edges(map_entry2):
                if ie.dst_conn[2:] == oe.src_conn[3:]:
                    second_state.add_edge(ie.src, ie.src_conn, oe.dst, oe.dst_conn, oe.data)
                    e_dict[ie.data.data] = [deepcopy(ie.data), deepcopy(oe.data)]

        for ie in second_state.in_edges(map_exit2):
            for oe in second_state.out_edges(map_exit2):
                if ie.dst_conn[2:] == oe.src_conn[3:]:
                    second_state.add_edge(ie.src, ie.src_conn, oe.dst, oe.dst_conn, ie.data)
                    e_dict[ie.data.data] = [deepcopy(oe.data), deepcopy(ie.data)]


        first_state.remove_node(map_entry1)
        second_state.remove_node(map_entry2)
        first_state.remove_node(map_exit1)
        second_state.remove_node(map_exit2)

        sdfg.save('stage2.sdfg')

        # create deepcopy of sdfg and nest it in new state
        new_sdfg = dace.sdfg.SDFG.from_json(sdfg.to_json())
        new_state = dace.sdfg.SDFGState("outer", sdfg)

        sdfg.remove_node(first_state)
        sdfg.remove_node(second_state)
        new_state = sdfg.add_state("outer")
        nsdfg = new_state.add_nested_sdfg(new_sdfg, parent = sdfg,
                inputs={'_a','_b'},
                outputs={'_c'})
        

        
        params, ranges = [], []
        for (p, r) in common_params.items():
            params.append(p)
            ranges.append(r)
        
        new_map = nodes.Map("map", map_entry1.params, map_entry1.range)
        new_map_entry = nodes.MapEntry(new_map)
        new_map_exit = nodes.MapExit(new_map)
        new_state.add_nodes_from([new_map_entry, new_map_exit])
        a = new_state.add_access('_a')
        b = new_state.add_access('_b')
        c = new_state.add_access('_c')
        data_dict = {'_a':a, '_b':b, '_c':c}
        source_nodes = {'_a','_b'}
        print("****", e_dict)
        for data, access_node in data_dict.items():
            if data in source_nodes:
                new_map_entry.add_in_connector('IN_'+data)
                new_map_entry.add_out_connector('OUT_'+data)
                new_state.add_edge(access_node, None, new_map_entry, 'IN_'+data, e_dict[data][0])
                new_state.add_edge(new_map_entry, 'OUT_'+data, nsdfg, data, e_dict[data][1])
            else:
                new_map_exit.add_in_connector('IN_'+data)
                new_map_exit.add_out_connector('OUT_'+data)
                e = new_state.add_edge(nsdfg, data, new_map_exit, 'IN_'+data, e_dict[data][1])
                e.data.wcr = None 
                e.data.wcr_nonatomic = False 
                e = new_state.add_edge(new_map_exit, 'OUT_'+data, access_node, None, e_dict[data][0])
                e.data.wcr = None 
                e.data.wcr_nonatomic = False 

        print(new_map_entry.in_connectors)
        print(new_map_entry.out_connectors)
        print(new_map_exit.in_connectors)
        print(new_map_exit.out_connectors)


        nsdfg.sdfg.replace('p0','0')
        nsdfg.sdfg.replace('p1','0')
        nsdfg.sdfg.replace('M', '1')
        nsdfg.sdfg.replace('K', '1')

        nsdfg.sdfg.data('_a').strides = dace.symbolic.pystr_to_symbolic('N,1')
        nsdfg.sdfg.data('_b').strides = dace.symbolic.pystr_to_symbolic('K,1')
        nsdfg.sdfg.data('_c').strides = dace.symbolic.pystr_to_symbolic('K,1')
