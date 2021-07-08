# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import ast
from dace.properties import make_properties
import astunparse
import dace
from dace.sdfg.graph import SubgraphView
from dace.sdfg.nodes import CodeNode, LibraryNode, AccessNode
from dace.sdfg.propagation import propagate_memlet
from dace.properties import Property, make_properties
from dace.libraries.standard.nodes.reduce import Reduce
from dace.sdfg.scope import ScopeTree as Scope
from dace.symbolic import pystr_to_symbolic
import itertools
import sympy
import sys
from typing import Any, Dict, List, Union

@make_properties
class MovementCounter: 
    _iprint = lambda *args: print(*args)

    # Do not use O(x) or Order(x) in sympy, it's not working as intended
    _bigo = sympy.Function('bigo')

    debug = Property(desc = "Output debug information",
                     dtype = bool, 
                     default = False)

    analyze_inner_traffic = Property(desc = "Analyze traffic inside maps. If it is"
                                            "associated to nodes that reside on global memory, "
                                            "propagate traffic volume and also accumulate"
                                            "it to the final result",
                                     dtype = bool,
                                     default = False)
    

    def count_moved_data(self, sdfg: dace.SDFG, symbols: Dict[str, Any] = None) -> int:
        result = 0
        symbols = symbols or {}
        for state in sdfg.nodes():
            result += self.count_moved_data_state(state, symbols) * state.executions
        return result

    def count_moved_data_state(self, state: dace.SDFGState, symbols: Dict[str,
                                                                    Any] = None) -> int:
        if self.analyze_inner_traffic == True: 
            return self._count_moved_data_state_composite(state, symbols) 
        else:
            return self._count_moved_data_state(state, symbols) 
    

    def _count_moved_data_state(self, state: dace.SDFGState, symbols: Dict[str,
                                                                    Any] = None) -> int:
        sdict = state.scope_children()
        result = 0
        symbols = symbols or {}


        edges_counted = set()

        for node in sdict[None]:
            node_result = 0
            if isinstance(node, (CodeNode, LibraryNode, Reduce)):
                inputs = sum(e.data.volume for e in state.in_edges(node)
                            if e not in edges_counted)
                outputs = sum(e.data.volume for e in state.out_edges(node)
                            if e not in edges_counted)
                # Do not count edges twice
                edges_counted |= set(state.all_edges(node))

                if self.debug:
                    self._iprint(
                        type(node).__name__, node, 'inputs:', inputs, 'outputs:',
                        outputs)
                node_result += inputs + outputs
            elif isinstance(node, dace.nodes.EntryNode):
                # Gather inputs from entry node
                inputs = sum(e.data.volume for e in state.in_edges(node)
                            if e not in edges_counted)
                # Do not count edges twice
                edges_counted |= set(state.in_edges(node))
                # Gather outputs from exit node
                exit_node = state.exit_node(node)
                outputs = sum(e.data.volume
                            for e in state.out_edges(exit_node)
                            if e not in edges_counted)
                edges_counted |= set(state.out_edges(exit_node))
                if self.debug:
                    self._iprint('Scope',
                        type(node).__name__, node, 'inputs:', inputs, 'outputs:',
                        outputs)
                node_result += inputs + outputs
            result += node_result

        return result

    def count_moved_data_subgraph(self, state: dace.SDFGState, subgraph: SubgraphView, symbols: Dict[str,Any] = None) -> int:

        stree_root = state.scope_tree()[None]
        sdict = state.scope_children()
        result = 0
        symbols = symbols or {}

        edges_counted = set()

        for node in sdict[None]:
            if node in subgraph:
                node_result = 0
                if isinstance(node, (CodeNode, LibraryNode, Reduce)):
                    inputs = sum(e.data.volume for e in state.in_edges(node)
                                if e not in edges_counted and e.src in subgraph)
                    outputs = sum(e.data.volume for e in state.out_edges(node)
                                if e not in edges_counted and e.dst in subgraph)
                    # Do not count edges twice
                    edges_counted |= set(state.all_edges(node))

                    if self.debug:
                        self._iprint(
                            type(node).__name__, node, 'inputs:', inputs, 'outputs:',
                            outputs)
                    node_result += inputs + outputs
                elif isinstance(node, dace.nodes.EntryNode):
                    # Gather inputs from entry node
                    inputs = sum(e.data.volume for e in state.in_edges(node)
                                if e not in edges_counted and e.src in subgraph)
                    # Do not count edges twice
                    edges_counted |= set(state.in_edges(node))
                    # Gather outputs from exit node
                    exit_node = state.exit_nodes(node)[0]
                    outputs = sum(e.data.volume
                                for e in state.out_edges(exit_node)
                                if e not in edges_counted and e.dst in subgraph)
                    edges_counted |= set(state.out_edges(exit_node))
                    if self.debug:
                        self._iprint('Scope',
                            type(node).__name__, node, 'inputs:', inputs, 'outputs:',
                            outputs)
                    node_result += inputs + outputs
                result += node_result
        return result


    def _count_moved_data_state_composite(self, state: dace.SDFGState, symbols: Dict[str,
                                                                    Any] = None) -> int:
        stree_root = state.scope_tree()[None]
        sdict = state.scope_children()
        result = 0
        symbols = symbols or {}
        # determine which scopes to search for
        # we restrict ourselves to scopes that are a direct
        # child from the outermost scope
        other_scopes_of_interest = set(sdict[None]) & set(sdict.keys())

        edges_counted = set()

        for node in sdict[None]:
            node_result = 0
            if isinstance(node, (CodeNode, LibraryNode, Reduce)):
                inputs = sum(e.data.volume for e in state.in_edges(node)
                            if e not in edges_counted)
                outputs = sum(e.data.volume for e in state.out_edges(node)
                            if e not in edges_counted)
                # Do not count edges twice
                edges_counted |= set(state.all_edges(node))

                if self.debug:
                    self._iprint(
                        type(node).__name__, node, 'inputs:', inputs, 'outputs:',
                        outputs)
                node_result += inputs + outputs
            elif isinstance(node, dace.nodes.EntryNode):
                # Gather inputs from entry node
                inputs = sum(e.data.volume for e in state.in_edges(node)
                            if e not in edges_counted)
                # Do not count edges twice
                edges_counted |= set(state.in_edges(node))
                # Gather outputs from exit node
                exit_node = state.exit_node(node)
                outputs = sum(e.data.volume
                            for e in state.out_edges(exit_node)
                            if e not in edges_counted)
                edges_counted |= set(state.out_edges(exit_node))
                if self.debug:
                    self._iprint('Scope',
                        type(node).__name__, node, 'inputs:', inputs, 'outputs:',
                        outputs)
                node_result += inputs + outputs
            result += node_result

        inner_storage_locations = [dace.dtypes.StorageType.Register,
                                dace.dtypes.StorageType.Default,
                                dace.dtypes.StorageType.GPU_Shared]
        for other_scope in other_scopes_of_interest:
            for node in sdict[other_scope]:
                if isinstance(node, AccessNode):
                    if state.parent.data(node.label).storage not in inner_storage_locations:
                        local_sum = 0
                        for e in itertools.chain(state.out_edges(node), state.in_edges(node)):
                            mm_propagated = propagate_memlet(state, e.data, other_scope, False)
                            local_sum += mm_propagated.volume
                        if self.debug:
                            self._iprint('Inner',
                                node.label, 'added:', mm_propagated.volume)
                        result += local_sum

        return result

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('USAGE: %s <SDFG FILE>' % sys.argv[0])
        exit(1)

    sdfg = dace.SDFG.from_file(sys.argv[1])
    print('Propagating memlets')
    dace.propagate_labels_sdfg(sdfg)
    print('Counting data movement')
    agent = MovementCounter()
    dm = agent.count_moved_data(sdfg)
    print('Total data movement', dm)
