""" This file implements the ExecutionScore class """

from dace.transformation.subgraph import SubgraphFusion, StencilTiling, helpers
from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.dataflow import DeduplicateAccess
from dace.properties import make_properties, Property, TypeProperty
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView
from dace.sdfg.nodes import Node, Tasklet
from dace.config import Config
from dace.memlet import Memlet

import dace.sdfg.propagation as propagation
import dace.sdfg.nodes as nodes
import dace.symbolic as symbolic
import dace.dtypes as dtypes
import dace.subsets as subsets
import numpy as np
import sympy
import ast
import math
import networkx as nx

from typing import Set, Union, List, Callable, Dict, Type

from dace.transformation.estimator.scoring import ScoringFunction, MemletScore

import json
import warnings
import os
import functools
import itertools
import sys


@make_properties
class RegisterScore(MemletScore):
    '''
    Evaluates Subgraphs by just running their
    SDFG and returning the runtime
    '''

    register_per_block = Property(desc="No of Registers per Block available",
                                  dtype=int,
                                  #default=65536,
                                  default=32768)

    datatype = TypeProperty(desc="Datatype of Input",
                            default = np.float32)


    def __init__(self,
                 sdfg: SDFG,
                 graph: SDFGState,
                 io: Dict,
                 subgraph: SubgraphView = None,
                 gpu: bool = None,
                 transformation_function: Type = CompositeFusion,
                 **kwargs):

        super().__init__(sdfg=sdfg,
                         graph=graph,
                         subgraph=subgraph,
                         io=io,
                         gpu=gpu,
                         transformation_function=transformation_function,
                         **kwargs)

        # for debugging purposes
        self._i = 0

    def propagate_outward(self, graph: SDFGState, memlets: List[Memlet],
                          context: List[Union[nodes.MapEntry,
                                              nodes.NestedSDFG]]):
        '''
        Propagates Memlets outwards given map entry nodes in context vector. 
        Assumes that first entry is innermost entry closest to memlet 
        '''
        for ctx in context:
            if isinstance(ctx, nodes.MapEntry):
                if ctx.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
                    # we are at thread block level - do not propagate
                    break
                for i, memlet in enumerate(memlets):
                    memlets[i] = propagation.propagate_memlet(
                        graph, memlet, ctx, False)
            else:
                raise NotImplementedError("TODO")

        return memlets

    def tasklet_symbols(self, astobj, max_size = 12):
        '''
        traverses a tasklet ast and primitively assigns 
        registers to symbols and constants 
        '''

        assignments = {}
        base_multiplier = dtypes._BYTES[self.datatype] / 4
        for n in ast.walk(astobj):
            if isinstance(n, ast.Name):
                assignments[n.id] = 1 * base_multiplier
            if isinstance(n, ast.Constant):
                assignments[n] = 1 * base_multiplier

        return assignments

                
    def evaluate_register_traffic(self,
                                  sdfg: SDFG,
                                  graph: SDFGState,
                                  scope_node: nodes.MapEntry,
                                  scope_dict: dict = None):
        ''' 
        Evaluates traffic in a scope subgraph with given scope 
        node that flows out of and into registers 
        '''

        if not scope_dict:
            scope_dict = graph.scope_dict()

        register_traffic = 0

        for node in (graph.scope_subgraph(scope_node) if scope_node else graph.nodes()):
            if isinstance(node, nodes.AccessNode) and sdfg.data(node.data).storage == dtypes.StorageType.Register:
                # see whether scope is contained in outer_entry
                scope = scope_dict[node]
                current_context = [scope_node]
                while scope and scope != scope_node:
                    current_context.append(scope)
                    scope = scope_dict[scope]
                assert scope == scope_node 
                # add to traffic
                memlets = list(e.data for e in 
                    itertools.chain(graph.out_edges(node),
                                    graph.in_edges(node)))
                self.propagate_outward(graph, memlets, current_context)
                for memlet in memlets:
                    register_traffic += memlet.volume
            
            if isinstance(node, nodes.NestedSDFG):
                for state in node.sdfg.nodes():
                    register_traffic += self.evaluate_register_traffic(node.sdfg, state, None)

        return self.symbolic_evaluation(register_traffic)

    def evaluate_tasklet_output(self,
                                sdfg: SDFG,
                                graph: SDFGState,
                                node: Union[nodes.Tasklet, nodes.NestedSDFG],
                                context: List[nodes.MapEntry]):
        
        '''
        NOTE: We don't need to be concerned about write to the same 
        array by multiple tasklets in one active set 
        -> this would invalidate the sdfg 
        '''

        result = 0
        for e in graph.out_edges(node):
            # see whether path comes from global map entry 
            for path_e in graph.memlet_path(e):
                if isinstance(path_e.dst, nodes.AccessNode) and sdfg.data(path_e.dst.data).storage == dtypes.StorageType.Register:
                    result += path_e.data.subset.num_elements()
        
        return self.symbolic_evaluation(result)
       
    def evaluate_tasklet_input(self,
                               sdfg: SDFG,
                               graph: SDFGState,
                               node: Union[nodes.Tasklet, nodes.NestedSDFG],
                               context: List[nodes.MapEntry]):
        
        
        result = 0
        if isinstance(node, nodes.Tasklet):
            for e in graph.in_edges(node):
                # see whether path comes from register node 
                # if so it already got covered in output 
                for e in graph.memlet_path(e):
                    if isinstance(e.src, nodes.AccessNode) and sdfg.data(e.src.data).storage == dtypes.StorageType.Register:
                        break 
                else:
                    result += e.data.subset.num_elements()
           
        return self.symbolic_evaluation(result)


    def evaluate_tasklet_inner(self,
                               sdfg: SDFG,
                               graph: SDFGState,
                               node: Union[nodes.Tasklet, nodes.NestedSDFG],
                               context: List[nodes.MapEntry],
                               max_size = 10):

        '''
        Evaluates a tasklet or nested sdfg node for a register size usage estimate 

        :param node:    tasklet or nsdfg node 
        :param context: list of map entries to propagate thru ending with toplevel 
        :param nested:  specify whether we are in a nested sdfg
        '''

        estimate_internal = 0
        if isinstance(node, nodes.Tasklet):
            if node.code.language == dtypes.Language.Python:
                names = set(n.id
                            for n in ast.walk(ast.parse(node.code.as_string))
                            if isinstance(n, ast.Name))
                names = self.tasklet_symbols(ast.parse(node.code.as_string))
                connector_symbols = set(itertools.chain(node.in_connectors.keys(), node.out_connectors.keys()))
                estimate_internal = sum([v for (k,v) in names.items() if k not in connector_symbols])
            
            else:
                warnings.warn(
                    'WARNING: Register Score cannot evaluate non Python block')
                # just use some shady estimate
                estimate_internal = min(
                    8, math.ceil(node.code.as_string.count('\n') / 2))
        else:
            # nested sdfg
            # get an estimate for each nested sdfg state and max over these values
            estimate_internal = max([
                self.evaluate_state(node.sdfg, s, None)
                for s in node.sdfg.nodes()
            ])
            if estimate_internal == 0:
                warnings.warn('Detected a Nested SDFG where Tasklets were found inside for analysis')


        estimate_internal = self.symbolic_evaluation(estimate_internal)

        if self.debug:
            print(f"Node estimate {node} =", estimate_internal)
        
        return min(estimate_internal, max_size)


        
    def evaluate_state(self,
                       sdfg: SDFG,
                       graph: SDFGState,
                       scope_node: Node,
                       nested: bool = False,
                       scope_dict: dict = None,
                       scope_children: dict = None):
        '''
        Evaluates Register spill for a whole state where scope_node indicates the outermost scope in which spills should be analyzed (within its inner scopes as well)
        '''
        print(f"Evaluating state {graph} with scope node {scope_node}")
        # get some variables if they haven not already been inputted
        if scope_dict is None:
            scope_dict = graph.scope_dict()
        if scope_children is None:
            scope_children = graph.scope_children()


        subgraph = graph.scope_subgraph(scope_node) if scope_node is not None else SubgraphView(graph, graph.nodes())

        # loop over all tasklets
        context = dict()
        for node in subgraph.nodes():
            if isinstance(node, (nodes.Tasklet, nodes.NestedSDFG)):
                # see whether scope is contained in outer_entry
                scope = scope_dict[node]
                current_context = []
                while scope and scope != scope_node:
                    current_context.append(scope)
                    scope = scope_dict[scope]
                assert scope == scope_node
                context[node] = current_context

        # similarly as in transient_reuse, create a proxy graph only
        # containing tasklets that are inside our fused map
        proxy = nx.MultiDiGraph()
        for n in subgraph.nodes():
            proxy.add_node(n)
        for n in subgraph.nodes():
            for e in graph.all_edges(n):
                if e.src in subgraph and e.dst in subgraph:
                    proxy.add_edge(e.src, e.dst)
    
        # remove all nodes that are not Tasklets
        for n in subgraph.nodes():
            if n not in context:
                for p in proxy.predecessors(n):
                    for c in proxy.successors(n):
                        proxy.add_edge(p, c)
                proxy.remove_node(n)

        # set up predecessor and successor array
        predecessors, successors = {}, {}
        for n in proxy.nodes():
            successors[n] = set(proxy.successors(n))
            predecessors[n] = set(proxy.predecessors(n))
       

        active_sets = list()
        for cc in nx.weakly_connected_components(proxy):
            queue = set([node for node in cc if len(predecessors[node]) == 0])
            while len(queue) > 0:
                active_sets.append(set(queue))
                for tasklet in queue:
                    # remove predecessor link (for newly added elements to queue)
                    for tasklet_successor in successors[tasklet]:
                        if tasklet in predecessors[tasklet_successor]:
                            predecessors[tasklet_successor].remove(tasklet)
                       
                # if all predecessor in degrees in successors are 0
                # then tasklet is ready to be removed from the queue
                next_tasklets = set()
                tasklets_to_remove = set()
                for tasklet in queue:
                    remove_tasklet = True
                    for tasklet_successor in successors[tasklet]:
                        print(tasklet, "->", tasklet_successor)
                        print(predecessors[tasklet_successor])
                        if len(predecessors[tasklet_successor]) > 0:
                            remove_tasklet = False 
                        elif tasklet_successor not in queue:
                            next_tasklets.add(tasklet_successor)
                    if remove_tasklet:
                        tasklets_to_remove.add(tasklet)

                if len(next_tasklets) == 0 and len(tasklets_to_remove) == 0:
                    sdfg.save('error_here.sdfg')
                    print("ERROR")
                    sys.exit(0)
                queue |= next_tasklets
                queue -= tasklets_to_remove

                assert len(next_tasklets & tasklets_to_remove) == 0

        # add up scores in respective active sets and return the max

        last_set = set()
        active_set_scores = list()

        input_scores, inner_scores, output_scores = {},{},{}
        for (node, ctx) in context.items():
            inner_scores[node] = self.evaluate_tasklet_inner(sdfg, graph, node, ctx)
            input_scores[node] = self.evaluate_tasklet_input(sdfg, graph, node, ctx)
            output_scores[node] = self.evaluate_tasklet_output(sdfg, graph, node, ctx)


        print(f"--- State Subgraph Analysis: {graph} ---")
        for tasklet_set in active_sets:
            # evaluate score for current tasklet set 
            set_score = 0
            for tasklet in tasklet_set:
                max_active = max(input_scores[n] + inner_scores[n] for n in tasklet_set)
                all_completed = sum(output_scores[n] for n in tasklet_set)
                set_score += max_active + all_completed


            print("Active Set:", tasklet_set)
            print("Score:", set_score)
            active_set_scores.append(set_score)
            last_set = tasklet_set 
        
        print("-----------------------------")
        return max(active_set_scores) if len(active_set_scores) > 0 else 0



    def evaluate_available(self, outer_map):
        ''' FORNOW: get register available count '''

        threads = Config.get('compiler', 'cuda', 'default_block_size')
        threads = functools.reduce(lambda a, b: a * b,
                                   [int(e) for e in threads.split(',')])
        reg_count_available = self.register_per_block / (min(
            1028, 16 * threads))

        return reg_count_available


    def estimate_spill(self, sdfg, graph, scope_node):
        ''' 
        estimates spill for a fused subgraph that contains one outer map entry 
        '''
        # get some variables
        scope_dict = graph.scope_dict()
        outer_map_entry = scope_node 
        outer_map = scope_node.map

        # get register used count
        print("EVALUATE STATE")
        reg_count_required = self.evaluate_state(sdfg=sdfg,
                                                 graph=graph,
                                                 scope_node=outer_map_entry,
                                                 scope_dict=scope_dict)
        # get register provided count (FORNOW)
        reg_count_available = self.evaluate_available(outer_map = outer_map)
        # get register traffic count
        if reg_count_available < reg_count_required:
            register_read_write = self.evaluate_register_traffic(
                sdfg = sdfg,
                graph = graph,
                scope_node = outer_map_entry,
                scope_dict = scope_dict)
        else:
            register_read_write = 0

        spill_fraction = max(0, (reg_count_required - reg_count_available) /
                             reg_count_available)
        print("REGISTERS", reg_count_required)
        print("REGISTER TRAFFIC", register_read_write)
        print("SPILL FRACTION", spill_fraction)

        # next up, add spill fraction to movement
        return spill_fraction * register_read_write

    def score(self, subgraph: SubgraphView):
        '''
        Applies CompositeFusion to the Graph and compares Memlet Volumes
        with the untransformed SDFG passed to __init__().
        '''  

        sdfg_copy = SDFG.from_json(self._sdfg.to_json())
        graph_copy = sdfg_copy.nodes()[self._state_id]
        subgraph_copy = SubgraphView(graph_copy, [
            graph_copy.nodes()[self._graph.nodes().index(n)] for n in subgraph
        ])

        if self.debug:
            print("ScoreMemlet::Debug::Subgraph to Score:",
                  subgraph_copy.nodes())

        transformation_function = self._transformation(subgraph_copy)
        # assign properties to transformation
        for (arg, val) in self._kwargs.items():
            try:
                setattr(transformation_function, arg, val)
            except AttributeError:
                warnings.warn(f"Attribute {arg} not found in transformation")
            except (TypeError, ValueError):
                warnings.warn(f"Attribute {arg} has invalid value {val}")

        transformation_function.apply(sdfg_copy)
        if self.deduplicate:
            sdfg_copy.apply_transformations_repeated(DeduplicateAccess,
                                                     states=[graph_copy])
        if self.propagate_all or self.deduplicate:
            propagation.propagate_memlets_scope(sdfg_copy, graph_copy,
                                                graph_copy.scope_leaves())

        self._i += 1

        current_traffic = self.estimate_traffic(sdfg_copy, graph_copy)


        # NOTE: assume transformation function has 
        # .global_map_entry field, this makes it much easier 

        outer_entry = transformation_function._global_map_entry
        spill_traffic = self.estimate_spill(sdfg_copy, graph_copy,
                                            outer_entry)

        return (current_traffic + spill_traffic) / self._base_traffic

    @staticmethod
    def name():
        return "Estimated Memlet Traffic plus Spill"
