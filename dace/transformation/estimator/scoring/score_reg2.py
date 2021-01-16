""" This file implements the RegisterScore class """

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

from typing import Set, Union, List, Callable, Dict, Type, Iterable

from dace.transformation.estimator.scoring import ScoringFunction, MemletScore

import json
import warnings
import os
import functools
import itertools
import collections
import sys


@make_properties
class RegisterScore(MemletScore):
    '''
    Evaluates Subgraphs by just running their
    SDFG and returning the runtime
    '''

    register_per_block = Property(desc="No of Registers per Block available",
                                  dtype=int,
                                  default=65536)
                                  #default=32768)
    
    max_registers_allocated = Property(desc="")

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
                                              nodes.NestedSDFG]], penetrate_everything = False
                            ):
        '''
        Propagates Memlets outwards given map entry nodes in context vector. 
        Assumes that first entry is innermost entry closest to memlet 
        '''
        for ctx in context:
            if isinstance(ctx, nodes.MapEntry):
                if not penetrate_everything and ctx.schedule == dtypes.ScheduleType.GPU_ThreadBlock:
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
                self.propagate_outward(graph, memlets, current_context, penetrate_everything= True)
                for memlet in memlets:
                    register_traffic += memlet.subset.num_elements()
            
            if isinstance(node, nodes.NestedSDFG):
                for state in node.sdfg.nodes():
                    register_traffic += self.evaluate_register_traffic(node.sdfg, state, None)

        return self.symbolic_evaluation(register_traffic)

    def evaluate_tasklet_output(self,
                                sdfg: SDFG,
                                graph: SDFGState,
                                node: Union[nodes.Tasklet, nodes.NestedSDFG],
                                known_registers: Iterable,
                                context: List[nodes.MapEntry],
                                ):
        

        result = 0
        if isinstance(node, nodes.Tasklet):
            for e in graph.out_edges(node):
                # see whether path comes from global map entry 
                '''
                for path_e in graph.memlet_path(e):
                    if isinstance(path_e.dst, nodes.AccessNode) and sdfg.data(path_e.dst.data).storage == dtypes.StorageType.Register:
                        result += path_e.data.subset.num_elements()
                '''

                for path_e in graph.memlet_path(e):
                    if isinstance(path_e.dst, nodes.AccessNode) and path_e.dst.data in known_registers:
                        break 
        
                else:
                    # FORNOW: Propagation is still not working correctly
                    # use this as a fix 
                    result_subset = self.symbolic_evaluation(e.data.subset.num_elements())
                    result_volume = self.symbolic_evaluation(e.data.volume)
                    result += min(result_subset, result_volume)
                    print(f"*******NE {node}", e.data.subset.num_elements())


            
        return self.symbolic_evaluation(result)
       
    def evaluate_tasklet_input(self,
                               sdfg: SDFG,
                               graph: SDFGState,
                               node: Union[nodes.Tasklet, nodes.NestedSDFG],
                               known_registers: Iterable,
                               context: List[nodes.MapEntry]):
        
        
        result = 0
        if isinstance(node, nodes.Tasklet):
            for e in graph.in_edges(node):
                # see whether path comes from register node 
                # if so it already got covered in output 
                '''
                for e in graph.memlet_path(e):
                    if isinstance(e.src, nodes.AccessNode) and sdfg.data(e.src.data).storage == dtypes.StorageType.Register:
                        break 
                else:
                    result += e.data.subset.num_elements()
                '''
                for e in graph.memlet_path(e):
                    if isinstance(e.src, nodes.AccessNode) and e.src.data in known_registers:
                        break
                else:
                    result += e.data.subset.num_elements()
           
        return self.symbolic_evaluation(result)


    def evaluate_tasklet_inner(self,
                               sdfg: SDFG,
                               graph: SDFGState,
                               node: Union[nodes.Tasklet, nodes.NestedSDFG],
                               context: List[nodes.MapEntry],
                               known_registers: Set,
                               known_register_size: int,
                               max_size = 8):

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

            # first add all connectors that belong to a register location to our register set 
            in_conn_registers = set(node.in_connectors.keys())
            out_conn_registers = set(node.out_connectors.keys())
            for e in graph.in_edges(node):
                for path_e in graph.memlet_path(e):
                    if isinstance(path_e.src, nodes.AccessNode) and path_e.src.data in known_registers:
                        if e.dst_conn in in_conn_registers:
                            in_conn_registers.remove(e.dst_conn)
            for e in graph.out_edges(node):
                for path_e in graph.memlet_tree(e):
                    if isinstance(path_e.dst, nodes.AccessNode) and path_e.dst.data in known_registers:
                        if e.src_conn in out_conn_registers:
                            out_conn_registers.remove(e.src_conn)

            known_registers |= in_conn_registers
            known_registers |= out_conn_registers
            estimate_internal = max([
                self.evaluate_state(node.sdfg, s, None, 
                                    known_registers)
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
                       old_register_arrays = set(),
                       current_array_reg_size = 0,
                       scope_dict: dict = None,
                       scope_children: dict = None):
        '''
        Evaluates Register spill for a whole state where scope_node indicates the outermost scope in which spills should be analyzed (within its inner scopes as well)
        If scope_node is None, then the whole graph will get analyzed.
        '''
        print(f"Evaluating state {graph} with scope node {scope_node}")
        # get some variables if they haven not already been inputted
        if scope_dict is None:
            scope_dict = graph.scope_dict()
        if scope_children is None:
            scope_children = graph.scope_children()


        subgraph = graph.scope_subgraph(scope_node) if scope_node is not None else SubgraphView(graph, graph.nodes())

        # 1. Create a proxy graph to determine the topological order of all tasklets 
        #    that reside inside our graph 

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
       
        # 2. Create a list of active sets that represent the maximum 
        #    number of active units of execution at a given timeframe. 
        #    This serves as estimate later 

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

        # 3. Determine all new arrays that reside inside the subgraph and that are 
        #    allocated to registers.
        register_arrays = dict()
        for node in subgraph.nodes():
            if isinstance(snode, nodes.AccessNode) and sdfg.data(node.data).storage == dtypes.StorageType.Register:
                if node.data not in register_arrays:
                    register_arrays[node.data] = sdfg.data(node.data).total_size

        # 4. Determine which registers to discard to global memory starting with the 
        #    largest one. Store all the remaining arrays
        
        total_max_size = 100 
        total_size = current_array_reg_size
        register_pure = dict() 
        for (aname, size) in sorted(register_arrays.items(), key = lambda a: a[1]):
            # nested sdfg: do not recount array
            if aname not in old_register_arrays:
                current_size = self.symbolic_evaluation(size)
                if total_size + current_size > total_max_size:
                    break 
                else:
                    total_size += current_size
                    register_pure[aname] = current_size 
        
        if self.debug:
            print("Pure Arrays:", register_pure)
        # 5. For each tasklet, determine inner, output and input register estimates 
        #    and store them. Do not include counts from registers that are in 
        #    pure_registers as they are counted seperately (else they would be 
        #    counted twice in input and output)

        input_scores, inner_scores, output_scores = {},{},{}
        for (node, ctx) in context.items():
            inner_scores[node] = self.evaluate_tasklet_inner(sdfg, graph, node, context = ctx, known_registers = set(itertools.chain(old_register_arrays, register_pure.keys())), known_register_size = total_size)
            input_scores[node] = self.evaluate_tasklet_input(sdfg, graph, node, context = ctx, known_registers = set(itertools.chain(old_register_arrays, register_pure.keys())))
            output_scores[node] = self.evaluate_tasklet_output(sdfg, graph, node, context = ctx, known_registers = set(itertools.chain(old_register_arrays, register_pure.keys())))

        # 6. TODO: CHANGEDESC: For each active set, determine the number of registers that are used for storage 
        registers_after_tasklet = collections.defaultdict(dict)
        for tasklet in context.keys():
            for oe in graph.out_edges(tasklet):
                for e in graph.memlet_tree(oe):
                    if isinstance(e.dst, nodes.AccessNode) and e.dst.data in register_pure:
                        if e.dst in registers_after_tasklet[tasklet]:
                            raise NotImplementedError("Not implemented for multiple input edges at a pure register transient")

                        registers_after_tasklet[tasklet][e.dst] = e.data.subset.num_elements()


        # 7. TODO: Modify!! Loop over all active sets and choose the one with maximum
        #    register usage and return that estimate 
        print(f"--- State Subgraph Analysis: {graph} ---")
        previous_tasklet_set = set()
        active_set_scores = list()
        used_register_arrays = dict()
        for tasklet_set in active_sets:
            # evaluate score for current tasklet set 

            # calculate score from active registers 
            '''
            storage_registers = set() 
            for tasklet in tasklet_set:
                storage_registers |= tasklet_registers[tasklet]
            '''
            
            # NOTE: once we can calculate subset diffs, we can implement this much 
            # more accurate and elegantly by adding / diffing subsets out to / of used_register_arrays at each iteration step
            
            # add used register arrays 
            for tasklet in tasklet_set - previous_tasklet_set:
                for (reg_node, reg_volume) in registers_after_tasklet[tasklet].items():
                    used_register_arrays[reg_node] = reg_volume

            
            # remove used register arrays
            for tasklet in previous_tasklet_set - tasklet_set:
                for reg_node in registers_after_tasklet[tasklet]:
                    del used_register_arrays[reg_node]
            
            # calculate total size per register, keep in mind that we only have to 
            # count register whose data is the same only once 

            
            used_register_space = dict()
            for (reg_node, volume) in used_register_arrays.items():
                if reg_node in used_register_space:
                    used_register_space[reg_node] = subsets.union(used_register_space[reg_node], volume)
                else:
                    used_register_space[reg_node] = volume 

            # set current score to sum of current active register arrays
            array_scores = self.symbolic_evaluation(sum(used_register_space.values()))
      

            # now calculate the rest of the contributions from tasklets
            tasklet_scores = sum(input_scores[t] + output_scores[t] + inner_scores[t] for t in tasklet_set)
            print(f"DEBUG->TASKSET {tasklet_set} INPUT:", sum(input_scores[t] for t in tasklet_set))
            print(f"DEBUG->TASKSET {tasklet_set} OUTPUT:", sum(output_scores[t] for t in tasklet_set))
            print(f"DEBUG->TASKSET {tasklet_set} INNER:", sum(inner_scores[t] for t in tasklet_set))

            if self.debug:
                print("-----------------------------")
                print("Active Set =", tasklet_set)
                print("Tasklet Scores =", tasklet_scores)
                print("Array Scores =", array_scores)
                print("Score =", array_scores + tasklet_scores)
                print("-----------------------------")

            set_score = array_scores + tasklet_scores


            active_set_scores.append(set_score)
            previous_tasklet_set = tasklet_set 
        
        if self.debug:
            print("---------------------------------------------")
            print(f"Subgraph {subgraph}: Total Score = {max(active_set_scores) if len(active_set_scores) > 0 else 0}")
            print("---------------------------------------------")

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
        print("REGISTERS USED", reg_count_required)
        print("REGISTERS AVAILABLE", reg_count_available)
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
        sdfg_copy.save(f'inspect_{self._i}.sdfg')

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
