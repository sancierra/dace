""" This file implements the ExecutionScore class """

from dace.transformation.subgraph import SubgraphFusion, StencilTiling, helpers
from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.dataflow import DeduplicateAccess
from dace.properties import make_properties, Property
from dace.sdfg import SDFG, SDFGState
from dace.sdfg.graph import SubgraphView
from dace.config import Config

import dace.sdfg.propagation as propagation
import dace.sdfg.nodes as nodes
import dace.symbolic as symbolic
import dace.dtypes as dtypes
import dace.subsets as subsets
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


@make_properties
class RegisterScore(MemletScore):
    '''
    Evaluates Subgraphs by just running their
    SDFG and returning the runtime
    '''

    register_per_block = Property(desc="No of Registers per Block available",
                                  dtype=int,
                                  default=65536)
                                  #default = 81920)

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
        

    def propagate_outward(self,
                          memlets: List[dace.Memlet],
                          context: List[Union[nodes.MapEntry, nodes.NestedSDFG]]):
        '''
        Propagates Memlets outwards given map entry nodes in context vector. 
        Assumes that first entry is innermost entry closest to memlet 
        '''
        for ctx in (context):
            if isinstance(context, nodes.MapEntry):
                for i, memlet in enumerate(memlets):
                    memlets[i] = propagation.propagate_memlet(graph, memlet, ctx, False)
            else:
                raise NotImplementedError("TODO")
                
        return memlets 
            
    def evaluate_register_traffic(self,
                                  graph: SDFGState, 
                                  subgraph: SubgraphView,
                                  scope_node: nodes.Node,
                                  scope_dict: dict = None):
        
        ''' 
        Evaluates traffic in a scope subgraph with given scope node that flows out of and into registers 
        '''

        if not scope_dict:
            scope_dict = graph.scope_dict()

        register_traffic = 0

        for node in subgraph.nodes():
            if isinstance(node, nodes.AccessNode) and sdfg.data(node.data).storage == dtypes.StorageType.Register:
                # see whether scope is contained in outer_entry 
                scope = scope_dict[node]
                current_context = []
                while scope and scope != scope_node:
                    current_context.append(scope)
                    scope = scope_dict[scope]
                if scope == scope_node:
                    # add to traffic 
                    self.propagate_outward(memlets, current_context)
                    for memlet in memlets:
                        register_traffic += memlet.volume
        
        return self.symbolic_evaluation(register_traffic)
        

    def evaluate_tasklet(self, 
                         graph: SDFGState,
                         node: Union[nodes.Tasklet, nodes.NestedSDFG], 
                         context: List[nodes.MapEntry],
                         nested = False):
        
        '''
        Evaluates a tasklet or nested sdfg node for a register size usage estimate 
        :param node:    tasklet or nsdfg node 
        :param context: list of map entries to propagate thru ending with toplevel 
        :param nested:  specify whether we are in a nested sdfg
        '''

        estimate_connector = 0
        estimate_internal = 0

        if not nested:
            # if we aren't in a nested sdfg, add subset size estimates from connectors
            memlets = [e.data for e in itertools.chain(graph.in_edges(node), graph.out_edges(node))]
            self.propagate_outward(memlets, context)
            
            # NOTE: subsets.num_elements() yields a better estimate than volume() here 
            estimate_connector = sum([m.subset.num_elements() for m in memlets])

        if isinstance(node, nodes.Tasklet):
            # get an estimate from code symbols not in conn tasklets 
             
            if node.code.langague == dtypes.Language.Python:
                names = set(n.id for n in ast.walk(ast.parse(node.code.as_string)) if isinstance(n, ast.Name))
                # add them up and return
                estimate_internal = len(names - set(itertools.chain(node.in_connectors.keys(), node.out_connectors.keys())))
            else:
                warnings.warn('WARNING: Register Score cannot evaluate non Python block')
                # just use some shady estimate 
                estimate_internal = max(8, math.ceil(node.code.as_string.count('\n')/2))
        else:
            # get an estimate for each nested sdfg state and max over these values 
            estimate_internal = max([self.evaluate_state(node.sdfg, s, None, nested = True) for s in node.sdfg.nodes()])

            # do something different here 
        estimate_connector = self.symbolic_evaluation(estimate_connector)
        estimate_internal = self.symbolic_evaluation(estimate_internal)
        if self.debug:
            print(f"Node {node}")
            print(f"Estimator: Connector = {estimate_connector}")
            print(f"Estimate: Internal =", estimate_internal)

        return estimate_connector + estimate_internal



    def evaluate_state(self,
                       sdfg: SDFG, 
                       graph: SDFGState, 
                       scope_node: graph.Node, 
                       nested: bool  = False,
                       scope_dict: dict = None):
        '''
        Evaluates Register spill for a whole state where scope_node indicates the outermost scope in which spills should be analyzed (within its inner scopes as well)
        '''

        # get some variables if they haven not already been inputted 
        if scope_dict is None:
            scope_dict = graph.scope_dict()
        subgraph = SubgraphView(graph, scope_dict[scope_node])
        
        # loop over all tasklets 
        context = dict()
        for node in subgraph.nodes():
            if isinstance(node, nodes.Tasklet):
                # see whether scope is contained in outer_entry 
                scope = scope_dict[node]
                current_context = []
                while scope and scope != scope_node:
                    current_context.append(scope)
                    scope = scope_dict[scope]
                if scope == scope_node:
                    context[node] = current_context 
        
        # similarly as in transient_reuse, create a proxy graph only 
        # containing tasklets that are *INSIDE* our fused map

        proxy = nx.MultiDiGraph()
        for n in subgraph.nodes():
            if n in context: # has to be inside fused map
                proxy.add_node(n)
        for n in subgraph.nodes():
            if n in context: # has to be inside fused map
                for e in graph.all_edges(n):
                    proxy.add_edge(e.src, e.dst)
        
        # remove all nodes that are not Tasklets 
        for n in graph.nodes():
            if n in proxy.nodes():
                if not isinstance(n, nodes.Tasklet):
                    for p in proxy.predecessors(n):
                        for c in proxy.successors(n):
                            proxy.add_edge(p, c)   
                    proxy.remove(n)

        # set up ancestor and successor array 
        ancestors, successors = {}, {}  
        for n in proxy.nodes():
            successors[n] = set(proxy.successors(n))
            ancestors[n] = set(nx.ancestors(proxy, n))

        active_sets = list()
        for cc in nx.weakly_connected_components(proxy):
            queue = [node for node in cc if len(ancestors[node]) == 0]
            while len(queue) > 0:
                active_sets.append(queue.copy())
                # remove ancestor link 
                for tasklet in queue:
                    for tasklet_successor in successors[tasklet]:
                        if tasklet in ancestors[tasklet_successor]:
                            ancestors[tasklet_successor].remove(tasklet)
                # if all ancestor in degrees in successors are 0
                # then tasklet is ready to be removed from the queue 
                for tasklet in queue:
                    ready_to_remove = True  
                    for tasklet_successor in successors[tasklet]:
                        if len(ancestors[tasklet_successor]) > 0:
                            ready_to_remove = False 
                            break  
                    
                    if ready_to_remove:
                        queue.remove(tasklet)
        
        # evaluate each tasklet
        tasklet_scores = {tasklet : self.evaluate_tasklet(graph, tasklet, ctx) for (tasklet, ctx) in context.items()}

        # add up scores in respective active sets and return the max
        active_sets_scores = [sum(tasklet_scores[t] for t in tasklet_list) for tasklet_list in active_sets]
        return max(active_sets_scores)



    def evaluate_available(self, outer_map):
        ''' FORNOW: get register available count '''

        register_read_write = 0
        threads = Config.get('compiler', 'cuda', 'default_block_size')
        threads = functools.reduce(lambda a, b: a * b,
                                   [int(e) for e in threads.split(',')])
        reg_count_available = self.register_per_block / (min(
            1028, 16 * threads))

        # get number of total read/writes to registers
        param_substitute = {
            p: r
            for (p, r) in zip(outer_map.params, outer_map.range.size())
        }
        if not isinstance(register_read_write, int):
            register_read_write.subs(param_substitute)
            register_read_write = self.symbolic_evaluation(register_read_write)

        return register_read_write 



    def estimate_spill(self, sdfg, graph, subgraph):
        ''' 
        estimates spill for a fused subgraph that contains one outer map entry 
        '''
        # get some variables 
        scope_dict = graph.scope_dict()
        outer_map_entry = next(n for n in sc[None] if isinstance(n, nodes.MapEntry))
        outer_map = outer_map_entry.map

        # get register used count
        reg_count_required = self.evaluate_state(sdfg = sdfg, graph = graph, scope_node = outer_map_entry, scope_dict = scope_dict)
        # get register provided count (FORNOW)
        reg_count_available = self.evaluate_available(sdfg = sdfg, graph = graph)
        # get register traffic count 
        if reg_count_required > reg_count_available:
            register_read_write = self.evaluate_register_traffic(outer_map = outer_map)
        else:
            register_read_write = 0
       
        spill_fraction = max(0, (reg_count_required - reg_count_available) /
                             reg_count_available)
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
        spill_traffic = self.estimate_spill(sdfg_copy, graph_copy,
                                            subgraph_copy)

        return (current_traffic + spill_traffic) / self._base_traffic

    @staticmethod
    def name():
        return "Estimated Memlet Traffic plus Spill"
