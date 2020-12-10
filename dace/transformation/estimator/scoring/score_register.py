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

        self._i = 0

    def evaluate_tasklet(self, 
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
            for ctx in (context):
                for i, memlet in enumerate(memlets):
                    memlets[i] = propagate_memlet(graph, memlet, ctx, False)
            
            # NOTE: subsets.num_elements() yields a better estimate than volume()
            estimate_connector = sum([m.subset.num_elements() for m in memlets])

        if isinstance(node, nodes.Tasklet):
            # get an estimate from code symbols not in conn tasklets 
            names = set(n.id for n in ast.walk(ast.parse(node.code.as_string)) if isinstance(n, ast.Name))
            # add them up and return
            estimate_internal = len(names - set(itertools.chain(node.in_connectors.keys(), node.out_connectors.keys())))
        else:
            # get an estimate for each nested sdfg state and max over these values 
            estimate_internal = max([self.evaluate_state(node.sdfg, s, None, nested = True) for s in node.sdfg.nodes()])

            # do something different here 
        estimate_connector = self.symbolic_evaluation(estimate_connector)
        estimate_internal = self.symbolic_evaluation(estimate_internal)
        print("ESTIMATE CONNECTOR", )
        print("ESTIMATE INTERNAL")

    def evaluate_state(self,
                       sdfg: SDFG, 
                       graph: SDFGState, 
                       subgraph: SubgraphView, 
                       nested: bool  = False,
                       scope_dict: dict = None,
                       outer_map_entry: nodes.MapEntry = None):

        # get some variables if they haven not already been inputted 
        if subgraph is None:
            subgraph = SubgraphView(graph, graph.nodes())
        if scope_dict is None:
            scope_dict = graph.scope_dict 
        if outer_map_entry is None:
            outer_map_entry = next(n for n in subgraph.nodes() if isinstance(n, nodes.MapEntry) and scope_dict[n] is None)
        
        # loop over all tasklets 
        context = dict()
        for node in subgraph.nodes():
            if isinstance(node, nodes.Tasklet):
                # see whether scope is contained in outer_entry 
                scope = scope_dict[node]
                current_context = []
                while scope and scope != outer_map_entry:
                    current_context.append(scope)
                    scope = scope_dict[scope]
                if scope:
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

        connected_components = utils.concurrent_subgraphs(proxy)
        active_sets = list()
        for cc in connected_components:
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
        
        # for each tasklet, get current scope 



        




    def estimate_spill(self, sdfg, graph, subgraph):
        # get register used count
        reg_count_required = 0
        tasklet_symbols = set()
        visited_containers = set()
        register_read_write = 0

        sc = graph.scope_children()
        outer_map_entry = next(n for n in sc[None] if isinstance(n, nodes.MapEntry))
        outer_map = outer_map_entry.map

        for node in subgraph.nodes():
            if isinstance(node, nodes.AccessNode) and sdfg.data(node.data).storage == dtypes.StorageType.Register:
                # container requirements
                if node.data not in visited_containers:
                    reg_count_required += sdfg.data(node.data).total_size
                    visited_containers.add(node.data)
                # traffic requirements
                memlets = [e.data for e in itertools.chain(graph.out_edges(node), graph.in_edges(node))]
                result_mm = propagation.propagate_subset(memlets = memlets, arr = sdfg.data(node.data), params = outer_map.params, rng = outer_map.range)

                register_read_write += result_mm.volume
                print("REGRW", register_read_write)

            if node in sc[outer_map_entry] and isinstance(node, nodes.Tasklet):

                if node.code.language == dtypes.Language.Python:
                    node.code.code
                    names = set(node.id for node in ast.walk(ast.parse(node.code.as_string)) if isinstance(node, ast.Name))
                    tasklet_symbols = tasklet_symbols if len(tasklet_symbols) > len(names) else names 
                else:
                    warnings.warn('WARNING: Register Score cannot evaluate non Python block')
            
  

        # add length of max tasklet symbols to register count
        if not isinstance(reg_count_required, int):
            reg_count_required = self.symbolic_evaluation(reg_count_required)
            print("TO ADD", len(tasklet_symbols))
            reg_count_required += len(tasklet_symbols)
        print("TOTAL REQUIRED", reg_count_required)

        # get register available count
        threads = Config.get('compiler', 'cuda', 'default_block_size')
        threads = functools.reduce(lambda a, b: a * b,
                                   [int(e) for e in threads.split(',')])
        reg_count_available = self.register_per_block / (min(
            1028, 16 * threads))
        print("TOTAL AVAILABLE", reg_count_available)

        # get number of total read/writes to registers
        type
        param_substitute = {
            p: r
            for (p, r) in zip(outer_map.params, outer_map.range.size())
        }
        print("PARAM SUBSTITUTE", param_substitute)
        if not isinstance(register_read_write, int):
            register_read_write.subs(param_substitute)
            register_read_write = self.symbolic_evaluation(register_read_write)

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
