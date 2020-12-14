""" This file implements the ExecutionScore class """

from dace.transformation.subgraph import SubgraphFusion, StencilTiling, helpers
from dace.transformation.subgraph.composite import CompositeFusion
from dace.transformation.dataflow import DeduplicateAccess
from dace.properties import make_properties, Property
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

    def propagate_outward(self, graph: SDFGState, memlets: List[Memlet],
                          context: List[Union[nodes.MapEntry,
                                              nodes.NestedSDFG]]):
        '''
        Propagates Memlets outwards given map entry nodes in context vector. 
        Assumes that first entry is innermost entry closest to memlet 
        '''
        for ctx in (context):
            if isinstance(context, nodes.MapEntry):
                for i, memlet in enumerate(memlets):
                    memlets[i] = propagation.propagate_memlet(
                        graph, memlet, ctx, False)
            else:
                raise NotImplementedError("TODO")

        return memlets

    def evaluate_register_traffic(self,
                                  sdfg: SDFG,
                                  graph: SDFGState,
                                  subgraph: SubgraphView,
                                  scope_node: nodes.Node,
                                  scope_dict: dict = None):
        ''' 
        Evaluates traffic in a scope subgraph with given scope 
        node that flows out of and into registers 
        '''

        if not scope_dict:
            scope_dict = graph.scope_dict()

        register_traffic = 0

        for node in subgraph.nodes():
            if isinstance(node, nodes.AccessNode) and sdfg.data(
                    node.data).storage == dtypes.StorageType.Register:
                # see whether scope is contained in outer_entry
                scope = scope_dict[node]
                current_context = []
                while scope and scope != scope_node:
                    current_context.append(scope)
                    scope = scope_dict[scope]
                if scope == scope_node:
                    # add to traffic
                    memlets = list(
                        itertools.chain(graph.out_edges(node),
                                        graph.in_edges(node)))
                    self.propagate_outward(graph, memlets, current_context)
                    for memlet in memlets:
                        register_traffic += memlet.volume

        return self.symbolic_evaluation(register_traffic)

    def evaluate_tasklet(self,
                         graph: SDFGState,
                         node: Union[nodes.Tasklet, nodes.NestedSDFG],
                         context: List[nodes.MapEntry],
                         active: bool):
        '''
        Evaluates a tasklet or nested sdfg node for a register size usage estimate 

        :param node:    tasklet or nsdfg node 
        :param context: list of map entries to propagate thru ending with toplevel 
        :param nested:  specify whether we are in a nested sdfg
        :param active:  EXPERIMENTAL: specify whether access node is active or already done at this point 
        '''

        estimate_connector = 0
        estimate_internal = 0
        if isinstance(node, nodes.Tasklet):
            # 1. get an estimate from incoming / outgoing memlets 
            # if we aren't in a nested sdfg, add subset size estimates from connectors
            memlets = [
                e.data for e in itertools.chain(graph.in_edges(node) if active else [],
                                                graph.out_edges(node))
            ]
            self.propagate_outward(graph, memlets, context)
            # NOTE: subsets.num_elements() yields a better estimate than volume() here
            estimate_connector = sum([m.subset.num_elements() for m in memlets])

            # 2. get an estimate from code symbols not in conn tasklets
            if active:
                if node.code.langague == dtypes.Language.Python:
                    names = set(n.id
                                for n in ast.walk(ast.parse(node.code.as_string))
                                if isinstance(n, ast.Name))
                    # add them up and return
                    estimate_internal = len(names - set(
                        itertools.chain(node.in_connectors.keys(),
                                        node.out_connectors.keys())))
                else:
                    warnings.warn(
                        'WARNING: Register Score cannot evaluate non Python block')
                    # just use some shady estimate
                    estimate_internal = max(
                        8, math.ceil(node.code.as_string.count('\n') / 2))
        else:
            # nested sdfg
            # get an estimate for each nested sdfg state and max over these values
            if active:
                estimate_internal = max([
                    self.evaluate_state(node.sdfg, s, None)
                    for s in node.sdfg.nodes()
                ])
            else:
                memlets = [
                e.data for e in itertools.chain(graph.out_edges(node))
                ]
                self.propagate_outward(graph, memlets, context)
                # NOTE: subsets.num_elements() yields a better estimate than volume() here
                estimate_connector = sum([m.subset.num_elements() for m in memlets])


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
                       scope_node: Node,
                       nested: bool = False,
                       scope_dict: dict = None,
                       scope_children: dict = None):
        '''
        Evaluates Register spill for a whole state where scope_node indicates the outermost scope in which spills should be analyzed (within its inner scopes as well)
        '''
        print(f"Evaluating {graph} with scope node {scope_node}")
        # get some variables if they haven not already been inputted
        if scope_dict is None:
            scope_dict = graph.scope_dict()
        if scope_children is None:
            scope_children = graph.scope_children()

        subgraph = SubgraphView(graph, scope_children[scope_node])

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
        print("CONTEXTS", context)
        print("Building proxy graph")
        proxy = nx.MultiDiGraph()
        for n in subgraph.nodes():
            if n in context:  # has to be inside fused map
                proxy.add_node(n)
        for n in subgraph.nodes():
            if n in context:  # has to be inside fused map
                for e in graph.all_edges(n):
                    proxy.add_edge(e.src, e.dst)
        
        print("Remove all nodes not Tasklets")
        # remove all nodes that are not Tasklets
        for n in graph.nodes():
            if n in proxy.nodes():
                if not isinstance(n, nodes.Tasklet):
                    for p in proxy.predecessors(n):
                        for c in proxy.successors(n):
                            proxy.add_edge(p, c)
                    proxy.remove_node(n)
        print(f"Success!, Proxy = {proxy}")
        # set up ancestor and successor array
        print("Ancestors and Successors")
        ancestors, successors = {}, {}
        for n in proxy.nodes():
            successors[n] = set(proxy.successors(n))
            ancestors[n] = set(nx.ancestors(proxy, n))

        active_sets = list()
        print("Building active sets")
        for cc in nx.weakly_connected_components(proxy):
            print("NEXT CC")
            queue = [node for node in cc if len(ancestors[node]) == 0]
            while len(queue) > 0:
                active_sets.append(set(queue))
                for tasklet in queue:
                    # remove ancestor link (for newly added elements to queue)
                    for tasklet_successor in successors[tasklet]:
                        if tasklet in ancestors[tasklet_successor]:
                            ancestors[tasklet_successor].remove(tasklet)
                    if tasklet not in queue:
                        queue.append(tasklet)
                # if all ancestor in degrees in successors are 0
                # then tasklet is ready to be removed from the queue
                for tasklet in queue:
                    for tasklet_successor in successors[tasklet]:
                        if len(ancestors[tasklet_successor]) > 0:
                            break
                    else:
                        queue.remove(tasklet)

        # add up scores in respective active sets and return the max
        print("ACTIVE_SETS=", active_sets)

        last_set = set()
        active_set_scores = list()
        for tasklet_set in active_sets:
            set_score = 0
            for tasklet in tasklet_set:
                sc = self.evaluate_tasklet(graph = graph,
                                           node = tasklet,
                                           context = context[tasklet],
                                           active = True)
                # TODO: change active to false under certain conditions? 
                # or do a O(N) maximum evaluation among those that weren't in last 
                set_score += sc 
            active_set_scores.append(set_score)
            last_set = tasklet_set 
                        
        return max(active_set_scores)



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
        if reg_count_required > reg_count_available:
            register_read_write = self.evaluate_register_traffic(
                sdfg = sdfg,
                graph = graph,
                scope_node = outer_map,
                scope_dict = scope_dict)
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
        print("ESTIMATE SPILL")
        outer_entry = next(n for n in subgraph_copy.nodes() if isinstance(n, nodes.MapEntry))
        spill_traffic = self.estimate_spill(sdfg_copy, graph_copy,
                                            outer_entry)

        return (current_traffic + spill_traffic) / self._base_traffic

    @staticmethod
    def name():
        return "Estimated Memlet Traffic plus Spill"
