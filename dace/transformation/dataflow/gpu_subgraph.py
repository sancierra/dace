""" Contains the GPU Subgraph transformation """


from dace import data, dtypes, sdfg as sd, registry
from dace.graph import nodes, nxutil
from dace.graph.graph import SubgraphView
from dace.transformation import pattern_matching, helpers
from dace.properties import Property, make_properties


@registry.autoregister_params(singlestate = True)
@make_properties
class GPUTransformSubgraph(pattern_matching.Transformation):
    """ Implements GPUTransformSubgraph Transformation

        Transforms a subgraph to a GPU-scheduled subgraph, creating
        GPU arrays outside for access copies
    """

    fullcopy = Property(desc = "Copy whole arrays instead of used subset",
                        dtype = bool, default = False)

    # TODO: ???
    toplevel_trans = Property(desc="Make all GPU transients top-level",
                        dtype = bool, default = False)

    register_trans = Property(
        desc = "Make all transients inside GPU maps registers",
        dtype = bool, default = False)


    sequential_innermaps = Property(desc = "Make all internal maps Sequential",
                                    dtype = bool, default = False)

    _entry = nodes.Node()


    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(GPUTransformSubgraph._entry)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict = False):
        node = graph.nodes()[candidate[GPUTransformSubgraph._entry]]

        # input: no accessNode, no map/stream exit, no stream entry
        if isinstance(node, nodes.AccessNode):
            return False
        if isinstance(node, nodes.ExitNode):
            return False
        if isinstance(node, nodes.ConsumeExit):
            return False
        if isinstance(node, nodes.ConsumeEntry):
            return False
        # special case: nested sdfg
        if isinstance(node, nodes.NestedSDFG):
            return True

        # TODO: For now I just select all nodes from current node to end of scope
        # find all child nodes recursively till end of scope

        scope_dict = graph.scope_dict()
        start_scope = scope_dict[node]
        if start_scope is None:
            scope_subgraph = SubgraphView(graph, graph.nodes())
        else:
            scope_subgraph = graph.scope_subgraph(start_scope, include_exit = False)


        queue = [node]
        subgraph = []
        while len(queue) > 0:
            current = queue.pop(0)
            if sd.scope_contains_scope(scope_dict, node, current):
                # add to list
                if current not in subgraph and current in scope_subgraph:
                    subgraph.append(current)

                    for e in graph.out_edges(current):
                        # add to queue
                        queue.append(e.dst)

        # checks:

        # not inside device level schedules
        # use is_devicelevel
        if sd.is_devicelevel(sdfg, graph, node):
            return False


        for current_node in subgraph:
            # every relative top level map must not contain any dynamic inputs
            if isinstance(current_node, nodes.MapEntry):
                if scope_dict[current_node] == start_scope:
                    if sd.has_dynamic_map_inputs(graph, current_node):
                        return False

            # also check for map schedules diallowed to transform to GPUs in general
            if isinstance(current_node, nodes.MapEntry) or isinstance(current_node, nodes.Reduce):
                if current_node.schedule in [dtypes.ScheduleType.MPI, dtypes.GPU_SCHEDULES]:
                    return False



            # make sure internal arrays are not on non default storage locations
            if isinstance(current_node, nodes.AccessNode):
                if(current_node.desc(sdfg).storage != dtypes.StorageType.Default
                        and current_node.desc(sdfg).storage != dtypes.StorageType.Register):
                        return False

        '''
        # not needed: if we are not at relative scope level 0 and output is a stream, return False
        if start_scope is not None:
            for current_node in subgraph:
                if isinstance(current_node, nodes.ExitNode):
                    if scope_dict[scope_dict[current_node]] == start_scope:
                        for edge in graph.out_edges(current_node):
                            dst = graph.memlet_path(edge)[-1].dst
                            if isinstance(dst, nodes.AccessNode) \
                                    and isinstance(sdfg.arrays[dst.data], data.Stream):
                                print("False -- Ground Level output Stream")
                                return False
        '''

        # success
        return True

    @staticmethod
    def match_to_str(graph, candidate):
        return str(graph.nodes()[candidate[GPUTransformSubgraph._entry]])

    def apply(self, sdfg):
        # proceed analogously to GPUTransformMap:
        # create nested sdfg and push to gpu

        graph = sdfg.nodes()[self.state_id]
        entry = graph.nodes()[self.subgraph[GPUTransformSubgraph._entry]]

        if isinstance(entry, nodes.NestedSDFG):
            # nested sdfg: skip the whole procedure and go directly to end
            nsdfg_node = entry
        else:
            # recalculate subgraph set

            scope_dict = graph.scope_dict()
            start_scope = scope_dict[entry]
            if start_scope is None:
                scope_subgraph = SubgraphView(graph, graph.nodes())
            else:
                scope_subgraph = graph.scope_subgraph(start_scope, include_exit = False)

            queue = [entry]
            subgraph = []
            while len(queue) > 0:
                current = queue.pop(0)
                if sd.scope_contains_scope(scope_dict, entry, current):
                    # add to list
                    if current not in subgraph and current in scope_subgraph:
                        subgraph.append(current)

                        for e in graph.out_edges(current):
                            # add to queue
                            queue.append(e.dst)

            # create subgraph view and then sdfg
            subgraph_view = SubgraphView(graph, subgraph)
            #print("subgraph_view")
            #print(subgraph_view)
            #print(subgraph_view.nodes())

            # nest as a subgraph, then push to GPU
            nsdfg_node = helpers.nest_state_subgraph(sdfg, graph,
                                                    subgraph_view,
                                                    full_data = self.fullcopy)

        # Analogously to GPUTransformMap
        # avoid importing loops
        print("##############")
        print("BEFORE")
        print(nsdfg_node)
        print(nsdfg_node.sdfg.nodes()[0].nodes())
        print(graph.nodes())
        print("###############")

        from dace.transformation.interstate import GPUTransformSDFG
        transformation = GPUTransformSDFG(0,0,{},0)
        transformation.register_trans = self.register_trans
        transformation.sequential_innermaps = self.sequential_innermaps
        transformation.toplevel_trans = self.toplevel_trans

        # apply transformation
        transformation.apply(nsdfg_node.sdfg)


        print("##############")
        print("AFTER")
        print(nsdfg_node)
        print(nsdfg_node.sdfg.nodes()[0].nodes())
        print(graph.nodes())
        print("###############")

        # inline back
        sdfg.apply_strict_transformations()
