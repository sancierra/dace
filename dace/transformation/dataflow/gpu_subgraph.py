""" Contains the GPU Subgraph transformation """


from dace import data, dtypes, sdfg, registry
from dace.graph import nodes
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
    )

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

        return True



        # input: no accessNode, no map/stream exit
        if isinstance(node, nodes.AccessNode):
            return False
        if isinstance(node, nodes.ExitNode):
            return False
        if isinstance(node, nodes.ConsumeExit):
            return False
        # special case: nested sdfg
        if isinstance(node, nodes.NestedSDFG):
            return True

        # TODO: For now we just select all nodes from current node to end of scope
        # find all nodes recursively till end of scope

        scope = graph.scope_dict()
        start_scope = scope[node]

        scope_subgraph = graph.scope_subgraph(start_scope, include_exit = True)
        queue = [node]
        subgraph = []
        while len(queue) > 0:



        # check end of scope requirements
        # check in score requirements

        # ( )
        # check whether anything in there is on GPU / MPI / devicelevel
        # check whether any maps in there have dynamic inputs ???
        # streams?
        # ( )

    def apply(self, sdfg):
        # create nested sdfg and push to gpu
