# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Automatic optimization routines for SDFGs. """

from dace.sdfg.state import SDFGState
from dace import config, data as dt, dtypes, Memlet
from dace.sdfg import SDFG, nodes, graph as gr
from typing import Set, Tuple, Union
import warnings

# Transformations
from dace.transformation.dataflow import MapCollapse
from dace.transformation.interstate import LoopToMap

# Environments
from dace.libraries.blas.environments import intel_mkl as mkl, openblas

GraphViewType = Union[SDFG, SDFGState, gr.SubgraphView]


def greedy_fuse(graph_or_subgraph: GraphViewType, validate_all: bool) -> None:
    # TODO: If two maps share connected nodes (horizontal/vertical), fuse
    # TODO: run multiexpansion first
    pass


def tile_wcrs(graph_or_subgraph: GraphViewType, validate_all: bool) -> None:
    # Avoid import loops
    from dace.codegen.targets import cpp
    from dace.frontend import operations
    from dace.transformation import dataflow

    # Determine on which nodes to run the operation
    graph = graph_or_subgraph
    if isinstance(graph_or_subgraph, gr.SubgraphView):
        graph = graph_or_subgraph.graph
    if isinstance(graph, SDFG):
        for state in graph_or_subgraph.nodes():
            tile_wcrs(state, validate_all)
        return
    if not isinstance(graph, SDFGState):
        raise TypeError(
            'Graph must be a state, an SDFG, or a subgraph of either')
    sdfg = graph.parent

    edges_to_consider: Set[Tuple[gr.MultiConnectorEdge[Memlet],
                                 nodes.EntryNode]] = set()
    for edge in graph_or_subgraph.edges():
        if edge.data.wcr is not None:
            if (isinstance(edge.src, (nodes.ExitNode, nodes.NestedSDFG))
                    or isinstance(edge.dst, nodes.EntryNode)):
                # Do not consider intermediate edges
                continue
            reason = cpp.is_write_conflicted_with_reason(graph, edge)
            if reason is None or not isinstance(reason, nodes.EntryNode):
                # Do not consider edges that will not generate atomics or
                # atomics we cannot transform
                continue
            edges_to_consider.add((edge, reason))

    tile_size = config.Config.get('optimizer', 'autotile_size')
    debugprint = config.Config.get_bool('debugprint')

    transformed = set()
    for edge, mapentry in edges_to_consider:
        if mapentry in transformed:
            continue
        transformed.add(mapentry)
        # NOTE: The test below is crafted for Sympy to be "definitely True"
        if (mapentry.map.range.num_elements() < tile_size) == True:
            # If smaller than tile size, don't transform and instead make map sequential
            if debugprint:
                print(f'Making map "{mapentry}" sequential due to being '
                      'smaller than tile size')
            mapentry.map.schedule = dtypes.ScheduleType.Sequential
            continue

        print('will transform', mapentry)
        # MapTiling -> AccumulateTransient / AccumulateStream
        outer_mapentry = dataflow.MapTiling.apply_to(
            sdfg, dict(tile_sizes=(tile_size, )), _map_entry=mapentry)

        # Transform all outgoing WCR and stream edges
        mapexit = graph.exit_node(mapentry)
        outer_mapexit = graph.exit_node(outer_mapentry)
        for e in graph.out_edges(mapexit):
            if isinstance(sdfg.arrays[e.data.data], dt.Stream):
                mpath = graph.memlet_path(e)
                tasklet = mpath[0].src
                if not isinstance(tasklet, nodes.Tasklet) or len(mpath) != 3:
                    # TODO(later): Implement StreamTransient independently of tasklet
                    continue
                dataflow.StreamTransient.apply_to(sdfg,
                                                  _tasklet=tasklet,
                                                  _map_exit=mapexit,
                                                  _outer_map_exit=outer_mapexit)
            else:
                if e.data.is_empty() or e.data.wcr is None or e.data.wcr_nonatomic:
                    continue

                dtype = sdfg.arrays[e.data.data].dtype
                redtype = operations.detect_reduction_type(e.data.wcr)
                dataflow.AccumulateTransient.apply_to(
                    sdfg,
                    options=dict(identity=dtypes.reduction_identity(dtype, redtype),
                                 array=e.data.data),
                    _map_exit=mapexit,
                    _outer_map_exit=outer_mapexit)

    if debugprint and len(transformed) > 0:
        print(f'Optimized {len(transformed)} write-conflicted maps')


def find_fast_library(device: dtypes.DeviceType) -> str:
    # Returns the optimized library node implementations for the given target
    # device
    if device is dtypes.DeviceType.GPU:
        return ['cuBLAS', 'CUB', 'pure']
    elif device is dtypes.DeviceType.CPU:
        # TODO: add "is_installed" checks to environments
        result = []

        # BLAS calls
        # if mkl.IntelMKL.is_installed():
        #     result.append('MKL')
        # elif openblas.OpenBLAS.is_installed():
        #     result.append('OpenBLAS')

        return result + ['pure']

    return ['pure']


def auto_optimize(sdfg: SDFG,
                  device: dtypes.DeviceType,
                  validate: bool = True,
                  validate_all: bool = False) -> SDFG:
    """
    Runs a basic sequence of transformations to optimize a given SDFG to decent
    performance. In particular, performs the following:
        * Strict transformations
        * Strict auto-parallelization (loop-to-map)
        * Greedy application of SubgraphFusion
        * Tiled write-conflict resolution (MapTiling -> AccumulateTransient)
        * Tiled stream accumulation (MapTiling -> AccumulateTransient)
        * Collapse all maps to parallelize across all dimensions
        * Set all library nodes to expand to ``fast`` expansion, which calls
          the fastest library on the target device
    :param sdfg: The SDFG to optimize.
    :param validate: If True, validates the SDFG after all transformations
                     have been applied.
    :param validate_all: If True, validates the SDFG after every step.
    :return: The optimized SDFG.
    :note: Operates in-place on the given SDFG.
    :note: This function is still experimental and may harm correctness in
           certain cases. Please report an issue if it does.
    """
    # Strict transformations
    sdfg.apply_strict_transformations(validate=False, validate_all=validate_all)

    # Try to parallelize loops
    sdfg.apply_transformations_repeated(LoopToMap,
                                        strict=True,
                                        validate=False,
                                        validate_all=validate_all)

    # Map fusion
    greedy_fuse(sdfg, validate_all)

    # Tiled WCR and streams
    tile_wcrs(sdfg, validate_all)

    # Collapse maps
    sdfg.apply_transformations_repeated(MapCollapse,
                                        strict=True,
                                        validate=False,
                                        validate_all=validate_all)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.MapEntry):
            node.map.collapse = len(node.map.range)

    # Set all library nodes to expand to fast library calls
    implementation_prio = find_fast_library(device)
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, nodes.LibraryNode):
            for impl in implementation_prio:
                if impl in node.implementations:
                    node.implementation = impl
                    break
            else:
                warnings.warn('No fast library implementation found for "%s", '
                              'falling back to default.' % node.name)

    # TODO(later): Safe vectorization

    # Disable OpenMP parallel sections
    # TODO(later): Set on a per-SDFG basis
    config.Config.set('compiler', 'cpu', 'openmp_sections', value=False)

    # Validate at the end
    if validate or validate_all:
        sdfg.validate()

    return sdfg
