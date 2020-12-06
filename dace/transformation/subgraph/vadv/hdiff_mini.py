import dace
import math
import numpy as np
from dace.transformation.dataflow import MapFission, MapTiling, MapCollapse, DeduplicateAccess
from dace.transformation.interstate import InlineSDFG
from dace.transformation.subgraph import pipeline
from dace.transformation.subgraph.stencil_tiling import StencilTiling
from dace.transformation.subgraph.composite import CompositeFusion
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.sdfg.graph import SubgraphView
import dace.sdfg.nodes as nodes

from copy import deepcopy as dcpy

import sys

DATATYPE = np.float32

def view_graphs():
    dace.sdfg.SDFG.from_file('original_graphs/hdiff_v2.sdfg').view()
    dace.sdfg.SDFG.from_file('original_graphs/hdiff_full.sdfg').view()
    dace.sdfg.SDFG.from_file('original_graphs/dedup.sdfg').view()


def eliminate_k_memlet(sdfg):
    SYM = 'y'
    graph = sdfg.nodes()[0]
    ngraph, nsdfg = None, None
    for node in graph.nodes():
        if isinstance(node, dace.sdfg.nodes.NestedSDFG):
            nsdfg = node.sdfg
            ngraph = nsdfg.nodes()[0]

    for node in ngraph.nodes():
        if isinstance(node, nodes.MapEntry):
            for e in ngraph.out_edges(node):
                x = dace.symbol(SYM)
                e.data.subset.replace({x:'0'})
        if isinstance(node, nodes.MapExit):
            for e in ngraph.in_edges(node):
                x = dace.symbol(SYM)
                e.data.subset.replace({x:'0'})

def fix_arrays(sdfg):
    for container in sdfg.arrays:
        print(sdfg.data(container).shape)
        print(type(sdfg.data(container).shape))

    for container in sdfg.arrays:
        if len(sdfg.data(container).shape) == 3:
            sdfg.data(container).shape = (\
                sdfg.data(container).shape[0]-1, \
                sdfg.data(container).shape[1], \
                sdfg.data(container).shape[2])
            #sdfg.data(container).strides = [dace.data._prod(sdfg.data(container).shape[i + 1:]) for i in range(3)]
            #sdfg.data(container).total_size = dace.data._prod(sdfg.data(container).shape)

    for container in sdfg.arrays:
        print(sdfg.data(container).shape)
        print(type(sdfg.data(container).shape))


def apply_map_fission(sdfg):
    sdfg.apply_transformations(MapFission)

def apply_stencil_tiling(sdfg, nested = False,
                         tile_size = 1, sequential = True,
                         gpu = False, unroll = False):
    graph = sdfg.nodes()[0]
    ngraph, nsdfg = None, None
    if nested:
        for node in graph.nodes():
            if isinstance(node, dace.sdfg.nodes.NestedSDFG):
                nsdfg = node.sdfg
                ngraph = nsdfg.nodes()[0]
    else:
        nsdfg = sdfg
        ngraph = graph

    subgraph = SubgraphView(ngraph, ngraph.nodes())
    transformation = StencilTiling(subgraph, nsdfg.sdfg_id,
                                   nsdfg.nodes().index(ngraph))
    transformation.unroll_loops = unroll
    assert transformation.can_be_applied(sdfg, subgraph)
    assert CompositeFusion.can_be_applied(sdfg, subgraph)
    # TODO
    if len(tile_size) == 1:
        tile_size = tile_size * 2
    transformation.strides = (1, 1, tile_size[0], tile_size[1])
    if sequential:
        transformation.schedule = dace.dtypes.ScheduleType.Sequential
    elif gpu:
        transformation.schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock

    transformation.apply(sdfg)

    return



def collapse_outer_maps(sdfg, nested = False):
    graph = sdfg.nodes()[0]
    ngraph, nsdfg = None, None
    if nested:
        for node in graph.nodes():
            if isinstance(node, dace.sdfg.nodes.NestedSDFG):
                nsdfg = node.sdfg
                ngraph = nsdfg.nodes()[0]
    else:
        nsdfg = sdfg
        ngraph = graph

    for outer_node in ngraph.nodes():
        if isinstance(outer_node, dace.sdfg.nodes.MapEntry)\
                    and outer_node.label == 'kmap_fission':
            inner_node = ngraph.out_edges(outer_node)[0].dst
            subgraph = \
                {MapCollapse._outer_map_entry: ngraph.nodes().index(outer_node),
                 MapCollapse._inner_map_entry: ngraph.nodes().index(inner_node)}
            transformation = MapCollapse(0,0, subgraph, 0)
            transformation.apply(nsdfg)

def fuse_stencils(sdfg, gpu,
                  nested = False,
                  deduplicate = False,
                  sequential = False):
    graph = sdfg.nodes()[0]
    ngraph, nsdfg = None, None

    kwargs = {}
    kwargs['propagate_source'] = False
    if gpu and not sequential:
        kwargs['transient_allocation'] = dace.dtypes.StorageType.GPU_Shared
        kwargs['schedule_innermaps'] = dace.dtypes.ScheduleType.GPU_ThreadBlock
    if gpu and sequential:
        kwargs['transient_allocation'] = dace.dtypes.StorageType.Register
        kwargs['schedule_innermaps'] = dace.dtypes.ScheduleType.Sequential
    if deduplicate:
        kwargs['consolidate_source'] = True
        kwargs['deduplicate_source'] = True
    if nested:
        for node in graph.nodes():
            if isinstance(node, dace.sdfg.nodes.NestedSDFG):
                nsdfg = node.sdfg
                ngraph = nsdfg.nodes()[0]
    else:
        nsdfg = sdfg
        ngraph = graph

    pipeline.fusion(nsdfg, ngraph, **kwargs)

def test(compile = True, view = True,
         gpu = False, nested = False,
         tile_size = 32, deduplicate = False,
         sequential = False,
         datatype = np.float32,
         unroll = False):
    # define symbols
    I = np.int32(128)
    J = np.int32(128)
    K = np.int32(80)
    halo = np.int32(4)

    # define arrays
    pp_in = np.random.rand(J, K+1, I).astype(DATATYPE)
    u_in = np.random.rand(J, K+1, I).astype(DATATYPE)
    crlato = np.random.rand(J).astype(DATATYPE)
    crlatu = np.random.rand(J).astype(DATATYPE)
    acrlat0 = np.random.rand(J).astype(DATATYPE)
    crlavo = np.random.rand(J).astype(DATATYPE)
    crlavu = np.random.rand(J).astype(DATATYPE)
    hdmask = np.random.rand( J, K+1, I ).astype(DATATYPE)
    w_in = np.random.rand( J, K+1, I).astype(DATATYPE)
    v_in = np.random.rand( J, K+1, I).astype(DATATYPE)


    # compile -- first without anything
    sdfg = dace.sdfg.SDFG.from_file('hdiff_mini32.sdfg')
    sdfg.specialize({'I':I, 'J':J, 'K':K, 'halo':halo})
    #fix_arrays(sdfg)
    ###eliminate_k_memlet(sdfg)

    if gpu:
        sdfg.apply_gpu_transformations()

    if view:
        sdfg.view()
    
    if compile:
        
        pp4 = np.zeros([ J, K+1, I ], dtype = DATATYPE)
        w4 = np.zeros([ J, K+1, I ], dtype = DATATYPE)
        v4 = np.zeros([ J, K+1, I ], dtype = DATATYPE)
        u4 = np.zeros([ J, K+1, I ], dtype = DATATYPE)
        sdfg._name = 'baseline'
        csdfg = sdfg.compile()
        csdfg(pp_in = pp_in, crlato = crlato, crlatu = crlatu,
              acrlat0 = acrlat0, crlavo = crlavo, crlavu =crlavu,
              hdmask = hdmask, w_in = w_in, v_in = v_in, u_in = u_in,
              pp = pp4, w = w4, v = v4, u = u4,
              I=I, J=J, K=K, halo = halo)
 

    apply_stencil_tiling(sdfg, tile_size=tile_size,
                         nested=nested, sequential = sequential,
                         gpu = gpu, unroll = unroll)

    if view:
        sdfg.view()

    if compile:
        pass
        
        pp3 = np.zeros([ J, K+1, I ], dtype = DATATYPE)
        w3 = np.zeros([ J, K+1, I ], dtype = DATATYPE)
        v3 = np.zeros([ J, K+1, I ], dtype = DATATYPE)
        u3 = np.zeros([ J, K+1, I ], dtype = DATATYPE)
        sdfg._name = 'pre_tiling'
        csdfg = sdfg.compile()
        csdfg(pp_in = pp_in, crlato = crlato, crlatu = crlatu,
              acrlat0 = acrlat0, crlavo = crlavo, crlavu =crlavu,
              hdmask = hdmask, w_in = w_in, v_in = v_in, u_in = u_in,
              pp = pp3, w = w3, v = v3, u = u3,
              I=I, J=J, K=K, halo = halo)
        del csdfg
        
    if compile:
        print("Pre Tiling Results")
        print(np.allclose(pp4, pp3))
        print(np.allclose(w4, w3))
        print(np.allclose(v4, v3))
        print(np.allclose(u4, u3))
        
    fuse_stencils(sdfg,
                  gpu=gpu,
                  nested=nested,
                  deduplicate = deduplicate,
                  sequential = sequential)
    sdfg.apply_transformations_repeated(DeduplicateAccess)
    sdfg.apply_strict_transformations()
    if view:
        sdfg.view()
    if compile:
        pp5 = np.zeros([ J, K+1, I ], dtype = DATATYPE)
        w5 = np.zeros([ J, K+1, I ], dtype = DATATYPE)
        v5 = np.zeros([ J, K+1, I ], dtype = DATATYPE)
        u5 = np.zeros([ J, K+1, I ], dtype = DATATYPE)
        sdfg._name = 'fused'
        csdfg = sdfg.compile()
        csdfg(pp_in = pp_in, crlato = crlato, crlatu = crlatu,
              acrlat0 = acrlat0, crlavo = crlavo, crlavu =crlavu,
              hdmask = hdmask, w_in = w_in, v_in = v_in, u_in = u_in,
              pp = pp5, w = w5, v = v5, u = u5,
              I=I, J=J, K=K, halo = halo)
        del csdfg

    if compile:
        print(np.linalg.norm(pp4))
        print(np.linalg.norm(pp3))
        print(np.linalg.norm(pp5))

        print(np.linalg.norm(w4))
        print(np.linalg.norm(w3))
        print(np.linalg.norm(w5))

        print(np.linalg.norm(v4))
        print(np.linalg.norm(v3))
        print(np.linalg.norm(v5))

        print(np.linalg.norm(u4))
        print(np.linalg.norm(u3))
        print(np.linalg.norm(u5))

        print("Pre Tiling")
        print(np.allclose(pp4, pp3))
        print(np.allclose(w4, w3))
        print(np.allclose(v4, v3))
        print(np.allclose(u4, u3))
        print("Fusion")
        print(np.allclose(pp4, pp5))
        print(np.allclose(w4, w5))
        print(np.allclose(v4, v5))
        print(np.allclose(u4, u5))


if __name__ == '__main__':
    try:
        seq = sys.argv[1] == 'register'
        nseq = sys.argv[1] == 'shared'
        assert nseq == True or seq == True
        sequential = seq
        tile1 = int(sys.argv[2])
        tile2 = int(sys.argv[3])

    except:
        print("Usage: mode tile1 tile2")
        raise RuntimeError()
    test(view = False, compile = True, nested = False,
         gpu = True, deduplicate = False, tile_size = (tile1, tile2),
         sequential = sequential, unroll =  False)
