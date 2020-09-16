import dace
import math
import numpy as np
from dace.transformation.dataflow import MapFission, MapTiling, MapCollapse
from dace.transformation.interstate import InlineSDFG
from dace.transformation.subgraph import pipeline
from dace.transformation.subgraph.stencil_tiling import StencilTiling
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.sdfg.graph import SubgraphView
import dace.sdfg.nodes as nodes

from copy import deepcopy as dcpy

import sys 

SDFG_FILE = 'hdiff.sdfg'

def view_graphs():
    dace.sdfg.SDFG.from_file('hdiff.sdfg').view()
    dace.sdfg.SDFG.from_file('hdiff_v2.sdfg').view()
    dace.sdfg.SDFG.from_file('hdiff_fused.sdfg').view()
    dace.sdfg.SDFG.from_file('dedup.sdfg').view()
def get_sdfg():
    sdfg = dace.sdfg.SDFG.from_file(SDFG_FILE)
    sdfg.apply_strict_transformations()
    graph = sdfg.nodes()[0]
    # fix: make outer transients non-transient
    # for reproducibility
    #for node in graph.nodes():
    #    if isinstance(node, dace.sdfg.nodes.AccessNode):
    #        sdfg.data(node.data).transient = False
    return sdfg

def eliminate_k_memlet(sdfg):
    graph = sdfg.nodes()[0]
    ngraph, nsdfg = None, None
    for node in graph.nodes():
        if isinstance(node, dace.sdfg.nodes.NestedSDFG):
            nsdfg = node.sdfg
            ngraph = nsdfg.nodes()[0]

    for node in ngraph.nodes():
        if isinstance(node, nodes.MapEntry):
            for e in ngraph.out_edges(node):
                x = dace.symbol('x')
                e.data.subset.replace({x:'0'})
        if isinstance(node, nodes.MapExit):
            for e in ngraph.in_edges(node):
                x = dace.symbol('x')
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

    for container in sdfg.arrays:
        print(sdfg.data(container).shape)
        print(type(sdfg.data(container).shape))


def apply_map_fission(sdfg):
    sdfg.apply_transformations(MapFission)

def apply_stencil_tiling(sdfg, nested = False, tile_size = 1, sequential = True, gpu = False):
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

    first = [1365, 1392, 1419, 1446]
    reference_range = dcpy(next((node for node in ngraph.nodes()
                if isinstance(node, dace.sdfg.nodes.MapEntry)
                and node.label != 'kmap_fission'
                and not any([str(f) in node.label for f in first]))).range)
    print("ref", reference_range)

    subgraph = SubgraphView(ngraph, ngraph.nodes())
    transformation = StencilTiling(subgraph, nsdfg.sdfg_id,
                                   nsdfg.nodes().index(ngraph))
    print(transformation.match(sdfg, subgraph), "==", True)
    # TODO
    K = dace.symbol('K')
    if len(tile_size) == 1:
        tile_size = tile_size * 2
    transformation.strides = (1, 1, tile_size[0], tile_size[1])
    if sequential:
        transformation.schedule = dace.dtypes.ScheduleType.Sequential
    elif gpu:
        transformation.schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock
    
    transformation.apply(sdfg)

    return

def apply_pre_tiling(sdfg, nested = False, tile_size = 32):
    # for every map inside that is not kmap_fission
    # pre_tile it accordingly
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

    for node in ngraph.nodes():
        if isinstance(node, dace.sdfg.nodes.MapEntry) \
                        and node.label != 'kmap_fission':
            first = [1365, 1392, 1419, 1446]
            if any([str(f) in node.label for f in first]):
                subgraph = {MapTiling._map_entry: ngraph.nodes().index(node)}
                transformation = MapTiling(0, 0, subgraph, 0)
                transformation.tile_sizes = (tile_size + 2, 1, tile_size + 2)
                transformation.strides = (tile_size, 1, tile_size)
                transformation.tile_offset = (0,0)
                transformation.apply(nsdfg)
            else:
                subgraph = {MapTiling._map_entry: ngraph.nodes().index(node)}
                transformation = MapTiling(0, 0, subgraph, 0)
                transformation.tile_sizes = (tile_size, 1, tile_size)
                transformation.strides = (tile_size, 1, tile_size)
                transformation.tile_offset = (0,0)
                transformation.apply(nsdfg)
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
    if gpu:
        kwargs['transient_allocation'] = dace.dtypes.StorageType.GPU_Shared
    if sequential:
        # will override gpu, ok
        kwargs['transient_allocation'] = dace.dtypes.StorageType.Register
        kwargs['sequential_innermaps'] = True
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
         sequential = False):
    # define DATATYPE
    DATATYPE = np.float64
    # define symbols
    I = np.int32(128)
    J = np.int32(128)
    K = np.int32(80)
    halo = np.int32(4)

    # define arrays
    pp_in = np.random.rand(J, K, I).astype(DATATYPE)
    u_in = np.random.rand(J, K, I).astype(DATATYPE)
    crlato = np.random.rand(J).astype(DATATYPE)
    crlatu = np.random.rand(J).astype(DATATYPE)
    acrlat0 = np.random.rand(J).astype(DATATYPE)
    crlavo = np.random.rand(J).astype(DATATYPE)
    crlavu = np.random.rand(J).astype(DATATYPE)
    hdmask = np.random.rand( J, K, I ).astype(DATATYPE)
    w_in = np.random.rand( J, K, I).astype(DATATYPE)
    v_in = np.random.rand( J, K, I).astype(DATATYPE)


    # compile -- first without anyting
    sdfg = get_sdfg()
    sdfg._propagate = False
    sdfg.propagate = False
    sdfg.add_symbol('halo', int)

    fix_arrays(sdfg)
    eliminate_k_memlet(sdfg)

    if gpu:
        sdfg.apply_gpu_transformations()
        for node in sdfg.nodes()[0].nodes():
            if isinstance(node, dace.sdfg.nodes.NestedSDFG):
                node.schedule = dace.dtypes.ScheduleType.GPU_Device

    if view:
        sdfg.view()

    if compile:
        pp1 = np.zeros([ J, K, I ], dtype = DATATYPE)
        w1 = np.zeros([ J, K, I ], dtype = DATATYPE)
        v1 = np.zeros([ J, K, I ], dtype = DATATYPE)
        u1 = np.zeros([ J, K, I ], dtype = DATATYPE)

       # sdfg._name = 'baseline'
       # csdfg = sdfg.compile()
       # csdfg(pp_in = pp_in, crlato = crlato, crlatu = crlatu,
       #       acrlat0 = acrlat0, crlavo = crlavo, crlavu =crlavu,
       #       hdmask = hdmask, w_in = w_in, v_in = v_in, u_in = u_in,
       #       pp_out = pp1, w_out = w1, v_out = v1, u_out = u1,
       #       I=I, J=J, K=K, halo = halo)


    apply_map_fission(sdfg)
    if not nested:
        sdfg.apply_strict_transformations()
    if view:
        sdfg.view()
    if compile:
        pp2 = np.zeros([ J, K, I ], dtype = DATATYPE)
        w2 = np.zeros([ J, K, I ], dtype = DATATYPE)
        v2 = np.zeros([ J, K, I ], dtype = DATATYPE)
        u2 = np.zeros([ J, K, I ], dtype = DATATYPE)

        #sdfg._name = 'fission'
        #csdfg = sdfg.compile()
        #csdfg(pp_in = pp_in, crlato = crlato, crlatu = crlatu,
        #      acrlat0 = acrlat0, crlavo = crlavo, crlavu =crlavu,
        #      hdmask = hdmask, w_in = w_in, v_in = v_in, u_in = u_in,
        #      pp_out = pp2, w_out = w2, v_out = v2, u_out = u2,
        #      I=I, J=J, K=K, halo = halo)

    collapse_outer_maps(sdfg, nested=nested)
    if view:
        sdfg.view()
    if compile:
        pp4 = np.zeros([ J, K, I ], dtype = DATATYPE)
        w4 = np.zeros([ J, K, I ], dtype = DATATYPE)
        v4 = np.zeros([ J, K, I ], dtype = DATATYPE)
        u4 = np.zeros([ J, K, I ], dtype = DATATYPE)
        sdfg._name = 'baseline'
        csdfg = sdfg.compile()
        csdfg(pp_in = pp_in, crlato = crlato, crlatu = crlatu,
              acrlat0 = acrlat0, crlavo = crlavo, crlavu =crlavu,
              hdmask = hdmask, w_in = w_in, v_in = v_in, u_in = u_in,
              pp_out = pp4, w_out = w4, v_out = v4, u_out = u4,
              I=I, J=J, K=K, halo = halo)

    apply_stencil_tiling(sdfg, tile_size=tile_size, nested=nested, sequential = sequential, gpu = gpu)
    if view:
        sdfg.view()
    if compile:
        pp3 = np.zeros([ J, K, I ], dtype = DATATYPE)
        w3 = np.zeros([ J, K, I ], dtype = DATATYPE)
        v3 = np.zeros([ J, K, I ], dtype = DATATYPE)
        u3 = np.zeros([ J, K, I ], dtype = DATATYPE)
        sdfg._name = 'pre_tiling'
        csdfg = sdfg.compile()
        csdfg(pp_in = pp_in, crlato = crlato, crlatu = crlatu,
              acrlat0 = acrlat0, crlavo = crlavo, crlavu =crlavu,
              hdmask = hdmask, w_in = w_in, v_in = v_in, u_in = u_in,
              pp_out = pp3, w_out = w3, v_out = v3, u_out = u3,
              I=I, J=J, K=K, halo = halo)


    fuse_stencils(sdfg,
                  gpu=gpu,
                  nested=nested,
                  deduplicate = deduplicate,
                  sequential = sequential)
    if view:
        sdfg.view()
    if compile:
        pp5 = np.zeros([ J, K, I ], dtype = DATATYPE)
        w5 = np.zeros([ J, K, I ], dtype = DATATYPE)
        v5 = np.zeros([ J, K, I ], dtype = DATATYPE)
        u5 = np.zeros([ J, K, I ], dtype = DATATYPE)
        sdfg._name = 'fused'
        csdfg = sdfg.compile()
        csdfg(pp_in = pp_in, crlato = crlato, crlatu = crlatu,
              acrlat0 = acrlat0, crlavo = crlavo, crlavu =crlavu,
              hdmask = hdmask, w_in = w_in, v_in = v_in, u_in = u_in,
              pp_out = pp5, w_out = w5, v_out = v5, u_out = u5,
              I=I, J=J, K=K, halo = halo)


    if compile:
        print(np.linalg.norm(pp4))
        print(np.linalg.norm(w4))
        print(np.linalg.norm(v4))
        print(np.linalg.norm(u4))
        print("Baseline")
        print(np.allclose(pp4, pp4))
        print(np.allclose(w4, w4))
        print(np.allclose(v4, v4))
        print(np.allclose(u4, u4))
        print("Collapse")
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
        tile1 = int(sys.argv[2])
        tile2 = int(sys.argv[3])
        
    except:
        print("Useage: mode tile1 tile2")
        raise RuntimeError()
    test(view = False, compile = True, nested = False,
         gpu = True, deduplicate = False, tile_size = (tile1, tile2),
         sequential = False)
