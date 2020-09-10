import dace
import math
import numpy as np
from dace.transformation.dataflow import MapFission, MapTiling, MapCollapse
from dace.transformation.interstate import InlineSDFG
from dace.transformation.subgraph import pipeline
from dace.transformation.subgraph.stencil_tiling import StencilTiling
from dace.sdfg.propagation import propagate_memlets_sdfg

def view_graphs():
    dace.sdfg.SDFG.from_file('hdiff.sdfg').view()
    dace.sdfg.SDFG.from_file('hdiff_fused.sdfg').view()
    dace.sdfg.SDFG.from_file('dedup.sdfg').view()
def get_sdfg():
    sdfg = dace.sdfg.SDFG.from_file('hdiff.sdfg')
    sdfg.apply_strict_transformations()
    graph = sdfg.nodes()[0]
    # fix: make outer transients non-transient
    # for reproducibility
    #for node in graph.nodes():
    #    if isinstance(node, dace.sdfg.nodes.AccessNode):
    #        sdfg.data(node.data).transient = False
    return sdfg

def apply_map_fission(sdfg):
    sdfg.apply_transformations(MapFission)

def apply_stencil_tiling(sdfg, nested = False, tile_size = 1):
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
    reference_range = next((node for node in ngraph.nodes()
                if isinstance(node, dace.sdfg.nodes.MapEntry)
                and node.label != 'kmap_fission'
                and not any([str(f) in node.label for f in first]))).range

    for node in ngraph.nodes().copy():
        if isinstance(node, dace.sdfg.nodes.MapEntry) \
                        and node.label != 'kmap_fission':

            subgraph = {StencilTiling._map_entry: ngraph.nodes().index(node)}
            transformation = StencilTiling(0, 0, subgraph, 0)
            transformation.reference_range = reference_range
            transformation.stencil_size = ((-1,2),)
            transformation.strides = (tile_size,)
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

def fuse_stencils(sdfg, gpu, nested = False, deduplicate = False):
    graph = sdfg.nodes()[0]
    ngraph, nsdfg = None, None

    kwargs = {}
    kwargs['propagate_source'] = False
    if gpu:
        kwargs['transient_allocation'] = dace.dtypes.StorageType.GPU_Shared
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

def test(compile = True, view = True, gpu = False, nested = False,
         tile_size = 32, deduplicate = False):
    # define DATATYPE
    DATATYPE = np.float64
    # define symbols
    I = np.int32(48)
    J = np.int32(48)
    K = np.int32(48)
    halo = np.int32(2)

    # define arrays
    pp_in = np.random.rand(J, K + 1, I).astype(DATATYPE)
    u_in = np.random.rand(J, K + 1, I).astype(DATATYPE)
    crlato = np.random.rand(J).astype(DATATYPE)
    crlatu = np.random.rand(J).astype(DATATYPE)
    hdmask = np.random.rand( J, K + 1, I ).astype(DATATYPE)
    w_in = np.random.rand( J, K + 1, I).astype(DATATYPE)
    v_in = np.random.rand( J, K + 1, I).astype(DATATYPE)


    # compile -- first without anyting
    sdfg = get_sdfg()
    sdfg._propagate = False
    sdfg.propagate = False
    if gpu:
        sdfg.apply_gpu_transformations()

    #if view:
    #    sdfg.view()

    if compile:
        pp1 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        w1 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        v1 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        u1 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)

        sdfg._name = 'baseline'
        csdfg = sdfg.compile()
        csdfg(pp_in = pp_in, crlato = crlato, crlatu = crlatu,
              hdmask = hdmask, w_in = w_in, v_in = v_in, u_in = u_in,
              pp = pp1, w = w1, v = v1, u = u1,
              I=I, J=J, K=K, halo = halo)


    apply_map_fission(sdfg)
    if not nested:
        sdfg.apply_strict_transformations()
    #if view:
    #    sdfg.view()
    if compile:
        pp2 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        w2 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        v2 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        u2 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)

        sdfg._name = 'fission'
        csdfg = sdfg.compile()
        csdfg(pp_in = pp_in, crlato = crlato, crlatu = crlatu,
              hdmask = hdmask, w_in = w_in, v_in = v_in, u_in = u_in,
              pp = pp2, w = w2, v = v2, u = u2,
              I=I, J=J, K=K, halo = halo)


    apply_stencil_tiling(sdfg, tile_size=tile_size, nested=nested)
    if view:
        sdfg.view()
    if compile:
        pp3 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        w3 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        v3 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        u3 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        sdfg._name = 'pre_tiling'
        csdfg = sdfg.compile()
        csdfg(pp_in = pp_in, crlato = crlato, crlatu = crlatu,
              hdmask = hdmask, w_in = w_in, v_in = v_in, u_in = u_in,
              pp = pp3, w = w3, v = v3, u = u3,
              I=I, J=J, K=K, halo = halo)

    collapse_outer_maps(sdfg, nested=nested)
    if view:
        sdfg.view()
    if compile:
        pp4 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        w4 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        v4 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        u4 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        sdfg._name = 'collapse_outer'
        csdfg = sdfg.compile()
        csdfg(pp_in = pp_in, crlato = crlato, crlatu = crlatu,
              hdmask = hdmask, w_in = w_in, v_in = v_in, u_in = u_in,
              pp = pp4, w = w4, v = v4, u = u4,
              I=I, J=J, K=K, halo = halo)

    fuse_stencils(sdfg, gpu=gpu, nested=nested, deduplicate = deduplicate)
    if view:
        sdfg.view()
    if compile:
        pp5 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        w5 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        v5 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        u5 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        sdfg._name = 'fused'
        csdfg = sdfg.compile()
        csdfg(pp_in = pp_in, crlato = crlato, crlatu = crlatu,
              hdmask = hdmask, w_in = w_in, v_in = v_in, u_in = u_in,
              pp = pp5, w = w5, v = v5, u = u5,
              I=I, J=J, K=K, halo = halo)

    if compile:
        print("Fission")
        print(np.allclose(pp1, pp2))
        print(np.allclose(w1, w2))
        print(np.allclose(v1, v2))
        print(np.allclose(u1, u2))
        print("Pre_tiling")
        print(np.allclose(pp1, pp3))
        print(np.allclose(w1, w3))
        print(np.allclose(v1, v3))
        print(np.allclose(u1, u3))
        print("Collapse")
        print(np.allclose(pp1, pp4))
        print(np.allclose(w1, w4))
        print(np.allclose(v1, v4))
        print(np.allclose(u1, u4))
        print("Fusion")
        print(np.allclose(pp1, pp5))
        print(np.allclose(w1, w5))
        print(np.allclose(v1, v5))
        print(np.allclose(u1, u5))


if __name__ == '__main__':
    #view_graphs()
    test(view = True, compile = False, nested = False,
         gpu = False, deduplicate = True)
