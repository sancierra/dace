import dace
import math
import numpy as np
from dace.transformation.dataflow import MapFission, MapTiling, MapCollapse
#from dace.transformation.subgraph import pipeline

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

def apply_pre_tiling(sdfg, tile_size = 32):
    # get nested sdfg
    # for every map inside that is not kmap_fission
    # pre_tile it accordingly
    graph = sdfg.nodes()[0]
    for node in graph.nodes():
        if isinstance(node, dace.sdfg.nodes.NestedSDFG):
            nsdfg = node.sdfg
            ngraph = nsdfg.nodes()[0]
    for node in ngraph.nodes():
        if isinstance(node, dace.sdfg.nodes.MapEntry) \
                        and node.label != 'kmap_fission':
            subgraph = {MapTiling._map_entry: ngraph.nodes().index(node)}
            transformation = MapTiling(0, 0, subgraph, 0)
            transformation.tile_sizes = (tile_size, 0, tile_size)
            transformation.strides_offset = (1,1)
            transformation.apply(nsdfg)
    return


def collapse_outer_maps(sdfg):
    graph = sdfg.nodes()[0]
    for node in graph.nodes():
        if isinstance(node, dace.sdfg.nodes.NestedSDFG):
            nsdfg = node.sdfg
            ngraph = nsdfg.nodes()[0]
    for outer_node in ngraph.nodes():
        if isinstance(outer_node, dace.sdfg.nodes.MapEntry)\
                    and outer_node.label == 'kmap_fission':
            inner_node = ngraph.out_edges(outer_node)[0].dst
            subgraph = \
                {MapCollapse._outer_map_entry: ngraph.nodes().index(outer_node),
                 MapCollapse._inner_map_entry: ngraph.nodes().index(inner_node)}
            transformation = MapCollapse(0,0, subgraph, 0)
            transformation.apply(nsdfg)


def test(compile = True, view = True, tile_size = 32):
    # define DATATYPE
    DATATYPE = np.float64
    # define symbols
    I = 48
    J = 48
    K = 48
    halo = 2

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
    if view:
        sdfg.view()
    if compile:
        pp1 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        w1 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        v1 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        u1 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)

        csdfg = sdfg.compile()
        csdfg(pp_in = pp_in, crlato = crlato, crlatu = crlatu,
              hdmask = hdmask, w_in = w_in, v_in = v_in, u_in = u_in,
              pp = pp1, w = w1, v = v1, u = u1,
              I=I, J=J, K=K, halo = halo)


    apply_map_fission(sdfg)
    if view:
        sdfg.view()
    if compile:
        pp2 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        w2 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        v2 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        u2 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)

        csdfg = sdfg.compile()
        csdfg(pp_in = pp_in, crlato = crlato, crlatu = crlatu,
              hdmask = hdmask, w_in = w_in, v_in = v_in, u_in = u_in,
              pp = pp2, w = w2, v = v2, u = u2,
              I=I, J=J, K=K, halo = halo)


    apply_pre_tiling(sdfg, tile_size)
    if view:
        sdfg.view()
    if compile:
        pp3 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        w3 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        v3 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        u3 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)

        csdfg = sdfg.compile()
        csdfg(pp_in = pp_in, crlato = crlato, crlatu = crlatu,
              hdmask = hdmask, w_in = w_in, v_in = v_in, u_in = u_in,
              pp = pp3, w = w3, v = v3, u = u3,
              I=I, J=J, K=K, halo = halo)

    collapse_outer_maps(sdfg)
    if view:
        sdfg.view()
    if compile:
        pp4 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        w4 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        v4 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        u4 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)

        csdfg = sdfg.compile()
        csdfg(pp_in = pp_in, crlato = crlato, crlatu = crlatu,
              hdmask = hdmask, w_in = w_in, v_in = v_in, u_in = u_in,
              pp = pp4, w = w4, v = v4, u = u4,
              I=I, J=J, K=K, halo = halo)

    if compile:
        print(np.linalg.norm(pp1))
        print(np.linalg.norm(pp2))
        print(np.linalg.norm(w1))
        print(np.linalg.norm(w2))
        print(np.linalg.norm(v1))
        print(np.linalg.norm(v2))
        print(np.linalg.norm(u1))
        print(np.linalg.norm(u2))

        assert np.allclose(pp1, pp2)
        assert np.allclose(w1, w2)
        assert np.allclose(v1, v2)
        assert np.allclose(u1, u2)

        assert np.allclose(pp1, pp3)
        assert np.allclose(w1, w3)
        assert np.allclose(v1, v3)
        assert np.allclose(u1, u3)

        assert np.allclose(pp1, pp4)
        assert np.allclose(w1, w4)
        assert np.allclose(v1, v4)
        assert np.allclose(u1, u4)



if __name__ == '__main__':
    test(view = False)
