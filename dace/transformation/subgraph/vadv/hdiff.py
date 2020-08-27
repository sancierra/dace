import dace
import math
import numpy as np
from dace.transformation.dataflow import MapFission
from dace.transformation.dataflow import MapTiling


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
            nsdfg = node
            ngraph = nsdfg.nodes()[0]
    for node in ngraph.nodes():
        if isinstance(node, dace.sdfg.nodes.MapEntry) \
                        and node.lable != 'kmap_fission':
            subgraph = {MapTiling._map_entry: ngraph.nodes().index(node)}
            transformation = MapTiling(0, 0, subgraph, 0)
            transformation.tile_sizes = (tile_size, 0, tile_size)
            transformation.apply(nsdfg)

    # TODO

def test(compile = True, view = True):
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



if __name__ == '__main__':
    test(view = False)
