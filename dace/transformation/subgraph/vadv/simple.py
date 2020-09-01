import dace
import math
import numpy as np
from dace.transformation.dataflow import MapFission, MapTiling, MapCollapse
from dace.transformation.subgraph import pipeline
from dace.sdfg.propagation import propagate_memlets_sdfg

datatype = dace.float64
I = dace.symbol('I')
J = dace.symbol('J')
K = dace.symbol('K')
halo = dace.symbol('halo')


@dace.program
def stencil_simple(w_in: datatype[J, K+1, I], crlatu: datatype[J], \
                   crlato: datatype[J], hdmask: datatype[J, K+1, I], \
                   w: datatype[J, K+1, I]):

    __tmp_lap_910 = np.ndarray([J, K+1, I], datatype)
    @dace.map
    def s1(k: _[0:K], x: _[halo-1:J-halo+1], z: _[halo-1:I-halo+1]):
        w_in_in << w_in[x-1:x+2, 0, z-1:z+2]
        crlato_in << crlato[x]
        crlatu_in << crlatu[x]
        __tmp_lap_910_out >> __tmp_lap_910[x,k,z]

        __local_delta_1476 = (w_in_in[(2, 1)] - w_in_in[(1, 1)])
        __local_delta_1479 = (w_in_in[((- 0), 1)] - w_in_in[(1, 1)])
        __local_laplacian_1473 = ((((w_in_in[(1, 2)] + w_in_in[(1, (- 0))]) - (2.0 * w_in_in[(1, 1)])) + (crlato_in * __local_delta_1476)) + (crlatu_in * __local_delta_1479))
        __tmp_lap_910_out = __local_laplacian_1473
    @dace.map
    def s2(k: _[0:K], x: _[halo:J-halo], z:_[halo:I-halo]):
        w_in_in << w_in[x-1:x+2, 0, z-1:z+2]
        crlato_in << crlato[x-1:x+1]
        hdmask_in << hdmask[x, 0, z]
        __tmp_lap_910_in << __tmp_lap_910[x-1:x+2, 0, z-1:z+2]

        w_out >> w[x,k,z]

        __local_delta_1482 = (__tmp_lap_910_in[(1, 2)] - __tmp_lap_910_in[(1, 1)])
        __local_flx_1041 = __local_delta_1482
        __local_delta_1488 = (w_in_in[(1, 2)] - w_in_in[(1, 1)])
        __local_diffusive_flux_x_1485 = (0.0 if ((__local_flx_1041 * __local_delta_1488) > 0.0) else __local_flx_1041)
        __local_delta_1491 = (__tmp_lap_910_in[(1, 1)] - __tmp_lap_910_in[(1, (- 0))])
        __local_flx_1073 = __local_delta_1491
        __local_delta_1497 = (w_in_in[(1, 1)] - w_in_in[(1, (- 0))])
        __local_diffusive_flux_x_1494 = (0.0 if ((__local_flx_1073 * __local_delta_1497) > 0.0) else __local_flx_1073)
        __local_delta_flux_x_1023 = (__local_diffusive_flux_x_1485 - __local_diffusive_flux_x_1494)
        __local_delta_1500 = (__tmp_lap_910_in[(2, 1)] - __tmp_lap_910_in[(1, 1)])
        __local_fly_1108 = (crlato_in[1] * __local_delta_1500)
        __local_delta_1506 = (w_in_in[(2, 1)] - w_in_in[(1, 1)])
        __local_diffusive_flux_y_1503 = (0.0 if ((__local_fly_1108 * __local_delta_1506) > 0.0) else __local_fly_1108)
        __local_delta_1509 = (__tmp_lap_910_in[(1, 1)] - __tmp_lap_910_in[((- 0), 1)])
        __local_fly_1142 = (crlato_in[(- 0)] * __local_delta_1509)
        __local_delta_1515 = (w_in_in[(1, 1)] - w_in_in[((- 0), 1)])
        __local_diffusive_flux_y_1512 = (0.0 if ((__local_fly_1142 * __local_delta_1515) > 0.0) else __local_fly_1142)
        __local_delta_flux_y_1088 = (__local_diffusive_flux_y_1503 - __local_diffusive_flux_y_1512)
        w_out = (w_in_in[(1, 1)] - (hdmask_in * (__local_delta_flux_x_1023 + __local_delta_flux_y_1088)))


def apply_map_fission(sdfg):
    sdfg.apply_transformations(MapFission)

def apply_pre_tiling(sdfg, tile_size = 32):
    # get nested sdfg
    # for every map inside that is not kmap_fission
    # pre_tile it accordingly
    graph = sdfg.nodes()[0]
    for node in graph.nodes():
        if isinstance(node, dace.sdfg.nodes.MapEntry) \
                        and node.label == 's1':
            subgraph = {MapTiling._map_entry: graph.nodes().index(node)}
            transformation = MapTiling(0, 0, subgraph, 0)
            transformation.tile_sizes = (1, tile_size + 2, tile_size +2)
            transformation.strides = (1, tile_size, tile_size)
            transformation.strides_offset = (0,0,0)
            transformation.apply(sdfg)

        if isinstance(node, dace.sdfg.nodes.MapEntry) \
                        and node.label == 's2':
            subgraph = {MapTiling._map_entry: graph.nodes().index(node)}
            transformation = MapTiling(0, 0, subgraph, 0)
            transformation.tile_sizes = (1, tile_size, tile_size)
            transformation.strides = (1, tile_size, tile_size)
            transformation.strides_offset = (0,0,0)
            transformation.apply(sdfg)
    return


def fuse_stencils(sdfg, gpu):
    graph = sdfg.nodes()[0]
    if not gpu:
        pipeline.fusion(sdfg, graph, transient_allocation = dace.dtypes.StorageType.CPU_Heap)
    else:
        pipeline.fusion(sdfg, graph, transient_allocation = dace.dtypes.StorageType.GPU_Shared)


def test(compile = True, compile_all = False, view = True, gpu = False, tile_size = 8):
    # define DATATYPE
    DATATYPE = np.float64
    # define symbols
    I = 21
    J = 21
    K = 1
    halo = 2

    # define arrays
    crlato = 2*np.random.rand(J).astype(DATATYPE)-1
    crlatu = 2*np.random.rand(J).astype(DATATYPE)-1
    hdmask = 2*np.random.rand( J, K + 1, I ).astype(DATATYPE)-2
    w_in = 4*np.random.rand( J, K + 1, I).astype(DATATYPE)-2


    # compile -- first without anyting
    sdfg = stencil_simple.to_sdfg()
    sdfg.apply_strict_transformations()

    if gpu:
        sdfg.apply_gpu_transformations()

    if view:
        sdfg.view()

    if compile:
        w1 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)

        sdfg._name = 'basline'
        csdfg = sdfg.compile()
        csdfg(crlato = crlato, crlatu = crlatu,
              hdmask = hdmask, w_in = w_in,
              w = w1,
              I=I, J=J, K=K, halo = halo)

    apply_pre_tiling(sdfg, tile_size)
    if view:
        sdfg.view()
    if compile:
        w2 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        sdfg._name = 'pre_tiled'
        csdfg = sdfg.compile()
        csdfg(crlato = crlato, crlatu = crlatu,
              hdmask = hdmask, w_in = w_in,
              w = w2,
              I=I, J=J, K=K, halo = halo)


    fuse_stencils(sdfg, gpu)
    if view:
        sdfg.view()
    if compile:
        w3 = np.zeros([ J, K + 1, I ], dtype = DATATYPE)
        sdfg._name = 'fused'
        csdfg = sdfg.compile()
        csdfg(crlato = crlato, crlatu = crlatu,
              hdmask = hdmask, w_in = w_in,
              w = w3,
              I=I, J=J, K=K, halo = halo)

    if compile:
        print(np.linalg.norm(w_in))
        print(np.linalg.norm(w1))
        print(np.linalg.norm(w2))
        print(np.linalg.norm(w3))


        assert np.allclose(w1, w2)
        assert np.allclose(w1, w3)

        print("PASS")

if __name__ == '__main__':
    test(compile = True, view = False, gpu = False)
