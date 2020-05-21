import dace
import numpy as np

import dace.perf.roofline
import dace.libraries.standard as stdlib
from dace.transformation.heterogeneous.reduce_map import ReduceMap


N = dace.symbol('N')
K = dace.symbol('K')
M = dace.symbol('M')

# method 1
@dace.program
def GEMM1(A: dace.float64[M, K], B: dace.float64[K, N],
             C: dace.float64[M, N]):
        # Transient variable
    #C[:] = 0
    tmp = np.ndarray([M, N, K], dtype=A.dtype)
    @dace.map
    def multiplication(i: _[0:M], j: _[0:N], k: _[0:K]):
        in_A << A[i,k]
        in_B << B[k,j]
        out >> tmp[i,j,k]
        out = in_A * in_B

    dace.reduce(lambda a, b: a * b, tmp, C, axis=2)


if __name__ == '__main__':
    M.set(300)
    N.set(300)
    K.set(300)


    sdfg = GEMM1.to_sdfg()
    sdfg.apply_strict_transformations()
    reduce_node = None
    # get reduce node
    for node in sdfg.nodes()[0].nodes():
        if isinstance(node, stdlib.Reduce):
            reduce_node = node


    
    transformation = ReduceMap(sdfg_id = sdfg.sdfg_list.index(sdfg),
                               state_id = 0,
                               subgraph = {ReduceMap._reduce: sdfg.nodes()[0].nodes().index(reduce_node)},
                               expr_index = 0)


    transformation.apply(sdfg)




    sdfg.view()
