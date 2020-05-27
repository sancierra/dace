import dace
import numpy as np

import dace.perf.roofline


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

    dace.reduce(lambda a, b: a + b, tmp, C, axis=2)


if __name__ == '__main__':
    M.set(300)
    N.set(300)
    K.set(300)


    sdfg = GEMM1.to_sdfg()

    peak_bandwidth = 1.867 * 64 * 2 / 8
    peak_performance = 2.7 * 4 * 2 * 4

    symbols = {M: 300, N:300, K:300}
    spec = dace.perf.roofline.PerformanceSpec(peak_bandwidth, peak_performance, dace.float64)
    roof = dace.perf.roofline.Roofline(spec, symbols, debug = True)

    #GT = dace.transformation.interstate.gpu_transform_sdfg.GPUTransformSDFG(0,0,{},0)
    #GT.apply(sdfg)

    sdfg.expand_library_nodes()
    sdfg.apply_strict_transformations()

    #dace.perf.sdfv_roofline.view(sdfg, roof)


    #print("SDFGRooflineOptimizer")
    optimizer = dace.perf.optimizer.SDFGRooflineOptimizer(sdfg, roof, inplace = False)
    optimizer.optimize()
    #sdfg.verify()
    #print("SDFGOptimizer")
    #optimizer = dace.transformation.optimizer.SDFGOptimizer(sdfg)
    #optimizer.optimize()



    '''
    A = np.random.rand(M.get(), K.get()).astype(np.float64)
    B = np.random.rand(K.get(), N.get()).astype(np.float64)
    C = np.zeros((M.get(), N.get()),dtype = np.float64)
    # this is a dace program that reduces A*B (3dim) onto C
    GEMM1(A=A, B=B, C=C, M=M, N=N, K=K)
    '''

    #sdfg.expand_library_nodes()
    #sdfg.apply_strict_transformations()

    #sdfg.view()
    #csdfg = sdfg.compile()
    #csdfg(A=A, B=B, C=C, N=N, M=M, K=K)
    #print(np.linalg.norm(C))
    #print(np.linalg.norm(A@B))
