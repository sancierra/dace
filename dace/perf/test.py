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
    tmp = np.ndarray([M, N, K], dtype=A.dtype)
    @dace.map
    def multiplication(i: _[0:M], j: _[0:N], k: _[0:K]):
        in_A << A[i,k]
        in_B << B[k,j]
        out >> tmp[i,j,k]
        out = in_A * in_B

    dace.reduce(lambda a, b: a + b, tmp, C, axis=2)

if __name__ == '__main__':
    #M.set(300)
    #N.set(300)
    #K.set(300)


    sdfg = GEMM1.to_sdfg()

    peak_bandwidth = 1.867 * 64 * 2 / 8
    peak_performance = 2.7 * 4 * 2 * 4

    symbols = {M: 300, N:300, K:300}
    spec = dace.perf.roofline.PerformanceSpec(peak_bandwidth, peak_performance, dace.float64)
    roof = dace.perf.roofline.Roofline(spec, symbols, debug = True)

    #dace.perf.sdfv_roofline.view(sdfg, roof)

    print("SDFGRooflineOptimizer")
    optimizer = dace.perf.optimizer.SDFGRooflineOptimizer(sdfg, roof, inplace = False)
    optimizer.optimize()

    #print("SDFGOptimizer")
    #optimizer = dace.transformation.optimizer.SDFGOptimizer(sdfg)
    #optimizer.optimize()
