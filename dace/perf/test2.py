import dace
import numpy as np

import dace.perf.roofline as roofline


N = dace.symbol('N')
K = dace.symbol('K')
M = dace.symbol('M')

# method 1
@dace.program
def TEST(A: dace.float64[N], B:dace.float64[N], C:dace.float64[N]):
        # Transient variable
    B[:] = A[:]+1
    B[:] = B[:] + A[:]

if __name__ == '__main__':
    #M.set(300)
    N.set(30)
    #K.set(300)



    peak_bandwidth = 1.867 * 64 * 2 / 8
    peak_performance = 2.7 * 4 * 2 * 4

    symbols = {N: 30}
    spec = roofline.PerformanceSpec(peak_bandwidth, peak_performance, dace.float64)
    roof = roofline.Roofline(spec, symbols, debug = True)

    sdfg = TEST.to_sdfg()
    dace.perf.sdfv_roofline.view(sdfg, roof)

    #print("SDFGRooflineOptimizer")
    #optimizer = dace.perf.optimizer.SDFGRooflineOptimizer(sdfg, roof, inplace = False)
    #optimizer.optimize()


    A = dace.ndarray([30], dtype = np.float64)
    B = dace.ndarray([30], dtype = np.float64)
    A[:] = 0
    B[:] = 0
    TEST(A=A, B=B, N=N)
    print(A)
    print(B)


    #print("SDFGOptimizer")
    #optimizer = dace.transformation.optimizer.SDFGOptimizer(sdfg)
    #optimizer.optimize()
