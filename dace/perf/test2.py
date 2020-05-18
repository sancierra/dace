import dace
import numpy as np

import dace.perf.roofline as roofline


N = dace.symbol('N')
K = dace.symbol('K')
M = dace.symbol('M')

# method 1
@dace.program
def TEST(A: dace.float64[M,N,K], B:dace.float64[M,N]):
        # Transient variable

    B[:] = dace.reduce(lambda a,b: a+b, A, axis = 2)

if __name__ == '__main__':
    M.set(2)
    N.set(4)
    K.set(6)


    '''
    peak_bandwidth = 1.867 * 64 * 2 / 8
    peak_performance = 2.7 * 4 * 2 * 4

    symbols = {N: 30}
    spec = roofline.PerformanceSpec(peak_bandwidth, peak_performance, dace.float64)
    roof = roofline.Roofline(spec, symbols, debug = True)

    #sdfg = TEST.to_sdfg()
    #dace.perf.sdfv_roofline.view(sdfg, roof)

    #print("SDFGRooflineOptimizer")
    #optimizer = dace.perf.optimizer.SDFGRooflineOptimizer(sdfg, roof, inplace = False)
    #optimizer.optimize()

    '''


    A = dace.ndarray([M.get(), N.get(), K.get()], dtype = np.float64)
    B = dace.ndarray([M.get(), N.get()], dtype = np.float64)
    A[:] = 0
    A[1,2,:] = 2
    print(A)
    B[:] = 0
    TEST(A=A, B=B, M=M, N=N, K=K)
    print(B)
