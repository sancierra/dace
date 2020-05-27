import dace
import numpy as np

import dace.perf.roofline as roofline


N = dace.symbol('N')
K = dace.symbol('K')
M = dace.symbol('M')

# method 1
@dace.program
def TEST(A: dace.float64[N], B:dace.float64[N]):
    tmp = np.ndarray([N], dtype = np.float64)
    for i in dace.map[0:N]:
        with dace.tasklet:
            input << A[i]
            input2 << A[i]
            input3 << A[i]
            out >> tmp[i]
            out = input + input2 + input3

        with dace.tasklet:
            input << A[i]
            input2 << A[i]
            out >> B[i]

            out = input + input2



if __name__ == '__main__':
    M.set(2)
    N.set(4)
    K.set(6)



    peak_bandwidth = 1.867 * 64 * 2 / 8
    peak_performance = 2.7 * 4 * 2 * 4

    symbols = {N: 30}
    spec = roofline.PerformanceSpec(peak_bandwidth, peak_performance, dace.float64)
    roof = roofline.Roofline(spec, symbols, debug = True)

    sdfg = TEST.to_sdfg()
    sdfg.view()
    #sdfg.expand_library_nodes()
    #sdfg.apply_strict_transformations()
    #dace.perf.sdfv_roofline.view(sdfg, roof)

    print("SDFGRooflineOptimizer")
    optimizer = dace.perf.optimizer.SDFGRooflineOptimizer(sdfg, roof, inplace = False)
    optimizer.optimize()


    A = dace.ndarray([M.get(), N.get(), K.get()], dtype = np.float64)
    B = dace.ndarray([1], dtype = np.float64)

    A[:] = 1
    A[1,2,:] = 1
    print(A)
    TEST(A=A, B=B, M=M, N=N, K=K)
    print(B)
