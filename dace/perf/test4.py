import dace
import numpy as np

import dace.perf.roofline as roofline


N = dace.symbol('N')

# method 1
@dace.program
def TEST(A: dace.float64[N], B:dace.float64[N], C:dace.float64[N]):
    # Transient variables
    #dace.reduce(lambda a,b: a+b, A, B, axis = 2)
    #tmp = np.ndarray([N], dtype = np.float64)
    for i in dace.map[0:N]:
        with dace.tasklet:
            input << A[i]
            input2 << B[i]
            out >> A[i]
            out2 >> B[i]

            out = input + 1
            out2 = input2 + 1

    for i in dace.map[0:N]:
        with dace.tasklet:
            input << A[i]
            input2 << C[i]
            out >> A[i]
            out2 >> C[i]

            out = input + 3
            out2 = input2 + 2

if __name__ == '__main__':
    N.set(4)



    peak_bandwidth = 1.867 * 64 * 2 / 8
    peak_performance = 2.7 * 4 * 2 * 4

    symbols = {N: 30}
    spec = roofline.PerformanceSpec(peak_bandwidth, peak_performance, dace.float64)
    roof = roofline.Roofline(spec, symbols, debug = True)

    sdfg = TEST.to_sdfg()
    #sdfg.view()
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
