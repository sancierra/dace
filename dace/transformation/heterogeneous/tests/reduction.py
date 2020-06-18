import dace
import numpy as np

N = dace.symbol('N')
M = dace.symbol('M')


N.set(300); M.set(300)

@dace.program
def TEST(A: dace.float32[M,N]):
    return dace.reduce(lambda a, b: max(a,b), A, axis=1, identity = 0)


A = np.random.rand(M.get(), N.get()).astype(np.float32)
if __name__ == '__main__':
    sdfg = TEST.to_sdfg()
    sdfg.apply_gpu_transformations()
    sdfg.view()
    csdfg1 = sdfg.compile_directly()
    result1 = csdfg1(A=A, N=N, M=M)
    print(np.linalg.norm(result1))

    dace.transformation.heterogeneous.pipeline.expand_reduce(sdfg, sdfg.nodes()[0])
    sdfg.view()
    csdfg2 = sdfg.compile_directly()
    result2 = csdfg2(A=A, N=N, M=M)
    print(np.linalg.norm(result2))
