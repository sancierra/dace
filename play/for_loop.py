import dace
import numpy as np


N = dace.symbol('N')
W = dace.symbol('W')


@dace.program
def DoubleFor(A: dace.float64[N], B: dace.float64[N],
             C: dace.float64[N]):
        # Transient variable
    for i in dace.map[0:10]:
        for j in dace.map[0:10]:
             
             C[i] += A[j] + B[i] 


if __name__ == '__main__':
    #GEMM1.compile(strict = True)
    N.set(300)
    W.set(10)
    A = np.random.rand(N.get())
    B = np.random.rand(N.get())
    R = np.ndarray(shape=[N.get()])

    DoubleFor(A=A,B=B,C=R)
