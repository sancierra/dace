import dace
from dace.transformation.subgraph.gemm import NestedMapFusion 
from dace.transformation.subgraph.gemm import NestOut

import numpy as np

M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')
L = dace.symbol('L')

M.set(20)
N.set(21)
K.set(22)
L.set(23)

@dace.program
def GEMM(A:dace.float32[M,N], B:dace.float32[N,K], C:dace.float32[K,L]):
    return A@B@C  

def test():
    sdfg = GEMM.to_sdfg()
    sdfg.save('gemm.sdfg')
    sdfg.expand_library_nodes()
    sdfg.apply_transformations(NestOut)
    sdfg.apply_strict_transformations()
    sdfg.save('after.sdfg')

def get_args():
    args = {'A': np.random.rand(M.get(), N.get()).astype(np.float32),
            'B': np.random.rand(N.get(), K.get()).astype(np.float32),
            'C': np.random.rand(K.get(), L.get()).astype(np.float32)
            }
    return args 

def run():
    sdfg = GEMM.to_sdfg()
    sdfg.expand_library_nodes()
    args = get_args()
    r1 = sdfg(M=M, N=N, K=K, L=L, **args)

    sdfg = GEMM.to_sdfg()
    sdfg.expand_library_nodes()
    sdfg.apply_transformations(NestOut)
    sdfg.apply_strict_transformations()
    r2 = sdfg(M=M, N=N, K=K, L=L, **args)

    print(np.linalg.norm(r1))
    print(np.linalg.norm(r2))



run()