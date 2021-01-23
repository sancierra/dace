import dace
from dace.transformation.subgraph.gemm import NestedMapFusion 


M = dace.symbol('M')
N = dace.symbol('N')
K = dace.symbol('K')
L = dace.symbol('L')


@dace.program
def GEMM(A:dace.float32[M,N], B:dace.float32[N,K], C:dace.float32[K,L]):
    return A@B@C  


sdfg = GEMM.to_sdfg()
sdfg.expand_library_nodes()
sdfg.apply_transformations_repeated(NestedMapFusion)
sdfg.save('gemm.sdfg')