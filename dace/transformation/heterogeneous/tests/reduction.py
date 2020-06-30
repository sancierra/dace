import dace
import numpy as np

from dace.transformation.heterogeneous.pipeline import expand_reduce, expand_maps, fusion
from dace.transformation.heterogeneous.reduce.cuda_block import CUDABlockAllReduce

from dace.libraries.standard.nodes.reduce import Reduce
N = dace.symbol('N')
M = dace.symbol('M')


N.set(300); M.set(300)


@dace.program
def TEST(A: dace.float32[M,N]):
    return dace.reduce(lambda a, b: max(a,b), A, axis=1, identity = 0)


@dace.program
def TEST2(A: dace.float32[M,N]):
    tmp_out = np.ndarray([M], dace.float32)
    for i in dace.map[0:M]:
        tmp_out[i] = dace.reduce(lambda a,b: max(a,b), A[i,:], identity=0)
    return tmp_out


A = np.random.rand(M.get(), N.get()).astype(np.float32)


if __name__ == '__main__':
    sdfg = TEST.to_sdfg()
    graph = sdfg.nodes()[0]
    for node in graph.nodes():
        if isinstance(node, Reduce):
            reduce_node = node

    reduce_node.implementation = 'CUDA (block)'
    # first, check whether we can apply:
    check = CUDABlockAllReduce.can_be_applied(graph,
                                     {CUDABlockAllReduce._reduce: graph.nodes().index(reduce_node)},
                                      0, sdfg)
    print(check)
    sdfg_id = 0
    state_id = 0
    subgraph = {CUDABlockAllReduce._reduce: graph.nodes().index(reduce_node)}
    transformation = CUDABlockAllReduce(sdfg_id, state_id, subgraph, 0)
    #sdfg.view()
    transformation.apply(sdfg)
    sdfg.view()

    '''
    ## sdfg2 does not work here
    #return1 = sdfg1.compile()(A=A, N=N, M=M)
    #return2 = sdfg2.compile()(A=A, N=N, M=M)

    #print(np.linalg.norm(return1))
    #print(np.linalg.norm(return2))
    '''
