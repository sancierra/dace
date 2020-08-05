import dace
import numpy as np

from dace.transformation.heterogeneous.pipeline import expand_reduce, expand_maps, fusion
from dace.transformation.heterogeneous.reduce.cuda_block import CUDABlockAllReduce

from dace.transformation.heterogeneous import ReduceExpansion

from dace.libraries.standard.nodes.reduce import Reduce
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

    # numerical test
    result1 = sdfg(A=A)
    #
    graph = sdfg.nodes()[0]
    for node in graph.nodes():
        if isinstance(node, Reduce):
            reduce_node = node
    sdfg.view()
    print(reduce_node)

    sdfg_id = 0
    state_id = 0
    subgraph = {ReduceExpansion._reduce: graph.nodes().index(reduce_node)}
    # expand first
    transform = ReduceExpansion(sdfg_id, state_id, subgraph, 0)
    transform.cuda_expand = False
    transform.reduce_implementation = 'CUDA (block)'
    transform.apply(sdfg)
    sdfg.view()

    for node in graph.nodes():
        if isinstance(node, Reduce):
            reduce_node = node
    # first, check whether we can apply:
    check = CUDABlockAllReduce.can_be_applied(graph,
                                     {CUDABlockAllReduce._reduce: graph.nodes().index(reduce_node)},
                                      0, sdfg)
    print(check)
    subgraph = {CUDABlockAllReduce._reduce: graph.nodes().index(reduce_node)}
    transform = CUDABlockAllReduce(sdfg_id, state_id, subgraph, 0)
    transform.apply(sdfg)
    sdfg.view()
