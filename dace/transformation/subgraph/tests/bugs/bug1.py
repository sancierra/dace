import dace
import numpy as np
import sys
from copy import deepcopy as dcpy

import dace.dtypes as dtypes

import dace.libraries.standard as stdlib



d32 = dace.float32
d16 = dace.float16
N = dace.symbol('N')
N.set(100)

@dace.program
def TEST(X: d32[N,N]):
    result = np.ndarray([N], dtype = d32)
    for i in dace.map[0:N]:
        tmp = dace.reduce(lambda a,b: a+b, X[i,:], identity = 0)
        result[i] = tmp
    return result
@dace.program
def TEST2(X: d16[N]):
    for i in dace.map[0:N]:
        with dace.tasklet:
            input << X[i]
            output >> X[i]

            output = math.exp(input - 42)



def bug1():
    sdfg = TEST.to_sdfg()
    sdfg.apply_gpu_transformations()
    graph = sdfg.nodes()[0]
    for node in graph.nodes():
        if isinstance(node, dace.sdfg.nodes.Tasklet):
            rm_node = node
        if isinstance(node, stdlib.nodes.Reduce):
            node.implementation = 'pure'
            sdfg.data(graph.out_edges(node)[0].dst.data).storage = dace.dtypes.StorageType.GPU_Shared


    graph.add_edge(u = graph.in_edges(rm_node)[0].src, u_connector = None,
                   v = graph.out_edges(rm_node)[0].dst, v_connector = 'IN_1',
                   memlet = dcpy(graph.out_edges(rm_node)[0].data))

    graph.remove_node(rm_node)

    sdfg.view()
    A = np.random.rand(N.get(), N.get()).astype(np.float32)
    csdfg = sdfg.compile()
    csdfg(A=A, N=N)

def bug2():
    sdfg = TEST2.to_sdfg()
    sdfg.apply_gpu_transformations()
    A = np.random.rand(N.get()).astype(np.float16)
    csdfg = sdfg.compile()
    csdfg(A=A, N=N)

if __name__ == '__main__':
    bug1()
    bug2()
