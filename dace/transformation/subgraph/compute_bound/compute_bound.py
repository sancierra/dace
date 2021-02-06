import dace 
import numpy as np 

from dace.sdfg.graph import SubgraphView 
from dace.transformation.subgraph import SubgraphFusion 


N = dace.symbol('N')
M = dace.symbol('M')


@dace.program 
def compute_bound(A: dace.float32[M,N], B: dace.float32[M,N]):
    tmp1 = np.ndarray((M,N), dtype = np.float32)
    tmp2 = np.ndarray((M,N), dtype = np.float32)    
    ret = np.ndarray((M,N), dtype = np.float32)

    for i, j in dace.map[0:M, 0:N]:
        tmp_var = np.float32(0.0)
        for k in range(100):
            with dace.tasklet:
                in1 << A[i,j]
                in2 << tmp_var
                out1 >> tmp_var

                out1 = out1 + sin(cos(cos(sin(in1))))

        tmp1[i,j] = tmp_var 

    for i,j in dace.map[0:M, 0:N]:
        tmp_var = np.float32(0.0)
        for k in range(100):
            with dace.tasklet:
                in1 << B[i,j]
                in2 << tmp_var
                out1 >> tmp_var

                out1 += sqrt(cos(sqrt(sin(sin(in1)))))

        tmp2[i,j] = tmp_var 

    for i,j in dace.map[0:M, 0:N]:
        with dace.tasklet:
            in1 << tmp1[i,j]
            in2 << tmp2[i,j]
            out1 >> ret[i,j]

            out1 = sin(cos(in1) + cos(in2)) / sin(cos(in1) + sin(in2))

    return ret 


def get_sdfg():
    return compute_bound.to_sdfg() 


def get_args():
    args = {}
    N = 512
    M = 512

    args['A'] = np.ndarray((M,N), dtype = np.float32)
    args['B'] = np.ndarray((M,N), dtype = np.float32)

    args.update({'N':N, 'M':M})
    return args 


def run(sdfg, args):
    result = sdfg(**args)
    return result 

def fuse(sdfg):
    graph = sdfg.nodes()[0]
    subgraph = SubgraphView(graph, graph.nodes())
    fusion = SubgraphFusion(subgraph)
    fusion.apply(sdfg)

def test_fusion(sdfg, args):
    # run 1 
    result1 = run(sdfg, args)
    # run 2 
    fuse(sdfg)
    result2 = run(sdfg, args)
    print(np.linalg.norm(result1))
    print(np.linalg.norm(result2))


sdfg = get_sdfg()
args = get_args() 
test_fusion(sdfg, args)
#sdfg.apply_gpu_transformations()
#fuse(sdfg)
#sdfg.validate()
#sdfg.save('compute_bound2.sdfg')
#run(sdfg, args)