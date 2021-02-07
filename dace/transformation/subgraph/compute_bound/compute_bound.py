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
        for k in range(200):
            with dace.tasklet:
                in1 << A[i,j]
                in2 << tmp_var
                out1 >> tmp_var

                out1 = in2 + sin(in1)

        tmp1[i,j] = tmp_var 

    for i,j in dace.map[0:M, 0:N]:
        tmp_var = np.float32(0.0)
        for k in range(200):
            with dace.tasklet:
                in1 << B[i,j]
                in2 << tmp_var
                out1 >> tmp_var

                out1 = in2 + cos(in1)

        tmp2[i,j] = tmp_var 

    for i,j in dace.map[0:M, 0:N]:
        '''
        with dace.tasklet:
            in1 << tmp1[i,j]
            in2 << tmp2[i,j]
            out1 >> ret[i,j]

            tt1 = sin(cos(in1) + cos(in2)) / (2+sin(cos(in1) + sin(in2)))
            tt2 = (cos(tt1) + sin(tt1)) / 8
            tt3 = cos(tt1) + sin(tt2) 
            out1 = tt3
        '''
        tmp_var = tmp1[i,j] + tmp2[i,j] 
        for k in range(100):
            with dace.tasklet:
                in1 << tmp_var 
                out1 >> tmp_var
                
                out1 = in1 + cos(in1) - sin(in1)  

        ret[i,j] = tmp_var

    return ret 


def get_sdfg():
    return compute_bound.to_sdfg() 


def get_args():
    args = {}
    N = 512
    M = 512

    args['A'] = np.random.rand(M,N).astype(np.float32)
    args['B'] = np.random.rand(M,N).astype(np.float32)

    args.update({'N':N, 'M':M})
    return args 


def run(sdfg, args):
    result = sdfg(**args)
    print(result[0][0:50])
    return result 

def fuse(sdfg):
    graph = sdfg.nodes()[0]
    subgraph = SubgraphView(graph, graph.nodes())
    fusion = SubgraphFusion(subgraph)
    fusion.apply(sdfg)

def test_fusion(sdfg, args, gpu = False):
    if gpu:
        sdfg.apply_gpu_transformations()
    # run 1 
    result1 = run(sdfg, args)
    # run 2 
    fuse(sdfg)
    result2 = run(sdfg, args)
    print(np.linalg.norm(result1))
    print(np.linalg.norm(result2))


sdfg = get_sdfg()
args = get_args() 
test_fusion(sdfg, args, gpu = True)
#sdfg.apply_gpu_transformations()
#fuse(sdfg)
#sdfg.validate()
#sdfg.save('compute_bound2.sdfg')
#run(sdfg, args)
