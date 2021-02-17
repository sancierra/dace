import numpy as np 
from dace.transformation.subgraph import ReduceExpansion
import dace.libraries.standard as std 
import dace

N = dace.symbol('N')
M = dace.symbol('M')

N.set(30)
M.set(40)

@dace.program 
def bugged(input: dace.float32[M,N]):
    output = dace.reduce(lambda a, b: a + b, input, identity = 0, axis=1)
    return output 


sdfg = bugged.to_sdfg() 
sdfg.apply_strict_transformations()
sdfg.apply_gpu_transformations()
sdfg.specialize({'N':N.get()})
sdfg.save('this.sdfg')
graph = sdfg.nodes()[0]
for n in graph.nodes():
    if isinstance(n, std.nodes.Reduce):
        r = ReduceExpansion(0,0,{ReduceExpansion._reduce: graph.nodes().index(n)},0)
        r.tile_size = 4
        r.apply(sdfg)

for n in graph.nodes():
    if isinstance(n, std.nodes.Reduce):
        print(f"Changing Implementation of Reduction node {n}")
        if n.axes == None:
            n.implementation = 'CUDA (block allreduce)'
        else:
            n.implementation = 'pure'
            n.schedule = dace.dtypes.ScheduleType.Sequential 


input = np.ndarray((M.get(), N.get()), dtype = np.float32)
sdfg.save('gpu_codegen_state_bug.sdfg')
sdfg(input = input, M=M, N=N)
