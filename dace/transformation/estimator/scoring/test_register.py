import dace  
from dace.transformation.estimator import RegisterScore 
from dace.transformation.interstate import StateFusion
from dace.sdfg.graph import SubgraphView
import numpy as np 



N = dace.symbol('N')

@dace.program 
def reg(A: dace.float32[N], B: dace.float32[N], C: dace.float32[N], D: dace.float32[N]):
    for i in dace.map[0:N]:
        tmp = np.ndarray([1], dtype = dace.float32)
        tmp2 = np.ndarray([1], dtype = dace.float32)
        tmp3 = np.ndarray([1], dtype = dace.float32)

        with dace.tasklet:
            in1 << A[i] 
            in2 << B[i] 
            out >> tmp

            asdf = in1 * in1 + in2
            out = asdf * in2 + in1 + asdf 
        
        with dace.tasklet:
            in1 << C[i]
            in2 << tmp 
            out >> tmp2

            asdf1 = in1 + in2 * 5.3 
            asdf2 = asdf1 * in2 + in1 
            out = asdf2 + 3 

        with dace.tasklet:

            in1 << C[i]
            in2 << tmp 
            out >> tmp3

            asdf1 = in1 + in2 * 3.3 
            asdf2 = asdf1 * in2 + in1 
            out = asdf2 + 3.9

        with dace.tasklet:
            in1 << tmp2
            in2 << tmp3
            out >> D[i]

            out = tmp1 * tmp2 + 42 

def tester1():
    print("--- Test ---")
    sdfg = reg.to_sdfg()
    nsdfg = None 
    for node in sdfg.nodes()[0].nodes():
        if isinstance(node, dace.sdfg.nodes.NestedSDFG):
            nsdfg = node  
    
    
    graphs = [g for g in nsdfg.sdfg.nodes()]
    
    fusion = StateFusion(nsdfg.sdfg.sdfg_id, -1, {StateFusion.first_state: 0, StateFusion.second_state: 1}, 0)
    fusion.apply(nsdfg.sdfg)
    sdfg.apply_strict_transformations()

    entry = None 
    for node in sdfg.nodes()[0].nodes():
        if isinstance(node, dace.sdfg.nodes.MapEntry):
            entry = node 
            break  
    

    for node in sdfg.nodes()[0].scope_children()[entry]:
        if isinstance(node, dace.sdfg.nodes.AccessNode):
            sdfg.data(node.data).storage = dace.dtypes.StorageType.Register
            print(sdfg.data(node.data).storage)

    sdfg.save('register_test.sdfg')            
    
def tester2():
    print("--- Test ---")
    sdfg = reg.to_sdfg()
    nsdfg = None 
    for node in sdfg.nodes()[0].nodes():
        if isinstance(node, dace.sdfg.nodes.NestedSDFG):
            nsdfg = node  
    
    for graph in nsdfg.sdfg.nodes():
        for node in graph.nodes():
            if isinstance(node, dace.sdfg.nodes.AccessNode):
                if nsdfg.sdfg.data(node.data).transient:
                    nsdfg.sdfg.data(node.data).storage = dace.dtypes.StorageType.Register
    sdfg.save('register_test2.sdfg')




def test_register(sdfg, graph, subgraph = None):
    subgraph = subgraph if subgraph else SubgraphView(graph, graph.nodes())
    N = 10
    io = ({}, {}, {'N':N})
    scoring_function = RegisterScore(sdfg = sdfg,
                                     graph = graph,
                                     io = io,
                                     subgraph = subgraph)
    
    outer_entry = next(n for n in subgraph.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry))
    return_value = scoring_function.estimate_spill(sdfg, graph, outer_entry)
    print(return_value)

if __name__ == '__main__':
    sdfg = dace.sdfg.SDFG.from_file('register_test.sdfg')
    graph = sdfg.nodes()[0]
    test_register(sdfg, graph)