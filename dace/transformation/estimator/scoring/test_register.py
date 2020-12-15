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
        
            out = in1 * in2 + 42 


@dace.program 
def reg_map(A: dace.float32[N, N], B: dace.float32[N, N], C: dace.float32[N, N], D: dace.float32[N, N]):
    for i in dace.map[0:N]:
        tmp = np.ndarray([N], dtype = dace.float32)
        tmp2 = np.ndarray([N], dtype = dace.float32)
        tmp3 = np.ndarray([N], dtype = dace.float32)

        for j in dace.map[0:N]:
            with dace.tasklet:
                in1 << A[i,j] 
                in2 << B[i,j] 
                out >> tmp[j] 

                asdf = in1 * in1 + in2
                out = asdf * in2 + in1 + asdf 
        
        for j in dace.map[0:N]:
            with dace.tasklet:
                in1 << C[i,j]
                in2 << tmp[j]
                out >> tmp2[j]

                asdf1 = in1 + in2 * 5.3 
                asdf2 = asdf1 * in2 + in1 
                out = asdf2 + 3 
        
        for j in dace.map[0:N]:
            with dace.tasklet:

                in1 << C[i,j]
                in2 << tmp[j]
                out >> tmp3[j]

                asdf1 = in1 + in2 * 3.3 
                asdf2 = asdf1 * in2 + in1 
                out = asdf2 + 3.9

        for j in dace.map[0:N]:
            with dace.tasklet:
                in1 << tmp2[j]
                in2 << tmp3[j]
                out >> D[i,j]
            
                out = in1 * in2 + 42 

@dace.program 
def reg_nested(A: dace.float32[N], B: dace.float32[N], C: dace.float32[N], D: dace.float32[N]):
    for i in dace.map[0:N]:
        tmp = np.ndarray([1], dtype = dace.float32)
        tmp2 = np.ndarray([1], dtype = dace.float32)
        tmp3 = np.ndarray([1], dtype = dace.float32)

        with dace.tasklet:
            out >> tmp2[0]
            out = 42.0


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
        
            out = in1 * in2 + 42 


def parse(program_name):
    sdfg = program_name.to_sdfg()
    
    entries = []
    for node in sdfg.nodes()[0].nodes():
        if isinstance(node, dace.sdfg.nodes.MapEntry):
            entries.append(node)
            break  
    
    for entry in entries:
        for node in sdfg.nodes()[0].scope_children()[entry]:
            if isinstance(node, dace.sdfg.nodes.AccessNode):
                sdfg.data(node.data).storage = dace.dtypes.StorageType.Register
                print(sdfg.data(node.data).storage)

    return sdfg       
   



def caller(sdfg, graph, subgraph = None):
    subgraph = subgraph if subgraph else SubgraphView(graph, graph.nodes())
    N = 10
    io = ({}, {}, {'N':N})
    scoring_function = RegisterScore(sdfg = sdfg,
                                     graph = graph,
                                     io = io,
                                     subgraph = subgraph)
    scope_children = graph.scope_children()
    outer_entry = next(n for n in subgraph.nodes() if isinstance(n, dace.sdfg.nodes.MapEntry) and n in scope_children[None])
    return_value = scoring_function.estimate_spill(sdfg, graph, outer_entry)
    print(return_value)


def test_easy():
    print("***********************************************")
    sdfg = parse(reg)
    caller(sdfg, sdfg.nodes()[0])
    print("***********************************************")


def test_medium():
    print("***********************************************")
    sdfg = parse(reg_map)
    caller(sdfg, sdfg.nodes()[0])
    print("***********************************************")

def test_nested():
    print("***********************************************")
    sdfg = parse(reg_nested)
    caller(sdfg, sdfg.nodes()[0])
    print("***********************************************")


if __name__ == '__main__':
    test_easy()
    test_medium()
    test_nested()