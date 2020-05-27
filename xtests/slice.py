import dace
import numpy as np

N = dace.symbol('N')

@dace.program
def test(A: dace.float64[N], B:dace.float64[N], C:dace.float64[N]):
    '''
    for i in dace.map[0:(N/2)]:
        B[2*i+1] *= 2
        B[(2*i):(2*i+2)] = A[(2*i):(2*i+2)]

        #B[2*i] = A[2*i]
        #B[2*i+1] = A[2*i+1]*2
    '''

    # wcr:
    for i in dace.map[0:(N-1)]:
        #B[i] = A[i]
        B[i+1] = C[i]

if __name__ == '__main__':

    N.set(20)
    A = np.ndarray(shape = N.get())
    C = np.ndarray(shape = N.get())
    for i in range(N.get()):
        A[i] = np.float64(i)
        C[i] = np.float64(-1)

    B = np.ndarray(shape = N.get())
    test(A,B,C)

    print(B)
    sdfg = test.to_sdfg(strict = True)


    for node in sdfg.nodes()[0].nodes():
        if isinstance(node, dace.graph.nodes.EntryNode):
            print("Node", node)
            print("Ranges", node.map.range.ranges)
            print("Tile sizes", node.map.range.tile_sizes)
