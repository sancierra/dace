import dace
import numpy as np

from dace.transformation.heterogeneous.pipeline import expand_reduce, expand_maps, fusion

N = dace.symbol('N')
M = dace.symbol('M')


N.set(300); M.set(300)

@dace.program
def TEST(A: dace.float32[M,N]):
    return dace.reduce(lambda a, b: max(a,b), A, axis=1, identity = 0)

A = np.random.rand(M.get(), N.get()).astype(np.float32)

if __name__ == '__main__':
    sdfg.apply_gpu_transformations()
    A = np.random.rand(H.get(), B.get(), SN.get(), SM.get()).astype(np.float32)
    #csdfg = sdfg.compile_directly()
    #result_base = csdfg(X_in=A, H=H, B=B, SN=SN, SM=SM)

    pipeline.expand_reduce(sdfg, sdfg.nodes()[0])
    sdfg.expand_library_nodes()

    for node in sdfg.nodes()[0].nodes():
        if isinstance(node, dace.sdfg.nodes.NestedSDFG):
            for state in node.sdfg.nodes():
                for snode in state.nodes():
                    for e in state.out_edges(snode):
                        e.data.wcr_conflict = False


    #csdfg2 = sdfg.compile_directly()
    #result_1 = csdfg2(X_in = A, H=H, B=B, SN=SN, SM=SM)

    print(np.allclose(result_base, result_1))

    pipeline.expand_maps(sdfg, sdfg.nodes()[0])
    #csdfg3 = sdfg.compile_directly()
    #result_2 = csdfg3(X_in = A, H=H, B=B, SN=SN, SM=SM)
    sdfg.view()
