import dace
import numpy as np

from dace.transformation.heterogeneous.pipeline import expand_reduce, expand_maps, fusion

N = dace.symbol('N')
M = dace.symbol('M')


N.set(300); M.set(300)


@dace.program
def TEST(A: dace.float32[M,N]):
    return dace.reduce(lambda a, b: max(a,b), A, axis=1, identity = 0)


@ dace.program
def TEST2(A: dace.float32[M,N]):
    tmp_out = np.ndarray([M], dace.float32)
    for i in dace.map[0:M]:
        tmp_out[i] = dace.reduce(lambda a,b: max(a,b), A[i,:], identity=0)
    return tmp_out



A = np.random.rand(M.get(), N.get()).astype(np.float32)



if __name__ == '__main__':
    sdfg1 = TEST.to_sdfg()
    sdfg2 = TEST2.to_sdfg()
    sdfg1.apply_gpu_transformations()
    sdfg2.apply_gpu_transformations()

    return1 = sdfg1.compile()(A=A, N=N, M=M)
    return2 = sdfg2.compile()(A=A, N=N, M=M)

    print(np.linalg.norm(return1))
    print(np.linalg.norm(return2))

    ###################
    sdfg1.expand_library_nodes()
    sdfg2.expand_library_nodes()

    for sdfg in [sdfg1, sdfg2]:
        for node in sdfg.nodes()[0].nodes():
            if isinstance(node, dace.sdfg.nodes.NestedSDFG):
                for state in node.sdfg.nodes():
                    for snode in state.nodes():
                        for e in state.out_edges(snode):
                            e.data.wcr_conflict = False
                            if isinstance(snode, dace.sdfg.nodes.MapEntry):
                                snode.schedule = dace.dtypes.ScheduleType.Sequential

    return1 = sdfg1.compile()(A=A, N=N, M=M)
    return2 = sdfg2.compile()(A=A, N=N, M=M)
