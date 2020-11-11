import dace
import numpy as np

from dace.perf.roofline import Roofline
from dace.perf.specs import *
from dace.perf.optimizer import SDFGRooflineOptimizer

from dace.transformation.subgraph import ReduceExpansion
from dace.transformation.subgraph import SubgraphFusion
from dace.transformation.subgraph import MultiExpansion
from dace.transformation.subgraph import pipeline

from dace.codegen import compiler

import dace.libraries.standard as stdlib

import timeit

from dace.measure.runner import Runner

from dace.sdfg.graph import SubgraphView
import dace.sdfg.nodes as nodes

dace_dtype = dace.float32
H, B, SN, SM = (dace.symbol(s) for s in ('H', 'B', 'SN', 'SM'))


@dace.program
def softmax(X_in: dace_dtype[H, B, SN, 512]):
    tmp_max = dace.reduce(lambda a, b: max(a, b), X_in, axis=3, identity = 0)
    #TEST[:] = tmp_max
    tmp_out = np.ndarray([H, B, SN, 512], dtype=dace_dtype)
    out = np.ndarray([H, B, SN, 512], dtype=dace_dtype)

    # No broadcasting rules
    for i, j, k, l in dace.map[0:H, 0:B, 0:SN, 0:512]:
        with dace.tasklet:
            inp << X_in[i, j, k, l]
            mx << tmp_max[i, j, k]
            o >> tmp_out[i, j, k, l]
            o = math.exp(inp - mx)
    #tmp_out = np.exp(X_in - tmp_max)

    tmp_sum = dace.reduce(lambda a, b: a + b, tmp_out, identity=0, axis=3)
    for i, j, k, l in dace.map[0:H, 0:B, 0:SN, 0:512]:
        with dace.tasklet:
            inp << tmp_out[i, j, k, l]
            sm << tmp_sum[i, j, k]
            o >> out[i, j, k, l]
            o = inp / sm

    return out


H.set(16); B.set(8); SN.set(512); SM.set(512)

def test_graph():
    sdfg = softmax.to_sdfg()
    ################ first, expand the reduce node
    print(sdfg.nodes()[0])
    sdfg.view()
    sdfg.apply_gpu_transformations()
    print(sdfg.nodes()[0])
    sdfg.view()
    #return
    pipeline.expand_reduce(sdfg, sdfg.nodes()[0])
    sdfg.view()
    return
    ############### second, do MultiExpansion
    pipeline.expand_maps(sdfg, sdfg.nodes()[0])
    sdfg.view()

    ############ third, do MapFusion
    pipeline.fusion(sdfg, sdfg.nodes()[0])

    sdfg.apply_strict_transformations()
    sdfg.view()

    sdfg.validate()


def test_result(debug = False):
    sdfg = softmax.to_sdfg()
    debugger = Runner(measure_mode = ['median', 'avg', 'std'],
                      view_roofline = True)

    debugger.go(sdfg, sdfg.nodes()[0], None, H, B, SN, SM,
                performance_spec = dace.perf.specs.PERF_CPU_CRAPBOOK,
                output=[])

    #############

def load_old_configuration(sdfg):
    # loads old configuration
    binary_filename = compiler.get_binary_name(sdfg.build_folder, sdfg.name)
    return compiler.load_from_file(sdfg, binary_filename)



def get_partition(sdfg, graph):
    subgraph1 = SubgraphView(graph, [])
    subgraph2 = SubgraphView(graph, [])

    cnt1 = 0
    for node in dace.sdfg.utils.dfs_topological_sort(graph):
        if isinstance(node, stdlib.nodes.reduce.Reduce):
            if cnt1 < 2:
                subgraph1._subgraph_nodes.append(node)
                cnt1 += 1
            else:
                subgraph2._subgraph_nodes.append(node)

        if isinstance(node, nodes.MapEntry):
            if cnt1 < 2:
                subgraph1._subgraph_nodes.append(node)
                cnt1 += 1
            else:
                subgraph2._subgraph_nodes.append(node)

    return [subgraph1, subgraph2]


def test_allfuse():

    sdfg = softmax.to_sdfg()
    sdfg.apply_gpu_transformations()
    graph = sdfg.nodes()[0]
   
    A = np.random.rand(H.get(), B.get(), SN.get(), SM.get()).astype(np.float32)
    
    for node in graph.nodes():
        if isinstance(node, stdlib.nodes.reduce.Reduce):
            node.implementation = 'CUDA (device)'
    sdfg._name = 'baseline'
    csdfg = sdfg.compile_directly()
    result_base = csdfg(X_in = A, H=H, B=B, SN=SN, SM=SM)
    
    for node in graph.nodes():
        if isinstance(node, stdlib.nodes.reduce.Reduce):
            node.implementation = 'pure'
    ####################################################
    pipeline.expand_reduce(sdfg, graph, cuda_expand = False)
     
    ## Manually fix codegen bug ## 
    sdfg.expand_library_nodes()
    '''
    for node in sdfg.nodes()[0].nodes():
        if isinstance(node, dace.sdfg.nodes.NestedSDFG):
            for state in node.sdfg.nodes():
                for snode in state.nodes():
                    for e in state.out_edges(snode):
                        e.data.wcr_conflict = False
                    if isinstance(snode, dace.sdfg.nodes.MapEntry):
                        snode.schedule = dace.dtypes.ScheduleType.Sequential
    ##
    '''
    sdfg._name = 'reduce'
    csdfg = sdfg.compile_directly()
    result1 = csdfg(X_in = A, H=H, B=B, SN=SN, SM=SM)
    print(np.linalg.norm(result_base))
    print(np.linalg.norm(result1))
    ######################################################
    pipeline.expand_maps(sdfg, graph)

    sdfg._name = 'expansion'
    csdfg = sdfg.compile_directly()
    result2 = csdfg(X_in = A, H=H, B=B, SN=SN, SM=SM)

    ######################################################
    

    sdfg = softmax.to_sdfg()
    sdfg.apply_gpu_transformations()
    graph = sdfg.nodes()[0]
    pipeline.expand_reduce(sdfg, graph, cuda_expand = True, reduce_implementation = 'CUDA (block allreduce)')
    pipeline.expand_maps(sdfg, graph)
    pipeline.fusion(sdfg, graph, transient_allocation = dace.dtypes.StorageType.GPU_Shared, schedule_innermaps = dace.dtypes.ScheduleType.GPU_ThreadBlock)
    
    sdfg.apply_strict_transformations()
    
    sdfg._name = 'fusion'
    csdfg = sdfg.compile_directly()
    result3 = csdfg(X_in = A, H=H, B=B, SN=SN, SM=SM)
    #csdfg = load_old_configuration(sdfg)
    #result3 = csdfg(X_in = A, H=H, B=B, SN=SN, SM=SM)



    ######################################################
    print("Evaluation")
    print("Norms")
    print(np.linalg.norm(result_base))
    print(np.linalg.norm(result1))
    print(np.linalg.norm(result2))
    print(np.linalg.norm(result3))
    #print(result_base)
    #print(result3)
    assert np.allclose(result_base, result1)
    assert np.allclose(result_base, result2)
    assert np.allclose(result_base, result3)

def test_partialfuse():
    
    sdfg = softmax.to_sdfg()
    sdfg.apply_gpu_transformations()
    graph = sdfg.nodes()[0]
    subgraph = get_partition(sdfg, graph)
    A = np.random.rand(H.get(), B.get(), SN.get(), SM.get()).astype(np.float32)
    #TEST = np.random.rand(H.get(), B.get(), SN.get()).astype(np.float32)

    sdfg._name = 'baseline'
    csdfg = sdfg.compile_directly()
    result_base = csdfg(X_in = A, H=H, B=B, SN=SN, SM=SM)


    ####################################################
    pipeline.expand_reduce(sdfg, graph, subgraph, cuda_expand = False, )
    
    ## Manually fix codegen bug ## 
    sdfg.expand_library_nodes()
    sdfg.view()
    for node in sdfg.nodes()[0].nodes():
        if isinstance(node, dace.sdfg.nodes.NestedSDFG):
            for state in node.sdfg.nodes():
                for snode in state.nodes():
                    for e in state.out_edges(snode):
                        e.data.wcr_conflict = False
                    if isinstance(snode, dace.sdfg.nodes.MapEntry):
                        snode.schedule = dace.dtypes.ScheduleType.Sequential
    ##
    sdfg._name = 'reduce'
    csdfg = sdfg.compile_directly()
    result1 = csdfg(X_in = A, H=H, B=B, SN=SN, SM=SM)
    #TEST_ref = TEST.copy()
    ######################################################
    pipeline.expand_maps(sdfg, graph, subgraph)
    sdfg.view()

    sdfg._name = 'expansion'
    csdfg = sdfg.compile_directly()
    result2 = csdfg(X_in = A, H=H, B=B, SN=SN, SM=SM)
    #TEST_ref2 = TEST.copy()
    ######################################################
    

    sdfg = softmax.to_sdfg()
    sdfg.apply_gpu_transformations()
    graph = sdfg.nodes()[0]
    subgraph = get_partition(sdfg, graph)
    pipeline.expand_reduce(sdfg, graph, subgraph, cuda_expand = True, reduce_implementation = 'CUDA (block)' )
    pipeline.expand_maps(sdfg, graph, subgraph)
    pipeline.fusion(sdfg, graph, subgraph)
    
    sdfg.apply_strict_transformations()
    
    sdfg._name = 'fusion'
    csdfg = sdfg.compile_directly()
    result3 = csdfg(X_in = A, H=H, B=B, SN=SN, SM=SM)
    #csdfg = load_old_configuration(sdfg)
    #result3 = csdfg(X_in = A, H=H, B=B, SN=SN, SM=SM)



    ######################################################
    print("Evaluation")
    print("Norms")
    print(np.linalg.norm(result_base))
    print(np.linalg.norm(result1))
    print(np.linalg.norm(result2))
    print(np.linalg.norm(result3))
    #print("TEST")
    #print(np.linalg.norm(TEST_ref))
    #print(np.linalg.norm(TEST_ref2))
    #print(np.linalg.norm(TEST))
    #print(result_base)
    #print(result3)
    assert np.allclose(result_base, result1)
    assert np.allclose(result_base, result2)
    assert np.allclose(result_base, result3)


if __name__ == "__main__":
    test_allfuse()
    
    
