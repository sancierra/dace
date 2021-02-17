import dace 
import numpy as np 
import torch 

from dace.transformation.subgraph import ReduceExpansion, SubgraphFusion, MultiExpansion
from dace.transformation.dataflow import MapCollapse, RedundantSecondArray, MapExpansion
from dace.transformation.interstate import InlineSDFG
from dace.transformation.estimator.programs.factory import get_args as factory_args
from dace.sdfg.graph import SubgraphView
from dace.sdfg.nodes import MapEntry, MapExit, AccessNode
import dace.libraries.standard as std 
from dace.codegen import compiler

import dace.sdfg.nodes as nodes 
import dace.transformation.helpers as helpers
import dace.dtypes as dtypes 




def get_partition(sdfg, graph):
    subgraph1 = SubgraphView(graph, [])
    subgraph2 = SubgraphView(graph, [])

    map_counter = 0
    for node in dace.sdfg.utils.dfs_topological_sort(graph):
  
        if map_counter < 3:
            subgraph1._subgraph_nodes.append(node)
        else:
            subgraph2._subgraph_nodes.append(node)
        
        if isinstance(node, MapExit):
            map_counter += 1

    return [subgraph1, subgraph2]



def get_sdfg():
    sdfg = dace.sdfg.SDFG.from_file('../../estimator/programs/softmax32.sdfg')
    return sdfg

def get_args():
    io = factory_args('softmax')
    (inputs, outputs, symbols) = io
    print("Symbol sizes:", symbols)
    return {**inputs, **symbols}



def apply_pre_transformations(sdfg, cuda_expand = True, strict = True):
    graph = sdfg.nodes()[0]
    sdfg.apply_transformations_repeated(ReduceExpansion)
    if cuda_expand:
        for node in graph.nodes():
            if isinstance(node, std.nodes.Reduce):
                node.implementation = 'CUDA (block allreduce)'
                node.expand(sdfg, graph)
        
    sdfg.apply_transformations_repeated(InlineSDFG)
    sdfg.apply_transformations(RedundantSecondArray)
        


def fully_fuse(sdfg):
    sdfg.apply_strict_transformations()
    apply_pre_transformations(sdfg)
    graph = sdfg.nodes()[0]
    subgraph = SubgraphView(graph, graph.nodes())
    me = MultiExpansion(subgraph)
    me.apply(sdfg)
    sf = SubgraphFusion(subgraph)
    sf.transient_allocation = dace.dtypes.StorageType.GPU_Shared 
    sf.schedule_innermaps = dace.dtypes.ScheduleType.GPU_ThreadBlock
    sf.apply(sdfg)


def partially_fuse(sdfg):
    sdfg.apply_strict_transformations()
    apply_pre_transformations(sdfg)
    graph = sdfg.nodes()[0]
    s1, s2 = get_partition(sdfg, graph)
    print(s1.nodes())
    print(s2.nodes())
    subgraph = SubgraphView(graph, graph.nodes())
    me = MultiExpansion(subgraph)
    me.apply(sdfg)

    sf = SubgraphFusion(s1)
    sf.transient_allocation = dace.dtypes.StorageType.GPU_Shared 
    sf.schedule_innermaps = dace.dtypes.ScheduleType.GPU_ThreadBlock
    sf.apply(sdfg)

    sf = SubgraphFusion(s2)
    sf.transient_allocation = dace.dtypes.StorageType.GPU_Shared 
    sf.schedule_innermaps = dace.dtypes.ScheduleType.GPU_ThreadBlock
    sf.apply(sdfg)
    

def fully_fuse_block_inner(sdfg):
    # inner collapse 
    apply_pre_transformations(sdfg)
    print("Applied Pre - Transformations")
    sdfg.apply_transformations_repeated(MapCollapse)
    # fully fuse 
    sdfg.save('debug.sdfg')

    graph = sdfg.nodes()[0]
    subgraph = SubgraphView(graph, graph.nodes())
    me = MultiExpansion(subgraph)
    me.apply(sdfg)
    sf = SubgraphFusion(subgraph)
    sf.transient_allocation = dace.dtypes.StorageType.Register 
    sf.schedule_innermaps = dace.dtypes.ScheduleType.GPU_ThreadBlock
    sf.apply(sdfg)
    # change storage
    for node in graph.nodes():
        if isinstance(node, AccessNode) and node.data in ['tmp_sum', 'tmp_max']:
            sdfg.data(node.data).storage = dace.dtypes.StorageType.GPU_Shared

    # now expand and collapse, assign thread block schedule to inner one 
    sdfg.apply_transformations(MapExpansion)
    x_in = graph.source_nodes()[0]
    gpu_x_in = graph.out_edges(x_in)[0].dst 

    for _ in range(2):
        outer = graph.out_edges(gpu_x_in)[0].dst 
        inner = graph.out_edges(outer)[0].dst 
        collapse = MapCollapse(sdfg.sdfg_id,
                               sdfg.nodes().index(graph),
                               {MapCollapse._outer_map_entry: graph.nodes().index(outer),
                                MapCollapse._inner_map_entry: graph.nodes().index(inner)},
                               0)
        collapse.apply(sdfg)

    outer = graph.out_edges(gpu_x_in)[0].dst 
    inner = graph.out_edges(outer)[0].dst 
    inner.map.schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock
    '''
    scope_subgraph = graph.scope_subgraph(inner, include_entry = False, include_exit = False)
    print(scope_subgraph.nodes())
    nsdfg = helpers.nest_state_subgraph(sdfg, graph, scope_subgraph)
    print(nsdfg)
    sdfg.save('nested.sdfg')
    '''


def fully_fuse_register(sdfg):
    apply_pre_transformations(sdfg, cuda_expand = False)
    graph = sdfg.nodes()[0]
    subgraph = SubgraphView(graph, graph.nodes())
    me = MultiExpansion(subgraph)
    me.apply(sdfg)
    sf = SubgraphFusion(subgraph)
    sf.transient_allocation = dace.dtypes.StorageType.Register 
    sf.schedule_innermaps = dace.dtypes.ScheduleType.Sequential
    sf.apply(sdfg)

    sdfg.expand_library_nodes()
    for n in graph.nodes():
        if isinstance(n, nodes.NestedSDFG):
            for ngraph in n.sdfg.nodes():
                for nn in ngraph.nodes():
                    if isinstance(nn, nodes.MapEntry):
                        print(f"Changed node Schedule of {nn}")
                        nn.map.schedule = dtypes.ScheduleType.Sequential
    return sdfg 


def run_cached(sdfg, args):
    binary_filename = compiler.get_binary_name(sdfg.build_folder, sdfg.name)    
    handle = compiler.load_from_file(sdfg, binary_filename)
    result = handle(**args)
    return result


def run(sdfg, args, fusion_handle = None, strict = True):
    if fusion_handle:
        fusion_handle(sdfg)
    else:
        for node in sdfg.nodes()[0].nodes():
            if isinstance(node, std.nodes.Reduce):
                node.implementation = "CUDA (device)"
    if strict:
        sdfg.apply_strict_transformations()
    
    sdfg.save('runnable.sdfg')

    return_value = sdfg(**args)
    return return_value 


def run_and_compare(sdfg, args, fusion_handle):
    # run baseline 
    result1 = run(sdfg, args, None)
    # run other 
    result2 = run(sdfg, args, fusion_handle)

    print(np.linalg.norm(result1))
    print(np.linalg.norm(result2))

def run_torch(args, cuda = True):
    H = args['H']
    B = args['B']
    SN = args['SN']
    SM = args['SM']

    device = torch.device("cuda:0" if cuda else "cpu")
    cpu = torch.device("cpu")
    x_in = torch.from_numpy(args['X_in']).to(device)
    softmax = torch.nn.Softmax(dim = 3)
    result = softmax(x_in)
    norm = np.linalg.norm(result.to(cpu).numpy())
    print(norm)



def test_tiled_reduction(sdfg):
    sdfg.apply_gpu_transformations()
    graph = sdfg.nodes()[0]
    for n in graph.nodes():
        if isinstance(n, std.nodes.Reduce):
            r = ReduceExpansion(0,0,{ReduceExpansion._reduce: graph.nodes().index(n)},0)
            r.tile_size = 4
            print("APPLY")
            r.apply(sdfg)
    
    for n in graph.nodes():
        if isinstance(n, std.nodes.Reduce):
            if n.axes == None:
                n.implementation = 'CUDA (block allreduce)'
            else:
                n.schedule = dace.dtypes.ScheduleType.Sequential 

        if isinstance(n, nodes.EntryNode) and n.label == 'reduce_values':
            n.map.schedule = dace.dtypes.ScheduleType.GPU_ThreadBlock

    '''
    sdfg.expand_library_nodes()
    for n in graph.nodes():
        if isinstance(n, nodes.NestedSDFG):
            for ngraph in n.sdfg.nodes():
                for nn in ngraph.nodes():
                    if isinstance(nn, nodes.MapEntry):
                        print(f"Changed node Schedule of {nn}")
                        nn.map.schedule = dtypes.ScheduleType.Sequential
    '''

def test_tiled_fusion(sdfg):
    sdfg.apply_gpu_transformations()
    graph = sdfg.nodes()[0]
    
    print("---- Reduction Nodes ----")
    for n in graph.nodes():
        if isinstance(n, std.nodes.Reduce):
            r = ReduceExpansion(0,0,{ReduceExpansion._reduce: graph.nodes().index(n)},0)
            r.tile_size = 4
            print(f"Applying ReduceExpansion on {n}")
            r.apply(sdfg)

    for n in graph.nodes():
        if isinstance(n, std.nodes.Reduce):
            print(f"Changing Implementation of Reduction node {n}")
            if n.axes == None:
                n.implementation = 'CUDA (block allreduce)'
            else:
                n.implementation = 'pure'
                n.schedule = dace.dtypes.ScheduleType.Sequential 
    
    sdfg.save('intermediate.sdfg')

    print("---- Tiled SubgraphFusion ----")

    subgraph = SubgraphView(graph, graph.nodes())
    me = MultiExpansion(subgraph)
    me.apply(sdfg)
    sf = SubgraphFusion(subgraph)
    sf.transient_allocation = dace.dtypes.StorageType.GPU_Shared 
    sf.schedule_innermaps = dace.dtypes.ScheduleType.GPU_ThreadBlock
    sf.inner_tile_sizes = (4,1)
    sf.apply(sdfg)

    sdfg.save('tiling_inspect.sdfg')
    

def tiled_handle(sdfg):
    test_tiled_reduction(sdfg)
        
sdfg = get_sdfg()
args = get_args()
sdfg.specialize({'SM': args['SM']})
del args['SM']






#sdfg.apply_gpu_transformations() 
test_tiled_reduction(sdfg)


#run(sdfg, args)
#run(sdfg, args, fusion_handle = fully_fuse)
#run(sdfg, args, fusion_handle = partially_fuse)
#run(sdfg, args, fusion_handle = apply_pre_transformations)
#run(sdfg, args, fusion_handle = fully_fuse_block_inner)
#run(sdfg, args, fusion_handle = fully_fuse_register)

#fully_fuse_block_inner(sdfg)
#rv = run_cached(sdfg, args)
#print(np.linalg.norm(rv))

#rv = run(sdfg, args)
#print(np.linalg.norm(rv))

#run_and_compare(sdfg, args, fusion_handle = fully_fuse)
#run_and_compare(sdfg, args, fusion_handle = fully_fuse_block_inner)

#run_torch(args, cuda = True)
