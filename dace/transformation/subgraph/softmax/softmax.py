import dace 
import numpy as np 
import torch 

from dace.transformation.subgraph import ReduceExpansion, SubgraphFusion, MultiExpansion
from dace.transformation.dataflow import MapCollapse, RedundantSecondArray
from dace.transformation.estimator.programs.factory import get_args as factory_args
from dace.sdfg.graph import SubgraphView
from dace.sdfg.nodes import MapEntry, MapExit, AccessNode
import dace.libraries.standard as std 



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


def apply_pre_transformations(sdfg, cuda_expand = True, strict = False):
    graph = sdfg.nodes()[0]
    sdfg.apply_transformations_repeated(ReduceExpansion)
    sdfg.save('sdfg_1.sdfg')
    if cuda_expand:
        for node in graph.nodes():
            if isinstance(node, std.nodes.Reduce):
                node.implementation = 'CUDA (block allreduce)'
        
        sdfg.expand_library_nodes()
        '''
        if strict:
            sdfg.apply_strict_transformations()
        '''
        sdfg.save('sdfg_2.sdfg')


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
    

def fully_fuse_register(sdfg):
    # inner collapse 
    apply_pre_transformations(sdfg)
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
    sdfg.apply_transformations(RedundantSecondArray)
    # change storage
    for node in graph.nodes():
        if isinstance(node, AccessNode) and node.data in ['tmp_sum', 'tmp_max']:
            sdfg.data(node.data).storage = dace.dtypes.StorageType.GPU_Shared



def run(sdfg, args, fusion_handle = None, strict = True):
    if fusion_handle:
        fusion_handle(sdfg)
    else:
        for node in sdfg.nodes()[0].nodes():
            if isinstance(node, std.nodes.Reduce):
                node.implementation = "CUDA (device)"
    sdfg.save('runnable.sdfg')
    if strict:
        sdfg.apply_strict_transformations()
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


sdfg = get_sdfg()
args = get_args()
sdfg.specialize({'SM': args['SM']})
del args['SM']
sdfg.apply_gpu_transformations() 



#run(sdfg, args)
#run(sdfg, args, fusion_handle = fully_fuse)
#run(sdfg, args, fusion_handle = partially_fuse)
run(sdfg, args, fusion_handle = apply_pre_transformations)
#run_torch(args, cuda = True)
