import dace
from dace.sdfg import SDFG
import dace.sdfg

from dace.transformation.subgraph.pipeline import fusion, expand_maps, expand_reduce
from dace.transformation.interstate import StateFusion, NestSDFG
from dace.transformation.helpers import nest_state_subgraph

import dace.transformation.subgraph as subgraph

from dace.measure import Runner

import itertools
import numpy as np
import copy

'''
vadv_unfused = SDFG.from_file('vadv-input-fixed.sdfg')
vadv_unfused.apply_transformations_repeated(StateFusion)
for node in vadv_unfused.nodes()[0].nodes():
    if isinstance(node, dace.sdfg.nodes.MapEntry):
        node.map.schedule = dace.dtypes.ScheduleType.Default
vadv_fused_partial = SDFG.from_file('vadv-2part.sdfg')
vadv_fused_full = SDFG.from_file('vadv-fused.sdfg')
print(vadv_unfused.symbols)
'''

def view_all():
    vadv_unfused.view()
    vadv_fused_partial.view()
    vadv_fused_full.view()


def test_matching():
    graph = vadv_unfused.nodes()[0]
    # should yield True
    subgraph = dace.sdfg.graph.SubgraphView(graph, [node for node in graph.nodes()])
    expansion = subgraph.MultiExpansion()
    print("True  ==", subgraph.MultiExpansion.match(vadv_unfused,subgraph))
    fusion = subgraph.SubgraphFusion()
    print("True  ==", subgraph.SubgraphFusion.match(vadv_unfused,subgraph))

def test_fuse_all(view = True):
    if view:
        vadv_unfused.view()
    graph = vadv_unfused.nodes()[0]
    subgraph = dace.sdfg.graph.SubgraphView(graph, [node for node in graph.nodes()])
    fusion(vadv_unfused, graph)
    if view:
        vadv_unfused.view()

def test_fuse_partial(view = False):
    if view:
        vadv_unfused.view()
    set1 = set()
    set2 = set()
    graph = vadv_unfused.nodes()[0]
    for node in graph.nodes():
        for edge in itertools.chain(graph.out_edges(node), graph.in_edges(node)):
            if isinstance(edge.dst, (dace.sdfg.nodes.MapEntry, dace.sdfg.nodes.MapExit)):
                if '0' in edge.dst.map.label or '1' in edge.dst.map.label or '2' in edge.dst.map.label:
                    set1.add(node)
                if '3' in edge.dst.map.label or '4' in edge.dst.map.label:
                    set2.add(node)
            if isinstance(edge.src, (dace.sdfg.nodes.MapEntry, dace.sdfg.nodes.MapExit)):
                if '0' in edge.src.map.label or '1' in edge.src.map.label or '2' in edge.src.map.label:
                    set1.add(node)
                if '3' in edge.src.map.label or '4' in edge.src.map.label:
                    set2.add(node)
            if isinstance(node, (dace.sdfg.nodes.MapEntry, dace.sdfg.nodes.MapExit)):
                if '0' in node.map.label or '1' in node.map.label or '2' in node.map.label:
                    set1.add(node)
                if '3' in node.map.label or '4' in node.map.label:
                    set2.add(node)

    subgraph1 = dace.sdfg.graph.SubgraphView(graph, list(set1))
    subgraph2 = dace.sdfg.graph.SubgraphView(graph, list(set2))
    print(list(set1))
    print(list(set2))
    fusion = subgraph.SubgraphFusion()
    fusion.apply(vadv_unfused, subgraph1)
    fusion.apply(vadv_unfused, subgraph2)
    vadv_unfused.view()


def test_fuse_all_numerically(gpu = False, view = False):
    I, J, K = (dace.symbol(s) for s in 'IJK')
    dtype = dace.float32
    np_dtype = np.float32
    _gt_loc__dtr_stage = dace.symbol('_gt_loc__dtr_stage', dace.float32)


    sdfg = dace.sdfg.SDFG.from_file('vadv32.sdfg')
    if gpu:
        sdfg.apply_gpu_transformations()
    graph = sdfg.nodes()[0]
    strides = {}
    for aname, arr in sdfg.arrays.items():
        if arr.transient:
            continue

        if len(arr.shape) == 3:
            istride = dace.symbol(f"_{aname}_I_stride")
            jstride = dace.symbol(f"_{aname}_J_stride")
            kstride = dace.symbol(f"_{aname}_K_stride")
            
            # bad:
            #dimtuple = (0,1,2) 
            # works best:
            #dimtuple = (0,2,1)
            # works ok:
            #dimtuple = (1,2,0)
            # works well:
            #dimtuple = (2,0,1)
            # ???
            dimtuple = (2,1,0)

            s = 1
            for i in reversed(dimtuple):
                strides[str(arr.strides[i])] = s
                s *= (I, J, K)[i]


    #dace.Config.set('compiler', 'cuda', 'default_block_size', value=block_size)

    I, J, K = 128, 128, 80
    syms = dict(I=I, J=J, K=K)
    dtype = np_dtype

    strides = {
        k: np.int32(dace.symbolic.evaluate(v, syms))
        if dace.symbolic.issymbolic(v) else np.int32(v)
        for k, v in strides.items()
    }

    wcon = np.random.rand(I+1, J, K).astype(np.float32)
    u_stage = np.random.rand(I, J, K).astype(np.float32)
    utens_stage = np.random.rand(I, J, K).astype(np.float32)
    u_pos = np.random.rand(I, J, K).astype(np.float32)
    utens = np.random.rand(I, J, K).astype(np.float32)


    args1 = dict(_gt_loc__dtr_stage=np.float32(1.424242424242),
                I=np.int32(I),
                J=np.int32(J),
                K=np.int32(K),
                **strides)

    sdfg.specialize(args1)
    for k, v in args1.items():
        print(k, v)
    args1 = {'wcon': wcon.copy(), 'u_stage': u_stage.copy(),
                  'utens_stage': utens_stage.copy(), 'u_pos': u_pos.copy(),
                  'utens': utens.copy()}

    args2 = copy.deepcopy(args1)

    if view:
        sdfg.view()
    sdfg(**args1)


    fusion(sdfg, graph)

    sdfg._name = 'fused'
    sdfg(**args2)

    if view:
        sdfg.view()


    assert np.allclose(args1['utens_stage'], args2['utens_stage'])
    print(np.linalg.norm(utens_stage))
    print("__________________")
    print(np.linalg.norm(args1['utens_stage']))
    print(np.linalg.norm(args2['utens_stage']))
    if gpu:
        print("GPU PASS")
    else:
        print("CPU PASS")



def test_fuse_partial_numerically(gpu = False, view = False):
    I, J, K = (dace.symbol(s) for s in 'IJK')
    dtype = dace.float32
    np_dtype = np.float32
    _gt_loc__dtr_stage = dace.symbol('_gt_loc__dtr_stage', dace.float32)


    sdfg = dace.sdfg.SDFG.from_file('vadv32.sdfg')
    if gpu:
        sdfg.apply_gpu_transformations()
    graph = sdfg.nodes()[0]

    set1 = set()
    set2 = set()
    graph = vadv_unfused.nodes()[0]
    for node in graph.nodes():
        for edge in itertools.chain(graph.out_edges(node), graph.in_edges(node)):
            if isinstance(edge.dst, (dace.sdfg.nodes.MapEntry, dace.sdfg.nodes.MapExit)):
                if '0' in edge.dst.map.label or '1' in edge.dst.map.label or '2' in edge.dst.map.label:
                    set1.add(node)
                if '3' in edge.dst.map.label or '4' in edge.dst.map.label:
                    set2.add(node)
            if isinstance(edge.src, (dace.sdfg.nodes.MapEntry, dace.sdfg.nodes.MapExit)):
                if '0' in edge.src.map.label or '1' in edge.src.map.label or '2' in edge.src.map.label:
                    set1.add(node)
                if '3' in edge.src.map.label or '4' in edge.src.map.label:
                    set2.add(node)
            if isinstance(node, (dace.sdfg.nodes.MapEntry, dace.sdfg.nodes.MapExit)):
                if '0' in node.map.label or '1' in node.map.label or '2' in node.map.label:
                    set1.add(node)
                if '3' in node.map.label or '4' in node.map.label:
                    set2.add(node)

    subgraph1 = dace.sdfg.graph.SubgraphView(graph, list(set1))
    subgraph2 = dace.sdfg.graph.SubgraphView(graph, list(set2))

    strides = {}
    for aname, arr in sdfg.arrays.items():
        if arr.transient:
            continue

        if len(arr.shape) == 3:
            istride = dace.symbol(f"_{aname}_I_stride")
            jstride = dace.symbol(f"_{aname}_J_stride")
            kstride = dace.symbol(f"_{aname}_K_stride")
            arr.strides = [istride, jstride, kstride]

            dimtuple = (0,2,1)

            s = 1
            for i in reversed(dimtuple):
                strides[str(arr.strides[i])] = s
                s *= (I, J, K)[i]

        if aname in sdfg.arrays:
            continue

        acpy = copy.deepcopy(arr)
        # TODO: change for GPU
        acopy.stroage = dace.StorageType.Default
        sdfg.add_datadesc(aname, acpy)

    # Cast SDFG from float64 to float32
    for sd, aname, arr in sdfg.arrays_recursive():
        arr.dtype = dace.float32
    for node, sd in sdfg.all_nodes_recursive():
        if not isinstance(node, dace.nodes.Node):
            continue
        for cname, conn in node.in_connectors.items():
            if isinstance(conn, dace.pointer):
                node.in_connectors[cname] = dace.pointer(dace.float32)
            elif conn is not None:
                node.in_connectors[cname] = dace.float32
        for cname, conn in node.out_connectors.items():
            if isinstance(conn, dace.pointer):
                node.out_connectors[cname] = dace.pointer(dace.float32)
            elif conn is not None:
                node.out_connectors[cname] = dace.float32

    #dace.Config.set('compiler', 'cuda', 'default_block_size', value=block_size)

    I, J, K = 128, 128, 80
    syms = dict(I=I, J=J, K=K)
    dtype = np_dtype

    strides = {
        k: np.int32(dace.symbolic.evaluate(v, syms))
        if dace.symbolic.issymbolic(v) else np.int32(v)
        for k, v in strides.items()
    }

    wcon = np.random.rand(I+1, J, K).astype(np.float32)
    u_stage = np.random.rand(I, J, K).astype(np.float32)
    utens_stage = np.random.rand(I, J, K).astype(np.float32)
    u_pos = np.random.rand(I, J, K).astype(np.float32)
    utens = np.random.rand(I, J, K).astype(np.float32)
    print(np.linalg.norm(utens_stage))

    args1 = dict(_gt_loc__dtr_stage=np.float32(1.424242424242),
                I=np.int32(I),
                J=np.int32(J),
                K=np.int32(K),
                **strides)
    sdfg.specialize(args1)
    args1.update({'wcon': wcon.copy(), 'u_stage': u_stage.copy(),
                  'utens_stage': utens_stage.copy(), 'u_pos': u_pos.copy(),
                  'utens': utens.copy()})

    args2 = copy.deepcopy(args1)

 
    if view:
        sdfg.view()
    csdfg = sdfg.compile()
    csdfg(**args1)
    del csdfg


    fusion(sdfg, graph, subgraph1)
    fusion(sdfg, graph, subgraph2)

    csdfg = sdfg.compile()
    csdfg(**args2)
    del csdfg

    if view:
        sdfg.view()
    
    print(np.linalg.norm(utens_stage))
    print(np.linalg.norm(args2['utens_stage']))
    print(np.linalg.norm(args1['utens_stage']))
    assert np.allclose(args1['utens_stage'], args2['utens_stage'])
    print("PASS")

#view_all()
#test_matching()
#test_fuse_all()
test_fuse_all_numerically(view = False, gpu = True)
#test_fuse_partial_numerically(view = False, gpu = False)

#test_fuse_partial()
