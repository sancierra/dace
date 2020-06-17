import dace
import numpy as np
import sys


from dace.transformation.heterogeneous import ReduceMap
from dace.transformation.heterogeneous import SubgraphFusion
from dace.transformation.heterogeneous import MultiExpansion
import dace.transformation.heterogeneous.pipeline as pipeline
from dace.sdfg.graph import SubgraphView

import dace.libraries.standard as stdlib

import timeit

import dace.sdfg.nodes as nodes


dace_dtype = dace.float32
H, B, SN, SM = (dace.symbol(s) for s in ('H', 'B', 'SN', 'SM'))


@dace.program
def softmax(X_in: dace_dtype[H, B, SN, SM]):
    tmp_max = dace.reduce(lambda a, b: max(a, b), X_in, axis=3, identity = 0)

    tmp_out = np.ndarray([H, B, SN, SM], dtype=dace_dtype)
    out = np.ndarray([H, B, SN, SM], dtype=dace_dtype)

    # No broadcasting rules
    for i, j, k, l in dace.map[0:H, 0:B, 0:SN, 0:SM]:
        with dace.tasklet:
            inp << X_in[i, j, k, l]
            mx << tmp_max[i, j, k]
            o >> tmp_out[i, j, k, l]
            o = math.exp(inp - mx)
    #tmp_out = np.exp(X_in - tmp_max)

    tmp_sum = dace.reduce(lambda a, b: a + b, tmp_out, identity=0, axis=3)
    for i, j, k, l in dace.map[0:H, 0:B, 0:SN, 0:SM]:
        with dace.tasklet:
            inp << tmp_out[i, j, k, l]
            sm << tmp_sum[i, j, k]
            o >> out[i, j, k, l]
            o = inp / sm

    return out


sdfg = softmax.to_sdfg()
H.set(10); B.set(10); SN.set(50); SM.set(50)
A = np.ndarray((H.get(), B.get(), SN.get(), SM.get()), dtype = np.float32)


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

def test_graph():
    ################ baseline
    sdfg.apply_gpu_transformations()
    sdfg.view()
    graph = sdfg.nodes()[0]
    sglist = get_partition(sdfg, graph)
    print(sglist[0], sglist[1])
    print(sglist[0].nodes())
    print(sglist[1].nodes())

    return
    #csdfg = sdfg.compile_directly()
    print("#### Baseline")
    #csdfg(X_in = A, H=H, B=B, SN=SN, SM=SM)

    ################ reduce expansion
    pipeline.expand_reduce(sdfg, graph)
    # change SDFG to evade the 2 bugs
    label_index = 0
    sdfg.expand_library_nodes()
    for node in graph.nodes():
        if isinstance(node, nodes.NestedSDFG):
            # change the naming scheme
            for nested_state in node.sdfg.nodes():
                for nested_node in nested_state.nodes():
                    if isinstance(nested_node, nodes.MapEntry):
                        nested_node.label += str(label_index)
                        label_index += 1
                    for iedge in nested_state.in_edges(nested_node):
                        iedge.data.wcr_conflict = False
                    for oedge in nested_state.out_edges(nested_node):
                        oedge.data.wcr_conflict = False
    # Done.
    csdfg = sdfg.compile_directly()
    csdfg(X_in = A, H=H, B=B, SM=SM, SN=SN)






    ############### second, do MultiExpansion
    pipeline.expand_maps(sdfg, sdfg.nodes()[0])
    #sdfg.view()

    ############ third, do MapFusion
    pipeline.fusion(sdfg, sdfg.nodes()[0])
    #sdfg.view()
    sdfg.apply_strict_transformations()
    sdfg.view()

    sdfg.expand_library_nodes()
    sdfg.view()
    sdfg.validate()




if __name__ == "__main__":

    test_graph()
