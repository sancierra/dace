import dace   
import numpy as np

import dace.sdfg.nodes as nodes 
import dace.libraries as lib

from dace.transformation.subgraph import ReduceExpansion
from dace.sdfg.graph import SubgraphView

from dace.transformation.subgraph.gemm import NestOut


def run_pre_expansions(sdfg):
    # expands a raw encoder sdfg using 
    # transformations suitable for encoding 

    graph = sdfg.nodes()[0]

    def process(sdfg, graph):
        for node in graph.nodes():

            if isinstance(node, nodes.NestedSDFG):
                # search in nested
                for g in node.sdfg.nodes():
                    process(node.sdfg, g)

            elif isinstance(node, lib.standard.nodes.Reduce):
                # expand reduction 
                print(f"REDUCE: {node}")
                t = ReduceExpansion(sdfg.sdfg_id, sdfg.nodes().index(graph),
                                    {ReduceExpansion._reduce: graph.nodes().index(node)},
                                    0)
                t.apply(sdfg)

            elif isinstance(node, lib.blas.nodes.Gemm):
                print("GEMM")
                pass
                # TODO GEMM 

            elif isinstance(node, lib.blas.nodes.BatchedMatMul):
                print("BMM")
                pass
                # TODO BMM
            
            elif isinstance(node, lib.blas.nodes.MatMul):
                print(f"MM: {node.label}")
                print(type(node))
                label = node.label
                handle_prior = next(iter(graph.in_edges(node))).src
                impl = node.expand(sdfg, graph)
                node_post = graph.out_edges(handle_prior)[0].dst
                impl = node_post.expand(sdfg, graph)
                nsdfg = graph.out_edges(handle_prior)[0].dst

                # case 1
                if label == 'einsum_gemm':
                    # case 1.1: Two maps
                    # apply vanilla NestOut to nested sdfg 
                    nsdfg.sdfg.apply_transformations(NestOut)                
                
                elif label == '_MatMult_':
                    pass #FORNOW

            elif isinstance(node, lib.blas.nodes.Transpose):
                print("TRP")
                pass #FORNOW

            
            elif isinstance(node, dace.sdfg.nodes.LibraryNode):
                raise RuntimeError(f"Library Node {node} not covered")
    
    process(sdfg, graph)
    sdfg.apply_strict_transformations()
    sdfg.validate()
    sdfg.save('preprocessed.sdfg')
    

def get_encoder():
    # returns a raw encoder sdfg
    sdfg = dace.sdfg.SDFG.from_file('../../estimator/programs/encoder.sdfg')
    return sdfg  

def get_args():
    kwargs = {}
    B = 16; SM = 20; P = 8; H = 5; emb = 15; N=P*H
    kwargs.update({'B':np.int32(B), 'SM': np.int32(SM), 'N':np.int32(N), 'P':np.int32(P), 'H':np.int32(H), 'emb':np.int32(emb)})
    
    kwargs['attn_wk'] = np.random.rand(P,H,N).astype(np.float32)
    kwargs['x'] = np.random.rand(B,SM,N).astype(np.float32)
    kwargs['attn_wv'] = np.random.rand(P,H,N).astype(np.float32)
    kwargs['attn_scale'] = np.float32(1.0)
    kwargs['attn_wq'] = np.random.rand(P,H,N).astype(np.float32)
    kwargs['attn_wo'] = np.random.rand(P,H,N).astype(np.float32)
    kwargs['attn_dropout'] = np.random.rand(B,SM,N).astype(np.float32)
    kwargs['norm1_bias'] = np.random.rand(N).astype(np.float32)
    kwargs['norm1_scale'] = np.random.rand(N).astype(np.float32)
    kwargs['linear1_w'] = np.random.rand(emb,N).astype(np.float32)
    kwargs['linear1_b'] = np.random.rand(emb).astype(np.float32)
    kwargs['linear1_dropout'] = np.random.rand(B,SM,emb).astype(np.float32)
    kwargs['linear2_b'] = np.random.rand(N).astype(np.float32)
    kwargs['linear2_w'] = np.random.rand(N,emb).astype(np.float32)
    kwargs['ff_dropout'] = np.random.rand(B,SM,N).astype(np.float32)
    kwargs['norm2_bias'] = np.random.rand(N).astype(np.float32)
    kwargs['norm2_scale'] = np.random.rand(N).astype(np.float32)
    
    return kwargs 

def expand_encoder(sdfg):
    # expands a raw encoder sdfg (default expansion)
    sdfg.expand_library_nodes()
    sdfg.save('../../estimator/programs/encoder_expanded.sdfg')


def run_encoder(sdfg, kwargs):
    result = sdfg(**kwargs)
    print(np.linalg.norm(result))

def test_transformation():
    sdfg = get_encoder()
    kwargs = get_args()

    result1 = sdfg(**kwargs)

    run_pre_expansions(sdfg)
    result2 = sdfg(**kwargs)

    print(np.linalg.norm(result1))
    print(np.linalg.norm(result2))


sdfg = get_encoder()
run_pre_expansions(sdfg)