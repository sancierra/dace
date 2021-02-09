import dace   
import numpy as np
import os

import dace.sdfg.nodes as nodes 
import dace.libraries as lib

from dace.transformation.subgraph import ReduceExpansion
from dace.transformation.subgraph.gemm import NestOut
from dace.transformation.interstate import InlineSDFG
from dace.sdfg.graph import SubgraphView

from dace.codegen import compiler
from dace.transformation.helpers import nest_state_subgraph


import substation
import substation.transformer as transformer


def run_pre_expansions(sdfg):
    # expands a raw encoder sdfg using 
    # transformations suitable for encoding 

    graph = sdfg.nodes()[0]

    print("PREPROCESSING...")
    print("EXPANDING NODES...")
    def process(sdfg, graph):
        for node in graph.nodes():

            if isinstance(node, nodes.NestedSDFG):
                # search in nested
                for g in node.sdfg.nodes():
                    process(node.sdfg, g)

            elif isinstance(node, lib.standard.nodes.Reduce):
                # expand reduction 
                #print(f"REDUCE: {node}")
                t = ReduceExpansion(sdfg.sdfg_id, sdfg.nodes().index(graph),
                                    {ReduceExpansion._reduce: graph.nodes().index(node)},
                                    0)
                t.apply(sdfg)

            elif isinstance(node, lib.blas.nodes.Gemm):
                #print("GEMM")
                pass

            elif isinstance(node, lib.blas.nodes.BatchedMatMul):
                #print("BMM")
                pass
            
            elif isinstance(node, lib.blas.nodes.MatMul):
                pass 
                '''
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
                '''

            elif isinstance(node, lib.blas.nodes.Transpose):
                #print("TRP")
                pass #FORNOW

            
            elif isinstance(node, dace.sdfg.nodes.LibraryNode):
                raise RuntimeError(f"Library Node {node} not covered")
    
    process(sdfg, graph)

    print("APPLYING STRICT TRAFOS...")
    sdfg.apply_strict_transformations()
    for node in graph.nodes():
        if isinstance(node, nodes.NestedSDFG) and 'bias' in node.label:
            
            InlineSDFG.apply_to(sdfg, _nested_sdfg = node)
    #sdfg.apply_transformations_repeated(InlineSDFG)

    print("FORNOW: ISOLATING BIAS LAYER...")
    for node in graph.nodes():
        if isinstance(node, nodes.MapEntry) and node.label == 'linear_with_bias_47':
            nest_state_subgraph(sdfg, graph, graph.scope_subgraph(node))
   
    print("VALIDATE...")
    sdfg.validate()
    print("DONE.")
    

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
    kwargs['norm1_bias'] = np.random.rand(N).astype(np.float32)
    kwargs['norm1_scale'] = np.random.rand(N).astype(np.float32)
    kwargs['linear1_w'] = np.random.rand(emb,N).astype(np.float32)
    kwargs['linear1_b'] = np.random.rand(emb).astype(np.float32)
    kwargs['linear2_b'] = np.random.rand(N).astype(np.float32)
    kwargs['linear2_w'] = np.random.rand(N,emb).astype(np.float32)
    kwargs['norm2_bias'] = np.random.rand(N).astype(np.float32)
    kwargs['norm2_scale'] = np.random.rand(N).astype(np.float32)
    
    kwargs['attn_dropout'] = np.ones((B,SM,N), dtype = np.float32)
    kwargs['linear1_dropout'] = np.ones((B,SM,emb), dtype = np.float32)
    kwargs['ff_dropout'] = np.ones((B,SM,N), dtype = np.float32)
    '''
    q, k, v have shape (batch, sequence length, embedding).
    wq, wk, wv have shape (heads, proj size, embedding).
    wo has shape (embedding, embedding).
    in_b is a bias for each linear projection for each head,
    shape (3, heads, proj size).
    out_b is a bias for wo, shape (embedding,).
    scale is a scalar.
    mask has shape (sequence length, sequence length).
    '''

    return kwargs

def get_args_numpy(args):
    kwargs = {} 

    # fetch required arguments from args 
    num_heads = args['H']
    proj_size = args['P']
    embed_size = args['N']

    kwargs['attn_in_b'] = np.zeros((3, num_heads, proj_size), dtype = np.float32)
    kwargs['attn_out_b'] = np.zeros((embed_size,), dtype = np.float32)
    
    kwargs['attn_dropout_p'] = 0
    kwargs['linear1_dropout_p'] = 0
    kwargs['ff_dropout_p'] = 0
    
    kwargs['x'] = args['x']
    kwargs['attn_wk'] = np.transpose(args['attn_wk'], (1,0,2))
    kwargs['attn_wv'] = np.transpose(args['attn_wv'], (1,0,2))
    kwargs['attn_wq'] = np.transpose(args['attn_wq'], (1,0,2))
    kwargs['attn_wo'] = np.reshape(args['attn_wo'], (args['N'], args['N']))

    kwargs['attn_scale'] = args['attn_scale']
    kwargs['norm1_bias'] = args['norm1_bias']
    kwargs['norm1_scale'] = args['norm1_scale']
    kwargs['linear1_w'] = args['linear1_w']
    kwargs['linear1_b'] = args['linear1_b']
    kwargs['linear2_b'] = args['linear2_b']
    kwargs['linear2_w'] = args['linear2_w']
    kwargs['norm2_bias'] = args['norm2_bias']
    kwargs['norm2_scale'] = args['norm2_scale']
    
    
    return kwargs 

    '''
    def encoder(x, attn_wq, attn_wk, attn_wv, attn_wo,
            attn_in_b, attn_out_b, attn_scale,
            norm1_scale, norm1_bias, norm2_scale, norm2_bias,
            linear1_w, linear1_b, linear2_w, linear2_b,
            attn_dropout_p, linear1_dropout_p, ff_dropout_p,
            activation='gelu'):
    '''

def run_encoder(sdfg, kwargs):
    sdfg.save('input.sdfg')
    result = sdfg(**kwargs)
    print(np.linalg.norm(result))
    return result 

def run_encoder_numpy(kwargs):
    result_vec = transformer.encoder(**kwargs)
    # normed2, ......
    result = result_vec[0]
    print(np.linalg.norm(result))
    return result 

    
def test_transformation():
    sdfg = get_encoder()
    kwargs = get_args()

    result1 = sdfg(**kwargs)

    run_pre_expansions(sdfg)
    result2 = sdfg(**kwargs)

    print(np.linalg.norm(result1))
    print(np.linalg.norm(result2))


def run_cached(sdfg, kwargs):
    
    binary_filename = compiler.get_binary_name(sdfg.build_folder,
                                                       sdfg.name)
    if os.path.isfile(binary_filename):
        executable = compiler.load_from_file(sdfg, binary_filename)
    else:
        raise RuntimeError()
    
    executable(**kwargs)
    
def run(run_baseline = True, 
        run_preprocessed = True,
        run_numpy = True,
        run_cached = False):
        
    results = {}

    sdfg = get_encoder()
    sdfg.validate() 

    kwargs_sdfg = get_args()
    kwargs_numpy = get_args_numpy(kwargs_sdfg)

    if gpu:
        gpu.apply_gpu_transformations()

    if run_baseline:
        ### vanilla sdfg 
        result1 = run_encoder(sdfg, kwargs_sdfg)
        results['baseline'] = result1

    if run_preprocessed:
        ### preprocessed sdfg 
        run_pre_expansions(sdfg)
        sdfg.validate()
        result2 = run_encoder(sdfg, kwargs_sdfg)
        results['preprocessed'] = result2


    if run_numpy:
        ### numpy reference
        result_np = run_encoder_numpy(kwargs_numpy)
        results['numpy_reference'] = result_np 

    if run_cached:
        ### cached
        result_cached = run_cached(sdfg, kwargs_sdfg)
        results['cached'] = result_cached 


    for (result_name, result_array) in results.items():
        print(np.linalg.norm(result_array), " -> ", result_name)
    
run(gpu = False,
    run_baseline = True, 
    run_preprocessed = True,
    run_numpy = True,
    run_cached = False)