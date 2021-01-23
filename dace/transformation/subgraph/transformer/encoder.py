import dace   
import numpy as np


def get_encoder():
    sdfg = dace.sdfg.SDFG.from_file('../../estimator/programs/encoder.sdfg')
    return sdfg  

def expand_encoder():
    sdfg = get_encoder()
    sdfg.expand_library_nodes()
    sdfg.save('../../estimator/programs/encoder_expanded.sdfg')

def run_encoder():
    sdfg = get_encoder()
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
 
    result = sdfg(**kwargs)
    print(np.linalg.norm(result))


run_encoder()
