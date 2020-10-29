from dace.sdfg import SDFG
import numpy as np
import os

from .util import expand_maps, expand_reduce, fusion, stencil_tiling


DATATYPE = np.float32
PATH = os.path.expanduser('~/dace/dace/transformation/estimator/programs')


def prepare_expansion(sdfg, graph):
    expand_reduce(sdfg, graph)
    expand_maps(sdfg, graph)

def get_program(program_name):
    '''
    returns a post-processed SDFG of the given program
    '''
    data_suffix = None
    if DATATYPE == np.float32:
        data_suffix = str(32)
    elif DATATYPE == np.float64:
        data_suffix = str(64)
    else:
        raise NotImplementedError("Wrong Datatype")


    if program_name == 'synthetic':
        sdfg = SDFG.from_file(os.path.join(PATH,'synthetic'+data_suffix+'.sdfg'))
        prepare_expansion(sdfg, sdfg.nodes()[0])
    elif program_name == 'vadv':
        sdfg = SDFG.from_file(os.path.join(PATH,'vadv'+data_suffix+'.sdfg'))
        prepare_expansion(sdfg, sdfg.nodes()[0])
    elif program_name == 'hdiff':
        sdfg = SDFG.from_file(os.path.join(PATH,'hdiff'+data_suffix+'.sdfg'))
    elif program_name == 'hdiff_mini':
        sdfg = SDFG.from_file(os.path.join(PATH,'hdiff_mini'+data_suffix+'.sdfg'))
    elif program_name == 'softmax':
        sdfg = SDFG.from_file(os.path.join(PATH,'softmax'+data_suffix+'.sdfg'))
        prepare_expansion(sdfg, sdfg.nodes()[0])
    elif program_name == 'correlation':
        sdfg = SDFG.from_file(os.path.join(PATH,'correlation'+data_suffix+'.sdfg'))
        prepare_expansion(sdfg, sdfg.nodes()[0])
    elif program_name == 'transformer':
        sdfg = SDFG.from_file(os.path.join(PATH,'transformer'+data_suffix+'.sdfg'))
        prepare_expansion(sdfg, sdfg.nodes()[0])
    else:
        raise NotImplementedError("Program not found")

    return sdfg

def get_args(program_name):
    '''
    returns a triple of dicts (inputs, ouputs, symbols)
    that serve as args for a given program
    '''
    if program_name == 'synthetic':
        N, M, O = 50, 60, 70
        return (
                {'A': np.random.rand(N).astype(DATATYPE),
                 'B': np.random.rand(M).astype(DATATYPE),
                 'C': np.random.rand(O).astype(DATATYPE)},
                {'out1': np.zeros((N, M), DATATYPE),
                 'out2': np.zeros((1), DATATYPE),
                 'out3': np.zeros((N, M, O))},
                {'N': N, 'M': M, 'O':O}
               )
    elif program_name == 'softmax':
        H, B, SN, SM = 16, 8, 20, 20
        return (
                {'X_in': np.random.rand(H, B, SN, SM).astype(DATATYPE)},
                {},
                {'H':H, 'B':B, 'SN':SN, 'SM':SM}
        )

    elif program_name == 'vadv':
        I, K, J = 128, 128, 80
        wcon = np.random.rand(I+1, J, K).astype(DATATYPE)
        u_stage = np.random.rand(I, J, K).astype(DATATYPE)
        utens_stage = np.random.rand(I, J, K).astype(DATATYPE)
        u_pos = np.random.rand(I, J, K).astype(DATATYPE)
        utens = np.random.rand(I, J, K).astype(DATATYPE)
        return({
                'wcon': wcon,
                'u_stage': u_stage,
                'utens_stage': utens_stage,
                'u_pos': u_pos,
                'utens': utens
               },
               {
               'utens_stage': utens_stage
               },
               {'_gt_loc__dtr_stage': 1.4242424,
                'I': I,
                'J': J,
                'K': 80,
                '_utens_stage_J_stride': 1,
                '_utens_stage_K_stride': 128,
                '_utens_stage_I_stride': 10240,
                '_u_stage_J_stride': 1,
                '_u_stage_K_stride': 128,
                '_u_stage_I_stride': 10240,
                '_wcon_J_stride': 1,
                '_wcon_K_stride': 128,
                '_wcon_I_stride': 10240,
                '_u_pos_J_stride': 1,
                '_u_pos_K_stride': 128,
                '_u_pos_I_stride': 10240,
                '_utens_J_stride': 1,
                '_utens_K_stride': 128,
                '_utens_I_stride': 10240
               }
        )

    elif program_name == 'hdiff_mini':
        I, J, K = 128, 128, 80
        halo = 4
        print("****")
        return(
               {
                'pp_in' : np.random.rand(J, K+1, I).astype(DATATYPE),
                'u_in' : np.random.rand(J, K+1, I).astype(DATATYPE),
                'crlato' : np.random.rand(J).astype(DATATYPE),
                'crlatu' : np.random.rand(J).astype(DATATYPE),
                'hdmask' : np.random.rand( J, K+1, I ).astype(DATATYPE),
                'w_in' : np.random.rand( J, K+1, I).astype(DATATYPE),
                'v_in' : np.random.rand( J, K+1, I).astype(DATATYPE)
               },
               {
                'pp' : np.zeros([ J, K+1, I ], dtype = DATATYPE),
                'w' : np.zeros([ J, K+1, I ], dtype = DATATYPE),
                'v' : np.zeros([ J, K+1, I ], dtype = DATATYPE),
                'u' : np.zeros([ J, K+1, I ], dtype = DATATYPE)
               },
               {'I': I, 'J': J, 'K': K, 'halo': halo}
        )

    elif program_name == 'hdiff':
        I, J, K = 128, 128, 80
        halo = 4

        return(
               {
                'pp_in' : np.random.rand(J, K, I).astype(DATATYPE),
                'u_in' : np.random.rand(J, K, I).astype(DATATYPE),
                'crlato' : np.random.rand(J).astype(DATATYPE),
                'crlatu' : np.random.rand(J).astype(DATATYPE),
                'acrlat0' : np.random.rand(J).astype(DATATYPE),
                'crlavo' : np.random.rand(J).astype(DATATYPE),
                'crlavu' : np.random.rand(J).astype(DATATYPE),
                'hdmask' : np.random.rand( J, K, I ).astype(DATATYPE),
                'w_in' : np.random.rand( J, K, I).astype(DATATYPE),
                'v_in' : np.random.rand( J, K, I).astype(DATATYPE)
               },
               {
                'pp_out' : np.zeros([ J, K, I ], dtype = DATATYPE),
                'w_out' : np.zeros([ J, K, I ], dtype = DATATYPE),
                'v_out' : np.zeros([ J, K, I ], dtype = DATATYPE),
                'u_out' : np.zeros([ J, K, I ], dtype = DATATYPE)
               },
               {'I': I, 'J': J, 'K': K, 'halo': halo}
        )

    elif program_name == 'transformer':
        raise NotImplementedError("TODO")
    else:
        raise NotImplementedError("Program not found")