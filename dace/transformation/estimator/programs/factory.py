from dace.sdfg import SDFG
import numpy as np
import os

from .util import expand_maps, expand_reduce, fusion, stencil_tiling


DATATYPE = np.float64
PATH = os.path.expanduser('~/dace/dace/transformation/estimator/programs')


def prepare_expansion(sdfg, graph):
    expand_reduce(sdfg, graph)
    expand_maps(sdfg, graph)

def prepare_stencil(sdfg, graph):
    expand_reduce(sdfg, graph)
    stencil_tiling(sdfg, graph)

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
        prepare_stencil(sdfg, sdfg.nodes()[0])
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
    elif program_name == 'hdiff':
        # TODO
        return None
    elif program_name == 'hdiff_mini':
        # TODO
        return None
    else:
        raise NotImplementedError("Program not found")
