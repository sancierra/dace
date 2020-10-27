from dace.sdfg import SDFG
import numpy as np
import os

DATATYPE = np.float64
PATH = os.path.expanduser('~/dace/dace/transformation/estimator/programs')

def get_program(program_name):
    data_suffix = None
    if DATATYPE == np.float32:
        data_suffix = str(32)
    elif DATATYPE == np.float64:
        data_suffix = str(64)
    else:
        raise NotImplementedError("Wrong Datatype")


    if program_name == 'synthetic':
        return SDFG.from_file(os.path.join(PATH,'synthetic'+data_suffix+'.sdfg'))
    elif program_name == 'vadv':
        return SDFG.from_file(os.path.join(PATH,'vadv'+data_suffix+'.sdfg'))
    elif program_name == 'hdiff':
        return SDFG.from_file(os.path.join(PATH,'hdiff'+data_suffix+'.sdfg'))
    elif program_name == 'softmax':
        return SDFG.from_file(os.path.join(PATH,'softmax'+data_suffix+'.sdfg'))
    elif program_name == 'correlation':
        return SDFG.from_file(os.path.join(PATH,'correlation'+data_suffix+'.sdfg'))
    elif program_name == 'transformer':
        return SDFG.from_file(os.path.join(PATH,'transformer'+data_suffix+'.sdfg'))
    else:
        raise NotImplementedError("Program not found")

def get_args(program_name):
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
                {'__return': np.zeros((H, B, SN, SM), DATATYPE)},
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
