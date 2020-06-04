import dace
import numpy as np

from dace.perf.roofline import Roofline
from dace.perf.specs import *
from dace.perf.optimizer import SDFGRooflineOptimizer

from dace.transformation.heterogeneous import ReduceMap
from dace.transformation.heterogeneous import SubgraphFusion
from dace.transformation.heterogeneous import MultiExpansion

import dace.libraries.standard as stdlib

import timeit

import dace.measure.pipeline as pipeline
from dace.measure.runner import Runner


dace_dtype = dace.float32
H, B, SN, SM = (dace.symbol(s) for s in ('H', 'B', 'SN', 'SM'))


@dace.program
def softmax(X_in: dace_dtype[H, B, SN, SM], TEST: dace_dtype[H, B, SN]):
    tmp_max = dace.reduce(lambda a, b: max(a, b), X_in, axis=3, identity = 0)
    TEST[:] = tmp_max[:]

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
roofline = Roofline(PERF_CPU_CRAPBOOK, symbols = {H:3, B:3, SN:5, SM:5})
graph = sdfg.nodes()[0]
H.set(10); B.set(10); SN.set(50); SM.set(50)

def test_graph():
    ################ first, expand the reduce node
    pipeline.expand_reduce(sdfg, graph)
    sdfg.view()

    ############### second, do MultiExpansion
    pipeline.expand_maps(sdfg, graph)
    sdfg.view()

    ############ third, do MapFusion
    pipeline.fusion(sdfg, graph)

    sdfg.apply_strict_transformations()
    sdfg.view()

    sdfg.validate()


def test_result(debug = False):
    '''
    X_in = np.random.rand(H.get(), B.get(), SN.get(), SM.get()).astype(np.float32)

    TEST = np.zeros([H.get(), B.get(), SN.get()], dtype = np.float32)

    X_out_baseline = np.zeros([H.get(), B.get(), SN.get(), SM.get()], dtype = np.float32)
    X_out_1 = np.zeros([H.get(), B.get(), SN.get(), SM.get()], dtype = np.float32)
    X_out_2 = np.zeros([H.get(), B.get(), SN.get(), SM.get()], dtype = np.float32)
    X_out_3 = np.zeros([H.get(), B.get(), SN.get(), SM.get()], dtype = np.float32)
    '''
    debugger = Runner(measure_mode = ['median', 'avg', 'std'],
                      view_roofline = True)

    # first test symbols dict build function
    symbols = Runner.build_symbols_dict(H, B, SM, SN)
    print(symbols)

    # second, test test_run function
    '''
    debugger.test_run(sdfg = sdfg, graph = graph,
                      inputs = {'X_in': X_in, 'TEST': TEST},
                      outputs = {'TEST': TEST},
                      symbols = symbols,
                      roofline = roofline)

    # thrid, test automatic argument generation
    input_args, output_args = \
        Runner.generate_arguments(sdfg, symbols,
                                  outputs = ['TEST'])

    print("Input_args")
    print("Output_args")
    print(input_args)
    print(output_args)

    '''
    debugger.go(sdfg, graph, None, H, B, SN, SM,
                performance_spec = dace.perf.specs.PERF_CPU_CRAPBOOK,
                output=['TEST'])


    #############




if __name__ == "__main__":
    test_result()
