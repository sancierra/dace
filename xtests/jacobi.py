#!/usr/bin/env python
from __future__ import print_function

import argparse
import dace
import numpy as np
from scipy import ndimage

W = dace.symbol('W')
H = dace.symbol('H')
MAXITER = dace.symbol('MAXITER')


@dace.program(dace.float32[H, W], dace.int32)
def jacobi(A, iterations):
    # Transient variable
    tmp = dace.define_local([H, W], dtype=A.dtype)

    @dace.map(_[0:H, 0:W])
    def reset_tmp(y, x):

        out >> tmp[y, x]
        out = dace.float32(0.0)

    for t in range(iterations):

        @dace.map(_[1:H - 1, 1:W - 1])
        def a2b(y, x):
            in_N << A[y - 1, x]
            in_S << A[y + 1, x]
            in_W << A[y, x - 1]
            in_E << A[y, x + 1]
            in_C << A[y, x]
            out >> tmp[y, x]

            out = dace.float32(0.2) * (in_C + in_N + in_S + in_W + in_E)

        # Double buffering
        @dace.map(_[1:H - 1, 1:W - 1])
        def b2a(y, x):
            in_N << tmp[y - 1, x]
            in_S << tmp[y + 1, x]
            in_W << tmp[y, x - 1]
            in_E << tmp[y, x + 1]
            in_C << tmp[y, x]
            out >> A[y, x]

            out = dace.float32(0.2) * (in_C + in_N + in_S + in_W + in_E)


if __name__ == "__main__":
    print("==== Program start ====")
    sdfg = jacobi.to_sdfg()
    sdfg.view()
    sdfg_id = 0
    state_id = 1
    entry1 = None
    entry2 = None
    for node in sdfg.nodes()[1].nodes():
        if isinstance(node, dace.sdfg.nodes.MapEntry):
            if node.label == 'a2b':
                entry1 = node
            if node.label == 'b2a':
                entry2 = node
    d1 = {dace.transformation.dataflow.tiling.MapTiling._map_entry: sdfg.nodes()[1].nodes().index(entry1)}
    d2 = {dace.transformation.dataflow.tiling.MapTiling._map_entry: sdfg.nodes()[1].nodes().index(entry2)}

    print(entry1)
    print(entry2)

    t1 = dace.transformation.dataflow.tiling.MapTiling(sdfg_id, state_id, d1, 0)
    t1.apply(sdfg)
    t2 = dace.transformation.dataflow.tiling.MapTiling(sdfg_id, state_id, d2, 0)
    t2.apply(sdfg)


    sdfg.view()
    asdfasdf

    parser = argparse.ArgumentParser()
    parser.add_argument("W", type=int, nargs="?", default=2000)
    parser.add_argument("H", type=int, nargs="?", default=2000)
    parser.add_argument("MAXITER", type=int, nargs="?", default=30)
    args = vars(parser.parse_args())

    W.set(args["W"])
    H.set(args["H"])
    MAXITER.set(args["MAXITER"])

    print('Jacobi 5-point Stencil %dx%d (%d steps)' %
          (W.get(), H.get(), MAXITER.get()))

    A = dace.ndarray([H, W], dtype=dace.float32)

    # Initialize arrays: Randomize A, zero B
    A[:] = dace.float32(0)
    A[1:H.get() - 1, 1:W.get() - 1] = np.random.rand(
        (H.get() - 2), (W.get() - 2)).astype(dace.float32.type)
    regression = np.ndarray([H.get() - 2, W.get() - 2], dtype=np.float32)
    regression[:] = A[1:H.get() - 1, 1:W.get() - 1]

    #print(A.view(type=np.ndarray))

    #############################################
    # Run DaCe program

    jacobi(A, MAXITER)

    # Regression
    kernel = np.array([[0, 0.2, 0], [0.2, 0.2, 0.2], [0, 0.2, 0]],
                      dtype=np.float32)
    for i in range(2 * MAXITER.get()):
        regression = ndimage.convolve(regression,
                                      kernel,
                                      mode='constant',
                                      cval=0.0)

    residual = np.linalg.norm(A[1:H.get() - 1, 1:W.get() - 1] -
                              regression) / (H.get() * W.get())
    print("Residual:", residual)

    #print(A.view(type=np.ndarray))
    #print(regression.view(type=np.ndarray))

    print("==== Program end ====")
    exit(0 if residual <= 0.05 else 1)
