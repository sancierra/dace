import dace
import numpy


def failed_test():
    raise NotImplementedError("Callback test failed")


def mysquarer(inp):
    return inp**2


def answertolifeuniverseandeverything():
    return 42


def consumer(inp):
    assert inp == 6


def arraysquarer(outp_array, inp_array):
    import numpy as np
    np.copyto(outp_array, np.square(inp_array))


M = dace.symbolic.symbol()
N = dace.symbolic.symbol()
O = dace.symbolic.symbol()
giveandtake = dace.symbol('giveandtake', dace.callback(dace.uint32,
                                                       dace.uint32))
take = dace.symbol('take', dace.callback(None, dace.uint32))
give = dace.symbol('give', dace.callback(dace.uint32))
donothing = dace.symbol('donothing', dace.callback(None))


@dace.program
def callback_test(A: dace.uint32[2], B: dace.uint32[2]):
    @dace.map(_[0:2])
    def index(i):
        a << A[i]
        b >> B[i]
        b = giveandtake(a)
        take(a + 1)
        if give() != 42:
            donothing()


arrfunc = dace.symbol('arrfunc',
                      dtype=dace.callback(None, dace.float64[M, N, O],
                                          dace.float64[M, N, O]))


@dace.program(
    dace.float64[M, N, O],
    dace.float64[M, N, O],
)
def callback_with_arrays(out_arr, in_arr):
    with dace.tasklet:
        out << out_arr
        inp << in_arr
        arrfunc(out, inp)


if __name__ == "__main__":

    A = dace.ndarray((2, ), dtype=dace.int32)
    B = dace.ndarray((2, ), dtype=dace.int32)

    A[:] = 5
    B[:] = 0

    M.set(2)
    N.set(3)
    O.set(4)

    arr_in = numpy.random.randn(M.get(), N.get(), O.get())
    arr_out = dace.ndarray((M, N, O), dtype=dace.float64)

    callback_test(
        A,
        B,
        giveandtake=mysquarer,
        take=consumer,
        give=answertolifeuniverseandeverything,
        donothing=failed_test,
    )
    for b in B:
        if b != 25:
            failed_test()
    callback_with_arrays(arr_out, arr_in, arrfunc=arraysquarer)
    assert numpy.linalg.norm(arr_out - numpy.square(arr_in)) < 1e-10
