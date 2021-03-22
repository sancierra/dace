import dace
import numpy as np


@dace.program
def nested_add1(A: dace.float64[3, 3], B: dace.float64[3, 3]):
    return A + B


@dace.program
def nested_add2(A: dace.float64[9], B: dace.float64[9]):
    return A + B


@dace.program
def reshape_node(A: dace.float64[9]):
    return A.reshape([3, 3])


def test_inline_reshape_views_work():
    @dace.program
    def test_inline_reshape_views_work(A: dace.float64[9], B: dace.float64[9]):
        result = dace.define_local([9], dace.float64)
        result[:] = nested_add1(A, B)
        return nested_add1(result, B)


    sdfg = test_inline_reshape_views_work.to_sdfg(strict=True)

    arrays = 0
    views = 0
    sdfg_used_desc = set([n.desc(sdfg) for n, _ in sdfg.all_nodes_recursive()
                          if isinstance(n, dace.nodes.AccessNode)])
    for desc in sdfg_used_desc:
        # View is subclas of Array, so we must do this check first
        if isinstance(desc, dace.data.View):
            views += 1
        elif isinstance(desc, dace.data.Array):
            arrays += 1
    
    assert(arrays == 4)
    assert(views == 3)


def test_regression_reshape_unsqueeze():
    nsdfg = dace.SDFG("nested_reshape_node")
    nstate = nsdfg.add_state()
    nsdfg.add_array("input", [9], dace.float64)
    nsdfg.add_view("view", [3, 3], dace.float64)
    nsdfg.add_array("output", [3, 3], dace.float64)

    R = nstate.add_read("input")
    A = nstate.add_access("view")
    W = nstate.add_write("output")

    nstate.add_edge(R, None, A, None, dace.Memlet("input[0:3, 0:3]"))
    nstate.add_edge(A, None, W, None, dace.Memlet("view[0:3, 0:3]"))

    @dace.program
    def test_reshape_unsqueeze(A: dace.float64[9], B: dace.float64[3, 3]):
        nsdfg(input=A, output=B)

    test_reshape_unsqueeze.to_sdfg()


def test_views_between_maps_work():
    @dace.program
    def test_inline_reshape_views_work(A: dace.float64[3, 3], B: dace.float64[9]):
        result = dace.define_local([9], dace.float64)
        result[:] = nested_add2(A, B)
        result_reshaped = reshape_node(result)

        return np.transpose(result_reshaped)


    sdfg = test_inline_reshape_views_work.to_sdfg(strict=False)
    sdfg.expand_library_nodes()
    sdfg.view()
    sdfg.apply_strict_transformations()

    assert(True)


if __name__ == "__main__":
    test_inline_reshape_views_work()
    test_views_between_maps_work()
