import math
import dace

from dace.sdfg.graph import SubgraphView
import dace.transformation.subgraph.pipeline as pipeline

N = dace.symbol('N')
datatype = dace.float64


def init_array(A, u1, v1, u2, v2, w, x, y, z, alpha, beta):
   n = N.get()

   alpha[0] = datatype(1.5)
   beta[0] = datatype(1.2)

   for i in range(n):
       u1[i] = i
       u2[i] = ((i + 1) / n) / 2.0
       v1[i] = ((i + 1) / n) / 4.0
       v2[i] = ((i + 1) / n) / 6.0
       y[i] = ((i + 1) / n) / 8.0
       z[i] = ((i + 1) / n) / 9.0
       x[i] = 0.0
       w[i] = 0.0
       for j in range(n):
           A[i, j] = datatype(i * j % n) / n


@dace.program(datatype[N, N], datatype[N], datatype[N], datatype[N],
             datatype[N], datatype[N], datatype[N], datatype[N], datatype[N],
             datatype[1], datatype[1])
def gemver(A, u1, v1, u2, v2, w, x, y, z, alpha, beta):
   @dace.map
   def add_uv(i: _[0:N], j: _[0:N]):
       iu1 << u1[i]
       iv1 << v1[j]
       iu2 << u2[i]
       iv2 << v2[j]
       ia << A[i, j]
       oa >> A[i, j]

       oa = ia + iu1 * iv1 + iu2 * iv2

   @dace.map
   def comp_y(i: _[0:N], j: _[0:N]):
       ib << beta
       ia << A[j, i]
       iy << y[j]
       ox >> x(1, lambda a, b: a + b)[i]

       ox = ib * ia * iy

   @dace.map
   def comp_xz(i: _[0:N]):
       ix << x[i]
       iz << z[i]
       ox >> x[i]
       ox = ix + iz

   @dace.map
   def comp_w(i: _[0:N], j: _[0:N]):
       ialpha << alpha
       ia << A[i, j]
       ix << x[j]
       ow >> w(1, lambda a, b: a + b)[i]
       ow = ialpha * ia * ix


def test(view = False,
         gpu = False,
         compile = False):

    sdfg = gemver.to_sdfg()
    if gpu:
        sdfg.apply_gpu_transformations()
    graph = sdfg.nodes()[0]
    subgraph = SubgraphView(graph, graph.nodes())
    if view:
        sdfg.view()

    if compile:
        # define inputs

        N = 100
        A = np.random.rand(N,N).astype(datatype)
        u1 = np.random.rand(N).astype(datatype)
        u2 = np.random.rand(N).astype(datatype)
        v1 = np.random.rand(N).astype(datatype)
        v2 = np.random.rand(N).astype(datatype)
        beta = np.random.rand(N).astype(datatype)
        y = np.random.rand(N).astype(datatype)
        w = np.zeros([N], dtype=datatype)

    if compile:
        A_copy = A.copy()
        w_copy = w.copy()
        x_copy = x.copy()
        sdfg(N=N, alpha=alpha, beta=beta,
             u1=u1, u2=u2, v1=v1, v2=v2,
             y=y, w=w_copy, x=x_copy, A=A_copy)


    pipeline.expand_reduce(sdfg, graph)
    pipeline.expand_maps(sdfg, graph)
    if view:
        sdfg.view()
    pipeline.fusion(sdfg, graph)
    if view:
        sdfg.view()


if __name__ == '__main__':
    sdfg = gemver.to_sdfg()
    test(view = True, gpu = False, compile = False)
