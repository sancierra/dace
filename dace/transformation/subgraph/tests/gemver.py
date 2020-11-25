import math
import dace

from dace.sdfg.graph import SubgraphView
import dace.transformation.subgraph.pipeline as pipeline

import numpy as np 

N = dace.symbol('N')
M = dace.symbol('M')
datatype = np.float32
dace_type = dace.float32 


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


@dace.program(dace_type[N, N], dace_type[N], dace_type[N], dace_type[N],
              dace_type[N], dace_type[N], dace_type[N], dace_type[N], dace_type[N],
              dace_type[1], dace_type[1])
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


@dace.program(dace_type[N, M], dace_type[M, M], dace_type[M])
def covariance(data, cov, mean):
    @dace.map
    def comp_mean(j: _[0:M], i: _[0:N]):
        inp << data[i, j]
        out >> mean(1, lambda x, y: x + y, 0)[j]
        out = inp

    @dace.map
    def comp_mean2(j: _[0:M]):
        inp << mean[j]
        out >> mean[j]
        out = inp / N

    @dace.map
    def sub_mean(i: _[0:N], j: _[0:M]):
        ind << data[i, j]
        m << mean[j]
        oud >> data[i, j]
        oud = ind - m

    @dace.mapscope
    def comp_cov_row(i: _[0:M]):
        @dace.mapscope
        def comp_cov_col(l: _[i:M]):
            @dace.map
            def comp_cov_k(k: _[0:N]):
                indi << data[k, i]
                indj << data[k, l]
                cov_ij >> cov(1, lambda x, y: x + y, 0)[i, l]
                cov_ij = (indi * indj)

    @dace.mapscope
    def symmetrize(i: _[0:M]):
        @dace.map
        def symmetrize_col(l: _[i:M]):
            cov_ij << cov[i, l]
            covout >> cov(2)[:, :]
            covout[i, l] = cov_ij / (N - 1)
            covout[l, i] = cov_ij / (N - 1)




@dace.program(dace_type[N, M], dace_type[M, M], dace_type[M], dace_type[M])
def correlation(data, corr, mean, stddev):
    @dace.map
    def comp_mean(j: _[0:M], i: _[0:N]):
        inp << data[i, j]
        out >> mean(1, lambda x, y: x + y, 0)[j]
        out = inp

    @dace.map
    def comp_mean2(j: _[0:M]):
        inp << mean[j]
        out >> mean[j]
        out = inp / N

    @dace.map
    def comp_stddev(j: _[0:M], i: _[0:N]):
        inp << data[i, j]
        inmean << mean[j]
        out >> stddev(1, lambda x, y: x + y, 0)[j]
        out = (inp - inmean) * (inp - inmean)

    @dace.map
    def comp_stddev2(j: _[0:M]):
        inp << stddev[j]
        out >> stddev[j]
        out = math.sqrt(inp / N)
        if out <= 0.1:
            out = 1.0

    @dace.map
    def center_data(i: _[0:N], j: _[0:M]):
        ind << data[i, j]
        m << mean[j]
        sd << stddev[j]
        oud >> data[i, j]
        oud = (ind - m) / (math.sqrt(float(N)) * sd)

    @dace.map
    def comp_corr_diag(i: _[0:M]):
        corrout >> corr[i, i]
        corrout = 1.0

    @dace.mapscope
    def comp_corr_row(i: _[0:M - 1]):
        @dace.mapscope
        def comp_corr_col(l: _[i + 1:M]):
            @dace.map
            def comp_cov_k(k: _[0:N]):
                indi << data[k, i]
                indj << data[k, l]
                cov_ij >> corr(1, lambda x, y: x + y, 0)[i, l]
                cov_ij = (indi * indj)

    @dace.mapscope
    def symmetrize(i: _[0:M - 1]):
        @dace.map
        def symmetrize_col(l: _[i + 1:M]):
            corrin << corr[i, l]
            corrout >> corr[l, i]
            corrout = corrin



def test(view = False,
         gpu = False,
         compile = False):

    sdfg = correlation.to_sdfg()
    if gpu:
        sdfg.apply_gpu_transformations(options = {'sequential_innermaps': False})
        #sdfg.apply_gpu_transformations()
    graph = sorted(sdfg.nodes(), key = lambda e: len(e.nodes()), reverse = True)[0]
    subgraph = SubgraphView(graph, graph.nodes())
    if view:
        sdfg.view()

    if compile:
        # define inputs

        N = 256
        M = 256
        data = np.random.rand(N,M).astype(datatype)
        cov = np.random.rand(M,M).astype(datatype)
        mean = np.random.rand(M).astype(datatype)
        stddev = np.random.rand(M).astype(datatype)

    if compile:
        data1 = data.copy()
        cov1 = cov.copy()
        mean1 = mean.copy()
        stddev1 = stddev.copy()
        sdfg(data=data1, corr=cov1, mean=mean1, stddev = stddev1, M=M, N=N)

    sdfg.save('pre.sdfg')
    pipeline.expand_reduce(sdfg, graph)
    pipeline.expand_maps(sdfg, graph)
    if view:
        sdfg.view()
    pipeline.fusion(sdfg,   
                    graph,
                    schedule_innermaps = dace.dtypes.ScheduleType.GPU_ThreadBlock,
                    transient_allocation = dace.dtypes.StorageType.GPU_Shared)
                    
    sdfg.save('post.sdfg')
    if view:
        sdfg.view()

    if compile:
        data2 = data.copy()
        cov2 = cov.copy()
        mean2 = mean.copy()
        stddev2 = stddev.copy()
        sdfg(data=data2, corr=cov2, mean=mean2, stddev = stddev2, M=M, N=N)

    if compile:

        print(np.linalg.norm(data1))
        print(np.linalg.norm(data2))
        print(np.linalg.norm(cov1))
        print(np.linalg.norm(cov2))
        print(np.linalg.norm(mean1))
        print(np.linalg.norm(mean2))
        assert np.allclose(data1, data2, rtol = 1e-4, atol=1e-6)
        assert np.allclose(cov1, cov2, rtol = 1e-4, atol=1e-6)
        assert np.allclose(mean1, mean2, rtol = 1e-4, atol=1e-6)

if __name__ == '__main__':
    test(view = False, gpu = True, compile = True)
