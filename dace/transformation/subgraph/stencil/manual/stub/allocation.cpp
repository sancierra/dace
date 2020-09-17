/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <dace/dace.h>

DACE_EXPORTED void __dace_runkernel_a_0_0_2(double * __restrict__ B, const double * __restrict__ gpu_A, int N);
DACE_EXPORTED void __dace_runkernel_b_0_0_5(const double * __restrict__ B, double * __restrict__ gpu_C, int N);
void __program_stencil2d_transient_internal(double * __restrict__ A, double * __restrict__ C, int N)
{
    double * B = nullptr;
    cudaMalloc(&B, (N * N) * sizeof(double));

    {
        double * gpu_A = nullptr;
        cudaMalloc(&gpu_A, (N * N) * sizeof(double));
        double * gpu_C = nullptr;
        cudaMalloc(&gpu_C, (N * N) * sizeof(double));
        
        
        cudaMemcpyAsync(gpu_A, A, (N * N) * sizeof(double), cudaMemcpyHostToDevice, dace::cuda::__streams[0]);
        __dace_runkernel_a_0_0_2(B, gpu_A, N);
        __dace_runkernel_b_0_0_5(B, gpu_C, N);
        cudaMemcpyAsync(C, gpu_C, (N * N) * sizeof(double), cudaMemcpyDeviceToHost, dace::cuda::__streams[0]);
        cudaStreamSynchronize(dace::cuda::__streams[0]);
        
        
        cudaFree(gpu_A);
        cudaFree(gpu_C);
    }
    cudaFree(B);
}

DACE_EXPORTED void __program_stencil2d_transient(double * __restrict__ A, double * __restrict__ C, int N)
{
    __program_stencil2d_transient_internal(A, C, N);
}
DACE_EXPORTED int __dace_init_cuda(double * __restrict__ A, double * __restrict__ C, int N);
DACE_EXPORTED int __dace_exit_cuda(double * __restrict__ A, double * __restrict__ C, int N);

DACE_EXPORTED int __dace_init_stencil2d_transient(double * __restrict__ A, double * __restrict__ C, int N)
{
    int __result = 0;
    __result |= __dace_init_cuda(A, C, N);

    return __result;
}

DACE_EXPORTED void __dace_exit_stencil2d_transient(double * __restrict__ A, double * __restrict__ C, int N)
{
    __dace_exit_cuda(A, C, N);
}

