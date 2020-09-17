#include "helpers.h"
#include <cuda_runtime.h>


inline int int_ceil(int N, int D){
    return int((N+D-1)/D);
}



__global__ void fused(const double * __restrict__ gpu_A, double * __restrict__ gpu_C, int N) {
    {
        {
            int stencil_j = (blockIdx.x * 32 + threadIdx.x);
            int stencil_i = (blockIdx.y * 1 + threadIdx.y);
            double B[9] = {0};
            if (stencil_j < (N - 4)) {
                {
                    {
                        for (auto i = (stencil_i + 1); i < (min((N - 2), (stencil_i + 3)) + 1); i += 1) {
                            for (auto j = (stencil_j + 1); j < (min((N - 2), (stencil_j + 3)) + 1); j += 1) {
                                {
                                    double a1 = gpu_A[((N * i) + j)];
                                    double a2 = gpu_A[(((N * i) + j) - 1)];
                                    double a3 = gpu_A[(((N * i) + j) + 1)];
                                    double a4 = gpu_A[((N * (i + 1)) + j)];
                                    double a5 = gpu_A[((N * (i - 1)) + j)];
                                    double b;

                                    ///////////////////
                                    // Tasklet code (a)
                                    b = (2 * ((((a1 + a2) + a3) + a4) + a5));
                                    ///////////////////

                                    B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 4)] = b;
                                }
                            }
                        }
                    }
                    {
                        for (auto i = (stencil_i + 2); i < (stencil_i + 3); i += 1) {
                            for (auto j = (stencil_j + 2); j < (stencil_j + 3); j += 1) {
                                {
                                    double a1 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 4)];
                                    double a2 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 5)];
                                    double a3 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 3)];
                                    double a4 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 1)];
                                    double a5 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 7)];
                                    double b;

                                    ///////////////////
                                    // Tasklet code (b)
                                    b = (2 * ((((a1 + a2) + a3) + a4) + a5));
                                    ///////////////////

                                    gpu_C[((N * i) + j)] = b;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


void run_fused(const double * __restrict__ gpu_A, double * __restrict__ gpu_C, int N)

    dim3 grid_sz = dim3(int_ceil(int_ceil((N - 4), 1), 32), int_ceil(int_ceil((N - 4), 1), 1), int_ceil(1, 1));
    dim3 block_sz = dim3(32, 1, 1);
    fused<<<grid_sz, block_sz>>>(gpu_A, gpu_C, N)

}
