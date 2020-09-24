#include "kernels.h"
#include <cuda_runtime.h>


__device__ __forceinline__ void state1(const float * __in_gpu_A, const float * __in_gpu_A_0, const float * __in_gpu_A_1, const float * __in_gpu_A_2, const float * __in_gpu_A_3, float* __out_B, int N, int stencil_j) {

    {
        
        
        {
            float a1 = __in_gpu_A[0];
            float a2 = __in_gpu_A_0[0];
            float a3 = __in_gpu_A_1[2];
            float a4 = __in_gpu_A_2[(N + 1)];
            float a5 = __in_gpu_A_3[1];
            float b;

            ///////////////////
            // Tasklet code (a)
            b = (2 * ((((a1 + a2) + a3) + a4) + a5));
            ///////////////////

            __out_B[0] = b;
        }
    }
    {
        
        
        {
            float a1 = __in_gpu_A[1];
            float a2 = __in_gpu_A_0[1];
            float a3 = __in_gpu_A_1[3];
            float a4 = __in_gpu_A_2[(N + 2)];
            float a5 = __in_gpu_A_3[2];
            float b;

            ///////////////////
            // Tasklet code (a)
            b = (2 * ((((a1 + a2) + a3) + a4) + a5));
            ///////////////////

            __out_B[1] = b;
        }
    }
    {
        
        
        {
            float a1 = __in_gpu_A[2];
            float a2 = __in_gpu_A_0[2];
            float a3 = __in_gpu_A_1[4];
            float a4 = __in_gpu_A_2[(N + 3)];
            float a5 = __in_gpu_A_3[3];
            float b;

            ///////////////////
            // Tasklet code (a)
            b = (2 * ((((a1 + a2) + a3) + a4) + a5));
            ///////////////////

            __out_B[2] = b;
        }
    }
    
}

__device__ __forceinline__ void state2(const float * __in_gpu_A, const float * __in_gpu_A_0, const float * __in_gpu_A_1, const float * __in_gpu_A_2, const float * __in_gpu_A_3, float* __out_B, int N, int stencil_j) {

    {
        
        
        {
            float a1 = __in_gpu_A[0];
            float a2 = __in_gpu_A_0[0];
            float a3 = __in_gpu_A_1[2];
            float a4 = __in_gpu_A_2[(N + 1)];
            float a5 = __in_gpu_A_3[1];
            float b;

            ///////////////////
            // Tasklet code (a)
            b = (2 * ((((a1 + a2) + a3) + a4) + a5));
            ///////////////////

            __out_B[0] = b;
        }
    }
    {
        
        
        {
            float a1 = __in_gpu_A[1];
            float a2 = __in_gpu_A_0[1];
            float a3 = __in_gpu_A_1[3];
            float a4 = __in_gpu_A_2[(N + 2)];
            float a5 = __in_gpu_A_3[2];
            float b;

            ///////////////////
            // Tasklet code (a)
            b = (2 * ((((a1 + a2) + a3) + a4) + a5));
            ///////////////////

            __out_B[1] = b;
        }
    }
    {
        
        
        {
            float a1 = __in_gpu_A[2];
            float a2 = __in_gpu_A_0[2];
            float a3 = __in_gpu_A_1[4];
            float a4 = __in_gpu_A_2[(N + 3)];
            float a5 = __in_gpu_A_3[3];
            float b;

            ///////////////////
            // Tasklet code (a)
            b = (2 * ((((a1 + a2) + a3) + a4) + a5));
            ///////////////////

            __out_B[2] = b;
        }
    }
    
}

__device__ __forceinline__ void stencil3(const float * __in_gpu_A, const float * __in_gpu_A_0, const float * __in_gpu_A_1, const float * __in_gpu_A_2, const float * __in_gpu_A_3, float* __out_B, int N, int stencil_j) {

    {
        
        
        {
            float a1 = __in_gpu_A[0];
            float a2 = __in_gpu_A_0[0];
            float a3 = __in_gpu_A_1[2];
            float a4 = __in_gpu_A_2[(N + 1)];
            float a5 = __in_gpu_A_3[1];
            float b;

            ///////////////////
            // Tasklet code (a)
            b = (2 * ((((a1 + a2) + a3) + a4) + a5));
            ///////////////////

            __out_B[0] = b;
        }
    }
    {
        
        
        {
            float a1 = __in_gpu_A[1];
            float a2 = __in_gpu_A_0[1];
            float a3 = __in_gpu_A_1[3];
            float a4 = __in_gpu_A_2[(N + 2)];
            float a5 = __in_gpu_A_3[2];
            float b;

            ///////////////////
            // Tasklet code (a)
            b = (2 * ((((a1 + a2) + a3) + a4) + a5));
            ///////////////////

            __out_B[1] = b;

        }

    }
    {
        
        
        {
            float a1 = __in_gpu_A[2];
            float a2 = __in_gpu_A_0[2];
            float a3 = __in_gpu_A_1[4];
            float a4 = __in_gpu_A_2[(N + 3)];
            float a5 = __in_gpu_A_3[3];
            float b;

            ///////////////////
            // Tasklet code (a)
            b = (2 * ((((a1 + a2) + a3) + a4) + a5));
            ///////////////////

            __out_B[2] = b;
        }
    }
    
}

__device__ __forceinline__ void caller(const float * __in_gpu_A, float* __out_B, int N, int stencil_i, int stencil_j) {

    {
        
        
        state1(&__in_gpu_A[(N + 1)], &__in_gpu_A[N], &__in_gpu_A[N], &__in_gpu_A[N], &__in_gpu_A[0], &__out_B[0], N, stencil_j);
    }
    {
        
        
        state2(&__in_gpu_A[((2 * N) + 1)], &__in_gpu_A[(2 * N)], &__in_gpu_A[(2 * N)], &__in_gpu_A[(2 * N)], &__in_gpu_A[N], &__out_B[3], N, stencil_j);
    }
    {
        
        
        state3(&__in_gpu_A[((3 * N) + 1)], &__in_gpu_A[(3 * N)], &__in_gpu_A[(3 * N)], &__in_gpu_A[(3 * N)], &__in_gpu_A[(2 * N)], &__out_B[6], N, stencil_j);
    }
    
}



__global__ void outer_fused_0_0_9(const float * __restrict__ gpu_A, float * __restrict__ gpu_C, int N) {
    {
        {
            int stencil_j = (blockIdx.x * 32 + threadIdx.x);
            int stencil_i = (blockIdx.y * 1 + threadIdx.y);
            float *B = new float[9];
            memset(B, 0, sizeof(float)*9);
            if (stencil_j < (N - 4)) {
                {
                    caller(&gpu_A[((N * stencil_i) + stencil_j)], &B[0], N, stencil_i, stencil_j);
                    {
                        for (auto i = (stencil_i + 2); i < (stencil_i + 3); i += 1) {
                            for (auto j = (stencil_j + 2); j < (stencil_j + 3); j += 1) {
                                {
                                    float a1 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 4)];
                                    float a2 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 5)];
                                    float a3 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 3)];
                                    float a4 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 1)];
                                    float a5 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 7)];
                                    float b;

                                    ///////////////////
                                    // Tasklet code (b)
                                    b = (2 * ((((a1 + a2) + a3) + a4) + a5));
                                    ///////////////////

                                    gpu_C[((N * i) + j)] = b;
                                }
                            }
                        }
                    }
                    delete[] B;
                }
            }
        }
    }
}

void run_fused(const float * __restrict__ gpu_A, float * __restrict__ gpu_C, int N, cudaStream_t stream){
    outer_fused_0_0_9<<<dim3(int_ceil(int_ceil((N - 4), 1), 32), int_ceil(int_ceil((N - 4), 1), 1), int_ceil(1, 1)), dim3(32, 1, 1)>>>(gpu_A, gpu_C, N);
    cudaError_t error = cudaGetLastError();
    if (error != 0){
        printf("ERROR in run_fused");
    }
}
