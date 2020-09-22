#include <cuda_runtime.h>
#include <stdio.h>
#include "kernels.h"

//#define MANUAL 

__global__ void fused(const dtype * __restrict__ gpu_A, dtype * __restrict__ gpu_C, int N) {
    {
        {
            int stencil_j = (blockIdx.x * 32 + threadIdx.x);
            int stencil_i = (blockIdx.y * 1 + threadIdx.y);
            
            if (stencil_j < (N - 4)) {
                {   // begin inner loop

                

                    #ifdef MANUAL
                    // make B 1-registers
                    //dtype b_up, b_mid, b_down, b_r, b_l;
                    dtype B[9] = {0};
                    {
                        // do everything manually.. oh god 
                        {
                            auto i = stencil_i + 1;
                            auto j = stencil_j + 2;
                            
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            
                            //b_down = dtype(0.2)*(a1+a2+a3+a4+a5);
                            B[0] = dtype(0.2)*(a1 + a2 + a3 + a4 + a5);
                        }
                        
                        {
                            auto i = stencil_i + 3;
                            auto j = stencil_j + 2;
                            
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            
                            //b_up = dtype(0.2)*(a1+a2+a3+a4+a5);
                            B[1] = dtype(0.2)*(a1 + a2 + a3 + a4 + a5);

                        }


                        {
                            auto i = stencil_i + 2;
                            auto j = stencil_j + 2;
                            
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            
                            //b_mid = dtype(0.2)*(a1+a2+a3+a4+a5);
                            B[2] = dtype(0.2)*(a1 + a2 + a3 + a4 + a5);
                        }
                        
                        {
                            auto i = stencil_i + 2;
                            auto j = stencil_j + 1;
                            
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            
                            //b_l = dtype(0.2)*(a1+a2+a3+a4+a5);
                            B[3] = dtype(0.2)*(a1 + a2 + a3 + a4 + a5);
                        }

                        {
                            auto i = stencil_i + 2;
                            auto j = stencil_j + 3;
                            
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            
                            //b_r = dtype(0.2)*(a1+a2+a3+a4+a5);
                            B[4] = dtype(0.2)*(a1 + a2 + a3 + a4 + a5);
                        }


                        {
                            auto i = stencil_i + 1;
                            auto j = stencil_j + 1;
                            
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            
                            //b_r = dtype(0.2)*(a1+a2+a3+a4+a5);
                            B[5] = dtype(0.2)*(a1 + a2 + a3 + a4 + a5);
                        }

                        {
                            auto i = stencil_i + 1;
                            auto j = stencil_j + 3;
                            
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            
                            //b_r = dtype(0.2)*(a1+a2+a3+a4+a5);
                            B[6] = dtype(0.2)*(a1 + a2 + a3 + a4 + a5);
                        }


                        {
                            auto i = stencil_i + 3;
                            auto j = stencil_j + 1;
                            
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            
                            //b_r = dtype(0.2)*(a1+a2+a3+a4+a5);
                            B[7] = dtype(0.2)*(a1 + a2 + a3 + a4 + a5);
                        }


                        {
                            auto i = stencil_i + 3;
                            auto j = stencil_j + 3;
                            
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            
                            //b_r = dtype(0.2)*(a1+a2+a3+a4+a5);
                            B[8] = dtype(0.2)*(a1 + a2 + a3 + a4 + a5);
                        }



                    }
                    
                    {
                        auto i = stencil_i + 2;
                        auto j = stencil_j + 2;

                        //dtype b = dtype(0.2)*(b_up + b_down + b_r + b_l + b_mid);
                        dtype b = dtype(0.2)*(B[0] + B[1] + B[2] + B[3] + B[4]);

                        gpu_C[((N * i) + j)] = b;
                    }
                    
                    #else 
                    dtype B[9] = {0};
                    
                   /* 
                    
                    {
                        for (auto i = (stencil_i + 1); i < (stencil_i + 3) + 1; i += 1) {
                            for (auto j = (stencil_j + 1); j < (stencil_j + 3) + 1; j += 1) {
                                {
                                    dtype a1 = gpu_A[((N * i) + j)];
                                    dtype a2 = gpu_A[(((N * i) + j) - 1)];
                                    dtype a3 = gpu_A[(((N * i) + j) + 1)];
                                    dtype a4 = gpu_A[((N * (i + 1)) + j)];
                                    dtype a5 = gpu_A[((N * (i - 1)) + j)];
                                    dtype b;

                                    ///////////////////
                                    // Tasklet code (a)
                                    b = (dtype(0.2) * ((((a1 + a2) + a3) + a4) + a5));
                                    ///////////////////

                                    B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 4)] = b;
                                }
                            }
                        }
                    }
                    
                    */
                    {
                        {

                            auto i = stencil_i + 1;
                            auto j = stencil_j + 1;
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            dtype b;

                            ///////////////////
                            // Tasklet code (a)
                            b = (dtype(0.2) * ((((a1 + a2) + a3) + a4) + a5));
                            ///////////////////

                            B[(((((3 * (stencil_i+1)) + (stencil_j+1)) - (3 * stencil_i)) - stencil_j) - 4)] = b;

                        }


                        {
                            auto i = stencil_i + 1;
                            auto j = stencil_j + 2;
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            dtype b;

                            ///////////////////
                            // Tasklet code (a)
                            b = (dtype(0.2) * ((((a1 + a2) + a3) + a4) + a5));
                            ///////////////////

                            B[(((((3 * (stencil_i+1)) + (stencil_j+2)) - (3 * stencil_i)) - stencil_j) - 4)] = b;

                        }
                        {

                            auto i = stencil_i + 1;
                            auto j = stencil_j + 3;
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            dtype b;

                            ///////////////////
                            // Tasklet code (a)
                            b = (dtype(0.2) * ((((a1 + a2) + a3) + a4) + a5));
                            ///////////////////

                            B[(((((3 * (stencil_i+1)) + (stencil_j+3)) - (3 * stencil_i)) - stencil_j) - 4)] = b;

                        }


                        {

                            auto i = stencil_i + 2;
                            auto j = stencil_j + 1;
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            dtype b;

                            ///////////////////
                            // Tasklet code (a)
                            b = (dtype(0.2) * ((((a1 + a2) + a3) + a4) + a5));
                            ///////////////////

                            B[(((((3 * (stencil_i+2)) + (stencil_j+1)) - (3 * stencil_i)) - stencil_j) - 4)] = b;

                        }


                        {

                            auto i = stencil_i + 2;
                            auto j = stencil_j + 2;
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            dtype b;

                            ///////////////////
                            // Tasklet code (a)
                            b = (dtype(0.2) * ((((a1 + a2) + a3) + a4) + a5));
                            ///////////////////

                            B[(((((3 * (stencil_i+2)) + (stencil_j+2)) - (3 * stencil_i)) - stencil_j) - 4)] = b;

                        }



                        {

                            auto i = stencil_i + 2;
                            auto j = stencil_j + 3;
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            dtype b;

                            ///////////////////
                            // Tasklet code (a)
                            b = (dtype(0.2) * ((((a1 + a2) + a3) + a4) + a5));
                            ///////////////////

                            B[(((((3 * (stencil_i+2)) + (stencil_j+3)) - (3 * stencil_i)) - stencil_j) - 4)] = b;

                        }


                        {

                            auto i = stencil_i + 3;
                            auto j = stencil_j + 1;
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            dtype b;

                            ///////////////////
                            // Tasklet code (a)
                            b = (dtype(0.2) * ((((a1 + a2) + a3) + a4) + a5));
                            ///////////////////

                            B[(((((3 * (stencil_i+3)) + (stencil_j+1)) - (3 * stencil_i)) - stencil_j) - 4)] = b;

                        }


                        {

                            auto i = stencil_i + 3;
                            auto j = stencil_j + 2;
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            dtype b;

                            ///////////////////
                            // Tasklet code (a)
                            b = (dtype(0.2) * ((((a1 + a2) + a3) + a4) + a5));
                            ///////////////////

                            B[(((((3 * (stencil_i+3)) + (stencil_j+2)) - (3 * stencil_i)) - stencil_j) - 4)] = b;

                        }



                        {

                            auto i = stencil_i + 3;
                            auto j = stencil_j + 3;
                            dtype a1 = gpu_A[((N * i) + j)];
                            dtype a2 = gpu_A[(((N * i) + j) - 1)];
                            dtype a3 = gpu_A[(((N * i) + j) + 1)];
                            dtype a4 = gpu_A[((N * (i + 1)) + j)];
                            dtype a5 = gpu_A[((N * (i - 1)) + j)];
                            dtype b;

                            ///////////////////
                            // Tasklet code (a)
                            b = (dtype(0.2) * ((((a1 + a2) + a3) + a4) + a5));
                            ///////////////////

                            B[(((((3 * (stencil_i+3)) + (stencil_j+3)) - (3 * stencil_i)) - stencil_j) - 4)] = b;

                        }

                    }

                

                    

                    
                    {
                        auto i = stencil_i + 2;
                        auto j = stencil_j + 2;
                        
                        //auto cse = ((((3 * i) + j) - (3 * stencil_i)) - stencil_j);
                        dtype a1 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 4)];
                        dtype a2 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 5)];
                        dtype a3 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 3)];
                        dtype a4 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 1)];
                        dtype a5 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 7)];
                        

                        //dtype a1 = B[(((((3 * (stencil_i+2)) + (stencil_j+2)) - (3 * stencil_i)) - stencil_j) - 4)];
                        //dtype a2 = B[(((((3 * (stencil_i+2)) + (stencil_j+2)) - (3 * stencil_i)) - stencil_j) - 5)];
                        //dtype a3 = B[(((((3 * (stencil_i+2)) + (stencil_j+2)) - (3 * stencil_i)) - stencil_j) - 3)];
                        //dtype a4 = B[(((((3 * (stencil_i+2)) + (stencil_j+2)) - (3 * stencil_i)) - stencil_j) - 1)];
                        //dtype a5 = B[(((((3 * (stencil_i+2)) + (stencil_j+2)) - (3 * stencil_i)) - stencil_j) - 7)];
                        dtype b;

                        ///////////////////
                        // Tasklet code (b)
                        b = (dtype(0.2) * ((((a1 + a2) + a3) + a4) + a5));
                        ///////////////////
                        gpu_C[((N * i) + j)] = b;
                    }
                    #endif
                    
                    /*
                    // original code (minus second inner trivial map):
                    {
                        for (auto i = (stencil_i + 1); i < (min((N - 2), (stencil_i + 3)) + 1); i += 1) {
                            for (auto j = (stencil_j + 1); j < (min((N - 2), (stencil_j + 3)) + 1); j += 1) {
                                {
                                    dtype a1 = gpu_A[((N * i) + j)];
                                    dtype a2 = gpu_A[(((N * i) + j) - 1)];
                                    dtype a3 = gpu_A[(((N * i) + j) + 1)];
                                    dtype a4 = gpu_A[((N * (i + 1)) + j)];
                                    dtype a5 = gpu_A[((N * (i - 1)) + j)];
                                    dtype b;

                                    ///////////////////
                                    // Tasklet code (a)
                                    b = (0.2 * ((((a1 + a2) + a3) + a4) + a5));
                                    ///////////////////

                                    B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 4)] = b;
                                }
                            }
                        }
                    }

                    {
                        auto i = stencil_i + 2;
                        auto j = stencil_j + 2;
                        dtype a1 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 4)];
                        dtype a2 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 5)];
                        dtype a3 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 3)];
                        dtype a4 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 1)];
                        dtype a5 = B[(((((3 * i) + j) - (3 * stencil_i)) - stencil_j) - 7)];
                        dtype b;

                        ///////////////////
                        // Tasklet code (b)
                        b = (0.2 * ((((a1 + a2) + a3) + a4) + a5));
                        ///////////////////

                        gpu_C[((N * i) + j)] = b;
                    }
                    
                    */
                }
            }
        }
    }
}


void run_fused(const dtype * __restrict__ gpu_A, dtype * __restrict__ gpu_C, int N, cudaStream_t stream){
    dim3 grid_sz = dim3(int_ceil(int_ceil((N - 4), 1), 32), int_ceil(int_ceil((N - 4), 1), 1), int_ceil(1, 1));
    dim3 block_sz = dim3(32, 1, 1);
    printf("Running Fused OPTIMIZED Kernel \n");
    fused<<<grid_sz, block_sz, 0, stream>>>(gpu_A, gpu_C, N);
    auto status = cudaPeekAtLastError();
    if(status != cudaSuccess){
        printf("ERROR");
    }
}


__global__ void kernel1(const dtype * __restrict__ gpu_A, dtype * __restrict__ B, int N) {
    {
        {
            int j = ((blockIdx.x * 32 + threadIdx.x) + 1);
            int i = ((blockIdx.y * 1 + threadIdx.y) + 1);
            if (j >= 1 && j < (N - 1)) {
                if (i >= 1) {
                    {
                        dtype a1 = gpu_A[((N * i) + j)];
                        dtype a2 = gpu_A[(((N * i) + j) - 1)];
                        dtype a3 = gpu_A[(((N * i) + j) + 1)];
                        dtype a4 = gpu_A[((N * (i + 1)) + j)];
                        dtype a5 = gpu_A[((N * (i - 1)) + j)];
                        dtype b;

                        ///////////////////
                        // Tasklet code (a)
                        b = (dtype(0.2) * ((((a1 + a2) + a3) + a4) + a5));
                        ///////////////////

                        B[((N * i) + j)] = b;
                    }
                }
            }
        }
    }
}


void run_kernel1(const dtype * __restrict__ gpu_A, dtype * __restrict__ gpu_B, int N, cudaStream_t stream){

    dim3 grid_sz = dim3(int_ceil(int_ceil((N - 2), 1), 32), int_ceil(int_ceil((N - 2), 1), 1), int_ceil(1, 1));
    dim3 block_sz = dim3(32, 1, 1);
    kernel1<<<grid_sz, block_sz, 0, stream>>>(gpu_A, gpu_B, N);
    auto status = cudaPeekAtLastError();
    if(status != cudaSuccess){
        printf("ERROR");
    }
}
__global__ void kernel2(const dtype * __restrict__ B, dtype * __restrict__ gpu_C, int N) {
    {
        {
            int j = ((blockIdx.x * 32 + threadIdx.x) + 2);
            int i = ((blockIdx.y * 1 + threadIdx.y) + 2);
            if (j >= 2 && j < (N - 2)) {
                if (i >= 2) {
                    {
                        dtype a1 = B[((N * i) + j)];
                        dtype a2 = B[(((N * i) + j) - 1)];
                        dtype a3 = B[(((N * i) + j) + 1)];
                        dtype a4 = B[((N * (i + 1)) + j)];
                        dtype a5 = B[((N * (i - 1)) + j)];
                        dtype b;

                        ///////////////////
                        // Tasklet code (b)
                        b = (dtype(0.2) * ((((a1 + a2) + a3) + a4) + a5));
                        ///////////////////

                        gpu_C[((N * i) + j)] = b;
                    }
                }
            }
        }
    }
}

void run_kernel2(const dtype * __restrict__ gpu_B, dtype * __restrict__ gpu_C, int N, cudaStream_t stream)
{
    dim3 grid_sz = dim3(int_ceil(int_ceil((N - 4), 1), 32), int_ceil(int_ceil((N - 4), 1), 1), int_ceil(1, 1));
    dim3 block_sz = dim3(32, 1, 1);
    kernel2<<<grid_sz, block_sz, 0, stream>>>(gpu_B, gpu_C, N);
    auto status = cudaPeekAtLastError();
    if(status != cudaSuccess){
        printf("ERROR");
    }
}
