/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <iostream>
#include <random>
#include <chrono>

#include "kernels.h"

int N = 512;
typedef double dtype;

void run(){
    std::cout << "---Runner---" << std::endl;
    std::cout << "Allocating Arrays...." << std::endl;
    int N = 512;
    dtype* A = new dtype[N][N];
    for(int i = 0; i<N; ++i){
        for(int j=0; j<N; ++j){
            A[i][j] = dtype(i*j)/N
        }
    }

    double* gpu_A = 0;
    double* gpu_B = 0;
    double* gpu_C = 0;

    cudaMalloc(&gpu_A, (N*N)*sizeof(dtype));
    cudaMalloc(&gpu_B, (N*N)*sizeof(dtype));
    cudaMalloc(&gpu_C, (N*N)*sizeof(dtype));

    // run the fused version
    std::cout << "Running Unfused Kernels" << std::endl;
    cudaStream_t* stream;
    cudaStreamCreate(stream);

    cudaMemcpyAsync(gpu_A, A, (N * N) * sizeof(dtype), cudaMemcpyHostToDevice, stream0);
    auto start = high_resolution_clock::now();
    run_kernel1(gpu_A, gpu_B, N);
    run_kernel2(gpu_B, gpu_C, N);
    auto end = high_resolution_clock::now();
    std::cout << "Timer: " << duration_cast<microseconds>(end-start).count() << std::endl;

    cudaMemcpyAsync(C, gpu_C, (N * N) * sizeof(dtype), cudaMemcpyDeviceToHost, stream0);
    cudaStreamSynchronize(stream0);

    // run the non-fused version
    std::cout << "Running Fused Kernels" << std::endl;
    cudaMemcpyAsync(gpu_A, A, (N * N) * sizeof(dtype), cudaMemcpyHostToDevice, stream0);
    start = high_resolution_clock::now();
    run_kernel_fused(gpu_A, gpu_C);
    end = high_resolution_clock::now();
    std::cout << "Timer: " << duration_cast<microseconds>(end-start).count() << std::endl;

    cudaMemcpyAsync(C, gpu_C, (N * N) * sizeof(dtype), cudaMemcpyDeviceToHost, stream0);
    cudaStreamSynchronize(stream0);

    cudaFree(gpu_A);
    cudaFree(gpu_C);
    std::cout << "Done."  << std::endl;

}

int main(){
    run();
}
