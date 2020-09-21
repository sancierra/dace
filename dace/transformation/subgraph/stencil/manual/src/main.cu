/* DaCe AUTO-GENERATED FILE. DO NOT MODIFY */
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>

#include "kernels.h"

const int N = 512;
const int TREPS = 1;


void run(){
    std::cout << "---Runner---" << std::endl;
    std::cout << "Allocating Arrays...." << std::endl;

    dtype* A = new dtype[N*N];
    dtype* C = new dtype[N*N];

    dtype* result1 = new dtype[N*N];
    dtype* result2 = new dtype[N*N];

    for(int i = 0; i<N; ++i){
        for(int j=0; j<N; ++j){
            A[i*N + j] = dtype(i*j)/(N*N);
        }
    }

    dtype* gpu_A = 0;
    dtype* gpu_B = 0;
    dtype* gpu_C = 0;

    cudaMalloc(&gpu_A, (N*N)*sizeof(dtype));
    cudaMalloc(&gpu_B, (N*N)*sizeof(dtype));
    cudaMalloc(&gpu_C, (N*N)*sizeof(dtype));

    // run the fused version
    std::cout << "Running Unfused Kernels" << std::endl;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(gpu_A, A, (N * N) * sizeof(dtype), cudaMemcpyHostToDevice, stream);
    // #########################
    auto runtimes = std::vector<double>(TREPS);
    for(int run=0; run<TREPS; run++){
        auto start = std::chrono::high_resolution_clock::now();
        run_kernel1(gpu_A, gpu_B, N, stream);
        #ifdef DEBUG
        auto error = cudaGetLastError();
        if(error != 0){
            std::cout << "ERROR: " << error << std::endl;
        }
        #endif
        run_kernel2(gpu_B, gpu_C, N, stream);
        #ifdef DEBUG
        error = cudaGetLastError();
        if(error != 0){
            std::cout << "ERROR: " << error << std::endl;
        }
        #endif 
        auto end = std::chrono::high_resolution_clock::now();
        runtimes[run] = (std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());

    }
    // #########################
    std::sort(runtimes.begin(), runtimes.end());

    cudaMemcpyAsync(C, gpu_C, (N * N) * sizeof(dtype), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::move(C, C+N*N, result1);
    // run the non-fused version
    std::cout << "Running Fused Kernels" << std::endl;
    cudaMemcpyAsync(gpu_A, A, (N * N) * sizeof(dtype), cudaMemcpyHostToDevice, stream);
    // ##########################
    runtimes.empty();
    for(int run=0; run<TREPS; run++){
        auto start = std::chrono::high_resolution_clock::now();
        run_fused(gpu_A, gpu_C, N, stream);
        auto end = std::chrono::high_resolution_clock::now();
        runtimes[run] = (std::chrono::duration_cast<std::chrono::microseconds>(end-start).count());
    }
    // ##########################

    std::sort(runtimes.begin(), runtimes.end());

    cudaMemcpyAsync(C, gpu_C, (N * N) * sizeof(dtype), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::move(C, C+N*N, result2);

    cudaFree(gpu_A);
    cudaFree(gpu_C);
    std::cout << "Done." << std::endl;
    std::cout << "Correctness Check" << std::endl;

    bool correct = true;
    dtype norm2_baseline = 0;
    dtype norm2_fused = 0;

    double tol = 1e-5;
    for(int i=0; i<N; ++i){
        for(int j=0; j<N; ++j){
            norm2_baseline += result1[i*N+j] * result1[i*N+j];
            norm2_fused += result2[i*N+j] * result2[i*N+j];
            if(std::abs(result1[i*N+j] - result2[i*N+j]) > tol){
                correct = false;
                break;
            }
        }
    }

    std::cout << "Evaluation     = "     << correct << std::endl;
    std::cout << "Norm2 Baseline = " << norm2_baseline << std::endl;
    std::cout << "Norm2 Fused    = "      << norm2_fused << std::endl;
}

int main(){
    run();
}
