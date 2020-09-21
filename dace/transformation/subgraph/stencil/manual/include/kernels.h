typedef float dtype;

inline int int_ceil(int N, int D){
    return int((N+D-1)/D);
}

void run_fused(const dtype * __restrict__ gpu_A, dtype * __restrict__ gpu_C, int N, cudaStream_t stream);
void run_kernel1(const dtype * __restrict__ gpu_A, dtype * __restrict__ gpu_B, int N, cudaStream_t stream);
void run_kernel2(const dtype * __restrict__ gpu_B, dtype * __restrict__ gpu_C, int N, cudaStream_t stream);
