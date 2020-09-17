
inline int int_ceil(int N, int D){
    return int((N+D-1)/D);
}

void run_fused(const double * __restrict__ gpu_A, double * __restrict__ gpu_C, int N, cudaStream_t stream);
void run_kernel1(const double * __restrict__ gpu_A, double * __restrict__ gpu_B, int N, cudaStream_t stream);
void run_kernel2(const double * __restrict__ gpu_B, double * __restrict__ gpu_C, int N, cudaStream_t stream);
