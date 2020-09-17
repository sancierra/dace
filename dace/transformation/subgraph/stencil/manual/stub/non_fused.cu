
#include <cuda_runtime.h>
#include <dace/dace.h>



DACE_EXPORTED int __dace_init_cuda(double * __restrict__ A, double * __restrict__ C, int N);
DACE_EXPORTED void __dace_exit_cuda(double * __restrict__ A, double * __restrict__ C, int N);



namespace dace { namespace cuda {
    cudaStream_t __streams[1];
    cudaEvent_t __events[1];
    int num_streams = 1;
    int num_events = 1;
} }

int __dace_init_cuda(double * __restrict__ A, double * __restrict__ C, int N) {
    int count;

    // Check that we are able to run cuda code
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        printf("ERROR: GPU drivers are not configured or cuda-capable device "
               "not found\n");
        return 1;
    }
    if (count == 0)
    {
        printf("ERROR: No cuda-capable devices found\n");
        return 2;
    }

    // Initialize cuda before we run the application
    float *dev_X;
    cudaMalloc((void **) &dev_X, 1);

    // Create cuda streams and events
    for(int i = 0; i < 1; ++i) {
        cudaStreamCreateWithFlags(&dace::cuda::__streams[i], cudaStreamNonBlocking);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventCreateWithFlags(&dace::cuda::__events[i], cudaEventDisableTiming);
    }

    

    return 0;
}

void __dace_exit_cuda(double * __restrict__ A, double * __restrict__ C, int N) {
    

    // Destroy cuda streams and events
    for(int i = 0; i < 1; ++i) {
        cudaStreamDestroy(dace::cuda::__streams[i]);
    }
    for(int i = 0; i < 1; ++i) {
        cudaEventDestroy(dace::cuda::__events[i]);
    }
}

__global__ void a_0_0_2(double * __restrict__ B, const double * __restrict__ gpu_A, int N) {
    {
        {
            int j = ((blockIdx.x * 32 + threadIdx.x) + 1);
            int i = ((blockIdx.y * 1 + threadIdx.y) + 1);
            if (j >= 1 && j < (N - 1)) {
                if (i >= 1) {
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

                        B[((N * i) + j)] = b;
                    }
                }
            }
        }
    }
}


DACE_EXPORTED void __dace_runkernel_a_0_0_2(double * __restrict__ B, const double * __restrict__ gpu_A, int N);
void __dace_runkernel_a_0_0_2(double * __restrict__ B, const double * __restrict__ gpu_A, int N)
{

    void  *a_0_0_2_args[] = { (void *)&B, (void *)&gpu_A, (void *)&N };
    cudaLaunchKernel((void*)a_0_0_2, dim3(int_ceil(int_ceil((N - 2), 1), 32), int_ceil(int_ceil((N - 2), 1), 1), int_ceil(1, 1)), dim3(32, 1, 1), a_0_0_2_args, 0, dace::cuda::__streams[0]);
}
__global__ void b_0_0_5(const double * __restrict__ B, double * __restrict__ gpu_C, int N) {
    {
        {
            int j = ((blockIdx.x * 32 + threadIdx.x) + 2);
            int i = ((blockIdx.y * 1 + threadIdx.y) + 2);
            if (j >= 2 && j < (N - 2)) {
                if (i >= 2) {
                    {
                        double a1 = B[((N * i) + j)];
                        double a2 = B[(((N * i) + j) - 1)];
                        double a3 = B[(((N * i) + j) + 1)];
                        double a4 = B[((N * (i + 1)) + j)];
                        double a5 = B[((N * (i - 1)) + j)];
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


DACE_EXPORTED void __dace_runkernel_b_0_0_5(const double * __restrict__ B, double * __restrict__ gpu_C, int N);
void __dace_runkernel_b_0_0_5(const double * __restrict__ B, double * __restrict__ gpu_C, int N)
{

    void  *b_0_0_5_args[] = { (void *)&B, (void *)&gpu_C, (void *)&N };
    cudaLaunchKernel((void*)b_0_0_5, dim3(int_ceil(int_ceil((N - 4), 1), 32), int_ceil(int_ceil((N - 4), 1), 1), int_ceil(1, 1)), dim3(32, 1, 1), b_0_0_5_args, 0, dace::cuda::__streams[0]);
}

