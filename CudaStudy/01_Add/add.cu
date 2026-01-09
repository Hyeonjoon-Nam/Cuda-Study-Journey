#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <math.h>

// =================================================================================
// Kernel v1: Single Thread
// ---------------------------------------------------------------------------------
// - Runs on a single thread (Block 1, Thread 1).
// - Simulates CPU sequential execution on the GPU.
// - Performance: Extremely slow due to lack of parallelism.
// =================================================================================
__global__
void add_v1_single_thread(int n, float* x, float* y)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] + y[i];
}

// =================================================================================
// Kernel v2: Single Block (Basic Parallelism)
// ---------------------------------------------------------------------------------
// - Uses multiple threads within a SINGLE block.
// - Demonstrates basic thread indexing (threadIdx.x).
// - Limitation: Can only utilize one Streaming Multiprocessor (SM).
// =================================================================================
__global__
void add_v2_single_block(int n, float* x, float* y)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

// =================================================================================
// Kernel v3: Grid-Stride Loop (Best Practice)
// ---------------------------------------------------------------------------------
// - Uses Multiple Blocks and Multiple Threads.
// - "Grid-Stride Loop" pattern allows the kernel to handle any data size (N),
//   regardless of the grid dimensions.
// - Scalability: Fully utilizes the GPU hardware.
// =================================================================================
__global__
void add_v3_grid_stride(int n, float* x, float* y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1 << 20;
    float* x, * y;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // ---------------------------------------------------------
    // Profile Target Selection (Uncomment one to test)
    // ---------------------------------------------------------

    // [v1] Single Thread
    // add_v1_single_thread<<<1, 1 >>>(N, x, y);

    // [v2] Single Block
    // add_v2_single_block<<<1, 256>>>(N, x, y);

    // [v3] Grid-Stride Loop
    add_v3_grid_stride<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i] - 3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}