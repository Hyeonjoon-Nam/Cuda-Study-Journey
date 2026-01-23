#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

// Configuration
const int N = 1 << 20; // 1 Million elements (2^20)
const int BLOCK_SIZE = 256;

// Utility: Random Initialization
void randomInit(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (rand() & 0xFF) / (float)255.0f;
    }
}

// ----------------------------------------------------------------------
// Kernel 0: Naive Interleaved Addressing
// [Goal] Baseline implementation of parallel reduction.
// [Problem] Highly divergent warps and inefficient memory access patterns.
// - "Interleaved" addressing implies stride doubles at each step (1, 2, 4...).
// - This causes threads within a warp to diverge (some active, some idle).
// ----------------------------------------------------------------------
__global__ void reduce0(float* g_idata, float* g_odata) {
    __shared__ float sdata[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input from Global Memory to Shared Memory
    sdata[tid] = g_idata[i];
    __syncthreads();

    // Reduction in Shared Memory
    // Problem: In the first iteration, only even threads (0, 2, 4...) are active.
    // This leads to >50% warp divergence, wasting GPU cycles.
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to Global Memory
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ----------------------------------------------------------------------
// Kernel 1: Sequential Addressing
// [Optimization] Solve Warp Divergence.
// [Technique] Change indexing to force adjacent threads to be active.
// - Instead of strided access, threads 0 to (blockDim/2) are active.
// - This keeps entire warps active while others are fully idle (skipped).
// ----------------------------------------------------------------------
__global__ void reduce1(float* g_idata, float* g_odata) {
    extern __shared__ float sdata[]; // Dynamic Shared Memory size

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = g_idata[i];
    __syncthreads();

    // Sequential Indexing:
    // Iteration 1: Threads 0-127 active (Warp 0-3 fully active).
    // Iteration 2: Threads 0-63 active (Warp 0-1 fully active).
    // No divergence within active warps.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ----------------------------------------------------------------------
// Device Function: Warp Unrolling
// [Optimization] Remove synchronization overhead for the last 32 threads.
// [Note] 'volatile' is crucial here!
// - It prevents the compiler from caching values in registers.
// - Ensures threads read the latest values written by other threads in Shared Mem.
// ----------------------------------------------------------------------
__device__ void warpReduce(volatile float* sdata, int tid) {
    // Implicit Synchronization: Threads in a warp execute in lockstep.
    // We unroll the loop to avoid loop overhead and __syncthreads().
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

// ----------------------------------------------------------------------
// Kernel 2: Loop Unrolling
// [Optimization] Reduce Instruction Overhead.
// [Technique] Unroll the last 6 iterations (when s <= 32).
// - When fewer than 32 threads are active, we are within a single warp.
// - We can skip __syncthreads() and loop checks, saving instructions.
// ----------------------------------------------------------------------
__global__ void reduce2(float* g_idata, float* g_odata) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = g_idata[i];
    __syncthreads();

    // Standard reduction until we hit the warp boundary (32 threads)
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Unrolled reduction for the last warp
    if (tid < 32) {
        warpReduce(sdata, tid);
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ----------------------------------------------------------------------
// Kernel 3: Algorithmic Cascading (Grid Halving)
// [Optimization] Increase Instruction Level Parallelism (ILP).
// [Technique] Each thread loads and adds 2 elements during the load phase.
// - Reduces the grid size by half (half the blocks needed).
// - Hides memory latency by issuing multiple load instructions.
// ----------------------------------------------------------------------
__global__ void reduce3(float* g_idata, float* g_odata) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    // Each block processes 2x BLOCK_SIZE elements
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    // Perform the first addition while loading from Global Memory
    // This reduces the reduction tree height and amortization overhead.
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];

    __syncthreads();

    // Same unrolled reduction loop as reduce2
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(sdata, tid);
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// ----------------------------------------------------------------------
// Kernel 4: Grid-Stride Loop (Final Optimized Version)
// [Optimization] Full Scalability & Register Accumulation.
// [Technique] Decouple Grid Size from Data Size (N).
// - A fixed number of blocks handle arbitrary N by looping (Grid Stride).
// - Accumulates intermediate sums in a Register (fastest) before writing to Shared Mem.
// ----------------------------------------------------------------------
__global__ void reduce4(float* g_idata, float* g_odata, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid Stride: The total number of threads in the grid
    unsigned int gridSize = blockDim.x * gridDim.x;

    // Use a register for accumulation (faster than repeatedly writing to shared mem)
    float sum = 0.0f;

    // Grid-Stride Loop:
    // Process multiple elements per thread if N > TotalThreads
    while (i < n) {
        sum += g_idata[i];
        i += gridSize;
    }

    // Write the accumulated register value to Shared Memory
    sdata[tid] = sum;
    __syncthreads();

    // Standard Reduction
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(sdata, tid);
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
    std::cout << "=== Project 03: Parallel Reduction (N=" << N << ") ===" << std::endl;

    // 1. Setup Host Memory
    size_t bytes = N * sizeof(float);
    std::vector<float> h_input(N);
    std::vector<float> h_output(N / BLOCK_SIZE);

    randomInit(h_input.data(), N);

    // Golden Reference (CPU Calculation)
    float cpu_sum = std::accumulate(h_input.begin(), h_input.end(), 0.0f);
    std::cout << "CPU Sum: " << cpu_sum << std::endl;

    // 2. Setup Device Memory
    float* d_input, * d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, (N / BLOCK_SIZE) * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE);
    dim3 grid(N / BLOCK_SIZE);

    // -------------------------------------------------------
    // Kernel 0: Naive
    // -------------------------------------------------------
    std::cout << "\n[Kernel 0: Naive Interleaved] Running..." << std::endl;
    reduce0<<<grid, block>>>(d_input, d_output);
    cudaDeviceSynchronize();

    std::vector<float> h_partial_sums(N / BLOCK_SIZE);
    cudaMemcpy(h_partial_sums.data(), d_output, (N / BLOCK_SIZE) * sizeof(float), cudaMemcpyDeviceToHost);
    float gpu_sum0 = std::accumulate(h_partial_sums.begin(), h_partial_sums.end(), 0.0f);

    std::cout << "GPU Sum: " << gpu_sum0 << std::endl;
    if (std::abs(cpu_sum - gpu_sum0) < 1.0f) std::cout << "-> Test PASSED" << std::endl;
    else std::cout << "-> Test FAILED" << std::endl;

    // -------------------------------------------------------
    // Kernel 1: Sequential Addressing
    // -------------------------------------------------------
    std::cout << "\n[Kernel 1: Sequential Addressing] Running..." << std::endl;
    reduce1<<<grid, block, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(h_partial_sums.data(), d_output, (N / BLOCK_SIZE) * sizeof(float), cudaMemcpyDeviceToHost);
    float gpu_sum1 = std::accumulate(h_partial_sums.begin(), h_partial_sums.end(), 0.0f);

    std::cout << "GPU Sum: " << gpu_sum1 << std::endl;
    if (std::abs(cpu_sum - gpu_sum1) < 1.0f) std::cout << "-> Test PASSED" << std::endl;
    else std::cout << "-> Test FAILED" << std::endl;

    // -------------------------------------------------------
    // Kernel 2: Loop Unrolling
    // -------------------------------------------------------
    std::cout << "\n[Kernel 2: Loop Unrolling] Running..." << std::endl;
    reduce2<<<grid, block, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(h_partial_sums.data(), d_output, (N / BLOCK_SIZE) * sizeof(float), cudaMemcpyDeviceToHost);
    float gpu_sum2 = std::accumulate(h_partial_sums.begin(), h_partial_sums.end(), 0.0f);

    std::cout << "GPU Sum: " << gpu_sum2 << std::endl;
    if (std::abs(cpu_sum - gpu_sum2) < 1.0f) std::cout << "-> Test PASSED" << std::endl;
    else std::cout << "-> Test FAILED" << std::endl;

    // -------------------------------------------------------
    // Kernel 3: Algorithmic Cascading (Multiple Adds)
    // -------------------------------------------------------
    std::cout << "\n[Kernel 3: Multiple Adds] Running..." << std::endl;
    dim3 grid3(N / (BLOCK_SIZE * 2)); // Halved Grid Size

    reduce3<<<grid3, block, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output);
    cudaDeviceSynchronize();

    int output_count3 = N / (BLOCK_SIZE * 2);
    cudaMemcpy(h_partial_sums.data(), d_output, output_count3 * sizeof(float), cudaMemcpyDeviceToHost);
    float gpu_sum3 = std::accumulate(h_partial_sums.begin(), h_partial_sums.begin() + output_count3, 0.0f);

    std::cout << "GPU Sum: " << gpu_sum3 << std::endl;
    if (std::abs(cpu_sum - gpu_sum3) < 1.0f) std::cout << "-> Test PASSED" << std::endl;
    else std::cout << "-> Test FAILED" << std::endl;

    // -------------------------------------------------------
    // Kernel 4: Grid-Stride Loop
    // -------------------------------------------------------
    std::cout << "\n[Kernel 4: Grid-Stride Loop] Running..." << std::endl;
    int blocks = 2048; // Tunable parameter
    dim3 grid4(blocks);

    reduce4<<<grid4, block, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_partial_sums.data(), d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    float gpu_sum4 = std::accumulate(h_partial_sums.begin(), h_partial_sums.begin() + blocks, 0.0f);

    std::cout << "GPU Sum: " << gpu_sum4 << std::endl;
    if (std::abs(cpu_sum - gpu_sum4) < 1.0f) std::cout << "-> Test PASSED" << std::endl;
    else std::cout << "-> Test FAILED" << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}