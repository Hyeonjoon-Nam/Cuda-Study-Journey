#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h" 
#include <iostream>
#include <vector>
#include <cmath>

// Configuration
const int N = 4096; // Large enough to stress Global Memory
const int BLOCK_SIZE = 16;

// Utility: Random Initialization
void randomInit(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

// Kernel 1: Naive Implementation
// [Goal] Baseline comparison.
// [Bottleneck] Global Memory Latency. Every arithmetic operation triggers a slow memory access.
__global__ void matrixMulNaive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Kernel 2: Shared Memory Tiling
// [Goal] Reduce Global Memory access by reusing data.
// [Technique] Loads data into Shared Memory (On-chip L1 Cache) and reuses it BLOCK_SIZE times.
__global__ void matrixMulShared(const float* A, const float* B, float* C, int n) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.f;

    for (int t = 0; t < (n / BLOCK_SIZE); t++) {
        // Cooperative Loading: Each thread loads one element
        s_A[threadIdx.y][threadIdx.x] = A[row * n + (t * BLOCK_SIZE + threadIdx.x)];
        s_B[threadIdx.y][threadIdx.x] = B[(t * BLOCK_SIZE + threadIdx.y) * n + col];
        
        __syncthreads(); // Barrier: Wait for loading

        // Compute using fast Shared Memory
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads(); // Barrier: Wait for computation
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Kernel 3: Shared Memory + Vectorized Loading (float4)
// [Goal] Maximize Memory Bandwidth efficiency.
// [Technique] Loads 4 floats (16 bytes) in a single instruction instead of 4 separate loads.
// [Note] Shifts bottleneck from Memory to Compute.
__global__ void matrixMulSharedOptimized(const float* A, const float* B, float* C, int n) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    // Reinterpret pointers for float4 access (Shared Memory)
    float4* s_A_f4 = reinterpret_cast<float4*>(&s_A[0][0]);
    float4* s_B_f4 = reinterpret_cast<float4*>(&s_B[0][0]);

    // Reinterpret pointers for float4 access (Global Memory)
    const float4* A_f4 = reinterpret_cast<const float4*>(A);
    const float4* B_f4 = reinterpret_cast<const float4*>(B);

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.f;

    // Linear thread ID for role splitting
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int t = 0; t < (n / BLOCK_SIZE); t++) {
        // Vectorized Loading Phase
        // Threads 0-63 load Matrix A (since 256 floats = 64 float4s)
        if (tid < (BLOCK_SIZE * BLOCK_SIZE / 4)) {
            int local_r = tid / (BLOCK_SIZE / 4);
            int local_c = tid % (BLOCK_SIZE / 4);
            int global_r = (blockIdx.y * BLOCK_SIZE) + local_r;
            int global_c = (t * BLOCK_SIZE / 4) + local_c;

            s_A_f4[tid] = A_f4[global_r * (n / 4) + global_c];
        }

        // Threads 64-127 load Matrix B
        if (tid >= 64 && tid < 128) {
            int b_tid = tid - 64;
            int local_r = b_tid / (BLOCK_SIZE / 4);
            int local_c = b_tid % (BLOCK_SIZE / 4);
            int global_r = (t * BLOCK_SIZE) + local_r;
            int global_c = (blockIdx.x * BLOCK_SIZE / 4) + local_c;

            s_B_f4[b_tid] = B_f4[global_r * (n / 4) + global_c];
        }

        __syncthreads();

        // Compute Phase (Standard float arithmetic)
        for (int k = 0; k < BLOCK_SIZE; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

int main() {
    std::cout << "=== Matrix Multiplication (N=" << N << ") ===" << std::endl;

    // 1. Allocation & Setup
    size_t bytes = N * N * sizeof(float);
    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C(N * N); 

    randomInit(h_A.data(), N * N);
    randomInit(h_B.data(), N * N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0;

    // =========================================================
    // BENCHMARK SECTION (Uncomment one by one for Nsight)
    // =========================================================

    // 1. Naive Kernel
    
    std::cout << "\n[Naive Kernel]" << std::endl;
    matrixMulNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N); // Warm-up
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    matrixMulNaive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Time: " << ms << " ms" << std::endl;
    

    // 2. Shared Memory Kernel
    
    std::cout << "\n[Shared Memory Kernel]" << std::endl;
    matrixMulShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N); // Warm-up
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    matrixMulShared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Time: " << ms << " ms" << std::endl;
    

    // 3. Shared + Optimization (float4)
    
    std::cout << "\n[Shared Memory + float4 Optimization]" << std::endl;
    matrixMulSharedOptimized<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N); // Warm-up
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    matrixMulSharedOptimized<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Time: " << ms << " ms" << std::endl;
    

    // 4. cuBLAS (Official Library)
    
    std::cout << "\n[cuBLAS Library]" << std::endl;
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    // Warm-up
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms_cublas = 0;
    cudaEventElapsedTime(&ms_cublas, start, stop);
    std::cout << "Time: " << ms_cublas << " ms" << std::endl;
    cublasDestroy(handle);
    

    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}