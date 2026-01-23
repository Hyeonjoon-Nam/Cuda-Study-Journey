#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>

// ----------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------
#define SOFTENING 1e-9f      // Prevents division by zero
const int BLOCK_SIZE = 256;  // Threads per block

// ----------------------------------------------------------------
// [Device] Core Physics: Body-Body Interaction
// Calculates the gravitational acceleration exerted by body 'bj' on body 'bi'
// ----------------------------------------------------------------
__device__ float3 bodyBodyInteraction(float3 ai, float4 bi, float4 bj) {
    float3 r;
    // 1. Distance vector r_ij
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // 2. Distance squared + Softening factor
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING;

    // 3. Inverse distance (Fast hardware approximation: rsqrtf)
    float invDist = rsqrtf(distSqr);
    float invDistCube = invDist * invDist * invDist;

    // 4. Accumulate acceleration: a += G * m_j * r / dist^3
    float s = bj.w * invDistCube;

    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}

// ----------------------------------------------------------------
// [Kernel 1] Naive Implementation
// Strategy: Each thread computes force for 1 body by reading N bodies from Global Memory.
// Bottleneck: Severe Global Memory bandwidth saturation (Memory Bound).
// ----------------------------------------------------------------
__global__ void bodyForceNaive(float4* p, float4* v, float dt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 myPos = p[i];
    float3 acc = { 0.0f, 0.0f, 0.0f };

    for (int j = 0; j < n; j++) {
        float4 otherPos = p[j]; // Heavy Global Memory Access
        acc = bodyBodyInteraction(acc, myPos, otherPos);
    }

    // Physics Update (Euler Integration)
    float4 myVel = v[i];
    myVel.x += acc.x * dt;
    myVel.y += acc.y * dt;
    myVel.z += acc.z * dt;

    myPos.x += myVel.x * dt;
    myPos.y += myVel.y * dt;
    myPos.z += myVel.z * dt;

    p[i] = myPos;
    v[i] = myVel;
}

// ----------------------------------------------------------------
// [Kernel 2] Shared Memory Tiling
// Strategy: Load a tile of bodies into Shared Memory to reuse data.
// Benefit: Reduces Global Memory traffic by a factor of BLOCK_SIZE.
// ----------------------------------------------------------------
__global__ void bodyForceShared(float4* p, float4* v, float dt, int n) {
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float4 myPos = { 0, 0, 0, 0 };
    if (i < n) myPos = p[i];

    float3 acc = { 0.0f, 0.0f, 0.0f };

    __shared__ float4 s_pos[BLOCK_SIZE];

    // Loop over all tiles
    for (int tile = 0; tile < gridDim.x; tile++) {
        int idx = tile * blockDim.x + tid; // Global index to load

        // Collaborative loading into Shared Memory
        if (idx < n) s_pos[tid] = p[idx];
        else         s_pos[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        __syncthreads(); // Wait for loading

        // Compute interaction with cached tile
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; j++) {
            acc = bodyBodyInteraction(acc, myPos, s_pos[j]);
        }

        __syncthreads(); // Wait for computation
    }

    // Write result
    if (i < n) {
        float4 myVel = v[i];

        myVel.x += acc.x * dt;
        myVel.y += acc.y * dt;
        myVel.z += acc.z * dt;

        myPos.x += myVel.x * dt;
        myPos.y += myVel.y * dt;
        myPos.z += myVel.z * dt;

        p[i] = myPos;
        v[i] = myVel;
    }
}

// ----------------------------------------------------------------
// [Kernel 3] Thread Coarsening (Factor 2)
// Strategy: Each thread processes 2 bodies to hide instruction latency.
// Note: Requires sufficient N to fill the GPU (Occupancy trade-off).
// ----------------------------------------------------------------
__global__ void bodyForceThreadCoarsening(float4* p, float4* v, float dt, int n) {
    int tid = threadIdx.x;
    int bDim = blockDim.x;

    // Process 2 elements per thread
    int iStart = (blockIdx.x * bDim + tid) * 2;

    float4 myPos0, myPos1;
    float3 acc0 = { 0.0f, 0.0f, 0.0f };
    float3 acc1 = { 0.0f, 0.0f, 0.0f };

    // Load 2 bodies to registers
    if (iStart < n)     myPos0 = p[iStart];
    else                myPos0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (iStart + 1 < n) myPos1 = p[iStart + 1];
    else                myPos1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    __shared__ float4 s_pos[BLOCK_SIZE];

    // Correct tile count calculation (Input size N is unchanged)
    int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < numTiles; tile++) {
        int idx = tile * bDim + tid;

        // Shared memory loading is same as Kernel 2 (1 element per thread for loading)
        // Note: Block size is unchanged, so we still load 256 elements per tile.
        if (idx < n) s_pos[tid] = p[idx];
        else         s_pos[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        __syncthreads();

        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; j++) {
            float4 otherPos = s_pos[j];
            // Compute 2 interactions per step
            acc0 = bodyBodyInteraction(acc0, myPos0, otherPos);
            acc1 = bodyBodyInteraction(acc1, myPos1, otherPos);
        }

        __syncthreads();
    }

    // Write back 2 results
    if (iStart < n) {
        float4 pVal = myPos0;
        float4 vVal = v[iStart];
        
        vVal.x += acc0.x * dt; vVal.y += acc0.y * dt; vVal.z += acc0.z * dt;
        pVal.x += vVal.x * dt; pVal.y += vVal.y * dt; pVal.z += vVal.z * dt;
        
        p[iStart] = pVal; v[iStart] = vVal;
    }

    if (iStart + 1 < n) {
        float4 pVal = myPos1;
        float4 vVal = v[iStart + 1];
        
        vVal.x += acc1.x * dt; vVal.y += acc1.y * dt; vVal.z += acc1.z * dt;
        pVal.x += vVal.x * dt; pVal.y += vVal.y * dt; pVal.z += vVal.z * dt;
        
        p[iStart + 1] = pVal; v[iStart + 1] = vVal;
    }
}

void randomizeBodies(float4* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i].x = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        data[i].y = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        data[i].z = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
        data[i].w = (rand() / (float)RAND_MAX) + 0.1f;
    }
}

int main() {
    // [Experiment] Try increasing N to 20480 or 40960 to see Kernel 3 benefits!
    const int N = 81920;      
    const float dt = 0.01f;
    const int ITERATIONS = 200;

    size_t bytes = N * sizeof(float4);

    float4* h_p = (float4*)malloc(bytes);
    float4* h_v = (float4*)malloc(bytes);

    randomizeBodies(h_p, N);
    for (int i = 0; i < N; i++) h_v[i] = { 0.0f, 0.0f, 0.0f, 0.0f };

    float4* d_p, * d_v;
    cudaMalloc(&d_p, bytes);
    cudaMalloc(&d_v, bytes);
    cudaMemcpy(d_p, h_p, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

    int nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::cout << ">>> Simulating " << N << " bodies for " << ITERATIONS << " iterations..." << std::endl;
    std::cout << ">>> GPU: RTX 3070 Laptop (High Performance)" << std::endl;

    // Common variables for timing
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    double interactions = (double)N * (double)N;
    double totalOps = interactions * (double)ITERATIONS * 20.0;
    double gflops = 0.0;

    // ---------------------------------------------------------
    // 1. Kernel 1: Naive
    // ---------------------------------------------------------
    std::cout << "\n>>> [Kernel 1] Naive Implementation" << std::endl;
    bodyForceNaive<<<nBlocks, BLOCK_SIZE>>>(d_p, d_v, dt, N); // Warm-up
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int iter = 0; iter < ITERATIONS; iter++) {
        bodyForceNaive<<<nBlocks, BLOCK_SIZE>>>(d_p, d_v, dt, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    gflops = (totalOps * 1e-9) / (milliseconds / 1000.0);
    std::cout << ">>> Time: " << milliseconds << " ms" << std::endl;
    std::cout << ">>> Performance: " << gflops << " GFLOPS" << std::endl;

    // ---------------------------------------------------------
    // 2. Kernel 2: Shared Memory Tiling
    // ---------------------------------------------------------
    std::cout << "\n>>> [Kernel 2] Shared Memory Tiling" << std::endl;
    // Reset data (optional, skipping for speed)
    bodyForceShared<<<nBlocks, BLOCK_SIZE>>>(d_p, d_v, dt, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int iter = 0; iter < ITERATIONS; iter++) {
        bodyForceShared<<<nBlocks, BLOCK_SIZE>>>(d_p, d_v, dt, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    gflops = (totalOps * 1e-9) / (milliseconds / 1000.0);
    std::cout << ">>> Time: " << milliseconds << " ms" << std::endl;
    std::cout << ">>> Performance: " << gflops << " GFLOPS" << std::endl;

    // ---------------------------------------------------------
    // 3. Kernel 3: Thread Coarsening (Factor 2)
    // ---------------------------------------------------------
    std::cout << "\n>>> [Kernel 3] Thread Coarsening (Factor 2)" << std::endl;
    // Block count is halved because each thread handles 2 bodies
    int nBlocksCoarsened = (N + (BLOCK_SIZE * 2) - 1) / (BLOCK_SIZE * 2);

    bodyForceThreadCoarsening<<<nBlocksCoarsened, BLOCK_SIZE>>>(d_p, d_v, dt, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int iter = 0; iter < ITERATIONS; iter++) {
        bodyForceThreadCoarsening<<<nBlocksCoarsened, BLOCK_SIZE>>>(d_p, d_v, dt, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    gflops = (totalOps * 1e-9) / (milliseconds / 1000.0);
    std::cout << ">>> Time: " << milliseconds << " ms" << std::endl;
    std::cout << ">>> Performance: " << gflops << " GFLOPS" << std::endl;

    // ---------------------------------------------------------
    // Output & Cleanup
    // ---------------------------------------------------------
    std::cout << "\n>>> Simulation Complete!" << std::endl;
    
    cudaMemcpy(h_p, d_p, bytes, cudaMemcpyDeviceToHost);
    std::cout << ">>> Writing 'nbody_result.csv'..." << std::endl;
    std::ofstream outFile("nbody_result.csv");
    outFile << "x,y,z,w\n";
    for (int i = 0; i < N; i++) {
        outFile << h_p[i].x << "," << h_p[i].y << "," << h_p[i].z << "," << h_p[i].w << "\n";
    }
    outFile.close();
    std::cout << ">>> Done." << std::endl;

    free(h_p); free(h_v);
    cudaFree(d_p); cudaFree(d_v);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}