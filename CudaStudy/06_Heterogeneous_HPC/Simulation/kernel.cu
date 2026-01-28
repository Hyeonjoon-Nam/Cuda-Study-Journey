#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include "kernel.cuh"
#include <math_constants.h>
#include <stdio.h>

// Kernel: Calculates the sine wave position directly into the OpenGL buffer
__global__ void simple_vbo_kernel(float4* pos, int width, int height, float time) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate 1D index for the flat array
    unsigned int idx = y * width + x;
    if (idx >= width * height) return;

    // Normalize coordinates to 0.0 ~ 1.0
    float u = x / (float)width;
    float v = y / (float)height;

    // Calculate Sine Wave (Simple animation logic)
    float freq = 4.0f;
    float w = sinf(u * freq + time) * cosf(v * freq + time) * 0.5f;

    // Write directly to VBO memory
    // Mapping: u -> x (-1.0 ~ 1.0), w -> y (height), v -> z (-1.0 ~ 1.0)
    pos[idx] = make_float4(u * 2.0f - 1.0f, w, v * 2.0f - 1.0f, 1.0f);
}

// 1. Register OpenGL Buffer to CUDA
void initCuda(cudaGraphicsResource** vbo_resource, unsigned int vbo, int num_particles) {
    cudaError_t err;
    cudaSetDevice(0);

    // Key Function: Registers the OpenGL buffer (VBO) for CUDA access.
    // 'cudaGraphicsMapFlagsWriteDiscard': CUDA will overwrite everything, so don't preserve previous content.
    err = cudaGraphicsGLRegisterBuffer(vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess) printf("CUDA Register Error: %s\n", cudaGetErrorString(err));
}

// 2. Map Resource -> Get Pointer -> Run Kernel -> Unmap
void runCuda(cudaGraphicsResource* vbo_resource, int num_particles, float time) {
    cudaError_t err;
    float4* dptr;
    size_t num_bytes;

    // Step A: Map the resource. This locks the buffer for CUDA use (OpenGL cannot touch it now).
    err = cudaGraphicsMapResources(1, &vbo_resource, 0);
    if (err != cudaSuccess) printf("Map Error: %s\n", cudaGetErrorString(err));

    // Step B: Get the device pointer (The actual GPU memory address of the VBO).
    err = cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, vbo_resource);
    if (err != cudaSuccess) printf("Get Pointer Error: %s\n", cudaGetErrorString(err));

    // Step C: Launch Kernel
    dim3 block(16, 16);
    dim3 grid(64, 64); // 64 * 16 = 1024 (Total 1024x1024 threads)
    simple_vbo_kernel << <grid, block >> > (dptr, 1024, 1024, time);

    // Step D: Unmap. Unlock the buffer so OpenGL can use it for rendering.
    cudaGraphicsUnmapResources(1, &vbo_resource, 0);
}

void cleanupCuda(cudaGraphicsResource* vbo_resource) {
    cudaGraphicsUnregisterResource(vbo_resource);
}