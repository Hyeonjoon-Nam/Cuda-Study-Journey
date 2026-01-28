#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include "kernel.cuh"
#include <math_constants.h>
#include <stdio.h>

// Global variable to store velocity on GPU (Device Memory)
// This is not managed by OpenGL, only by CUDA.
float4* dev_vel = nullptr;

// ------------------------------------------------------------------
// Helper: Simple Hash Function for Randomness
// Generates a deterministic random float (0.0 ~ 1.0) based on seed.
// ------------------------------------------------------------------
__device__ float random(unsigned int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return (float)(seed % 10000) / 10000.0f;
}

// ------------------------------------------------------------------
// Vector Helper Functions
// ------------------------------------------------------------------
__device__ float4 add(float4 a, float4 b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0.0f);
}

__device__ float4 sub(float4 a, float4 b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0.0f);
}

__device__ float4 mult(float4 a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, 0.0f);
}

__device__ float length(float4 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

// ------------------------------------------------------------------
// Kernel: Initialization
// Sets initial positions and velocities using hash-based randomness.
// ------------------------------------------------------------------
__global__ void init_particles_kernel(float4* pos, float4* vel, int num_particles, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Initialize Position: Random range -1.0 ~ 1.0
    float rx = random(idx * seed) * 2.0f - 1.0f;
    float ry = random(idx * seed + 1234) * 2.0f - 1.0f;
    
    // w must be 1.0f for OpenGL rendering
    pos[idx] = make_float4(rx, ry, 0.0f, 1.0f); 

    // Initialize Velocity: Small random vector
    float vx = (random(idx * seed + 5678) - 0.5f) * 0.01f;
    float vy = (random(idx * seed + 9999) - 0.5f) * 0.01f;
    vel[idx] = make_float4(vx, vy, 0.0f, 0.0f);
}

// ------------------------------------------------------------------
// Kernel: Naive Boids Simulation (O(N^2))
// Implements Reynolds' Boids rules: Cohesion, Separation, Alignment.
// ------------------------------------------------------------------
__global__ void boids_kernel(float4* pos, float4* vel, int num_particles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    float4 my_pos = pos[idx];
    float4 my_vel = vel[idx];

    // Force Accumulators
    float4 cohesion = make_float4(0, 0, 0, 0);
    float4 separation = make_float4(0, 0, 0, 0);
    float4 alignment = make_float4(0, 0, 0, 0);

    int neighbor_count = 0;

    // Simulation Parameters
    float visual_range = 0.2f;       // Radius to detect neighbors
    float protected_range = 0.04f;   // Radius to avoid collision (Separation)
    
    float centering_factor = 0.005f; // Strength of Cohesion
    float avoid_factor = 0.05f;      // Strength of Separation
    float align_factor = 0.05f;      // Strength of Alignment
    
    float max_speed = 0.005f;        // Limit max speed to prevent explosion

    // Naive Loop: Check against ALL other particles (O(N^2) complexity)
    for (int i = 0; i < num_particles; i++) {
        if (i == idx) continue;

        float4 other_pos = pos[i];
        float4 other_vel = vel[i];
        float dist = length(sub(my_pos, other_pos));

        // Check Visual Range
        if (dist < visual_range && dist > 0.0001f) {
            
            // Rule 1: Cohesion (Group together)
            cohesion = add(cohesion, other_pos);
            
            // Rule 3: Alignment (Match velocity)
            alignment = add(alignment, other_vel);
            
            neighbor_count++;

            // Rule 2: Separation (Avoid crowding)
            if (dist < protected_range) {
                float4 push = sub(my_pos, other_pos);
                // Weight by distance (closer = stronger push)
                separation.x += push.x / dist;
                separation.y += push.y / dist;
            }
        }
    }

    // Apply Rules if neighbors exist
    if (neighbor_count > 0) {
        // Cohesion: Steer towards average position
        cohesion.x /= neighbor_count; cohesion.y /= neighbor_count;
        cohesion = sub(cohesion, my_pos); // Vector to center
        cohesion = mult(cohesion, centering_factor);

        // Alignment: Steer towards average velocity
        alignment.x /= neighbor_count; alignment.y /= neighbor_count;
        alignment = mult(alignment, align_factor);

        // Separation: Steer away
        separation = mult(separation, avoid_factor);
    }

    // Update Velocity
    my_vel = add(my_vel, cohesion);
    my_vel = add(my_vel, separation);
    my_vel = add(my_vel, alignment);

    // Limit Speed
    float speed = length(my_vel);
    if (speed > max_speed) {
        my_vel = mult(my_vel, max_speed / speed);
    }

    // Boundary Wrapping (Teleport to opposite side)
    if (my_pos.x > 1.0f) my_pos.x = -1.0f;
    if (my_pos.x < -1.0f) my_pos.x = 1.0f;
    if (my_pos.y > 1.0f) my_pos.y = -1.0f;
    if (my_pos.y < -1.0f) my_pos.y = 1.0f;

    // Update Position
    my_pos = add(my_pos, my_vel);

    // [CRITICAL] Ensure w-component is 1.0 for OpenGL rendering
    my_pos.w = 1.0f;

    // Write back to Global Memory
    pos[idx] = my_pos;
    vel[idx] = my_vel;
}

// ------------------------------------------------------------------
// Host Function: Init CUDA
// Registers OpenGL buffer and allocates CUDA memory.
// ------------------------------------------------------------------
void initCuda(cudaGraphicsResource** vbo_resource, unsigned int vbo, int num_particles) {
    cudaSetDevice(0);
    
    // Register VBO. Use 'FlagsNone' to preserve data (needed for reading prev position).
    cudaGraphicsGLRegisterBuffer(vbo_resource, vbo, cudaGraphicsMapFlagsNone);

    // Allocate memory for velocity
    cudaMalloc((void**)&dev_vel, num_particles * sizeof(float4));

    // Map resource to get device pointer
    float4* dptr;
    size_t num_bytes;
    cudaGraphicsMapResources(1, vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource);

    // Launch Initialization Kernel
    int threads = 256;
    int blocks = (num_particles + threads - 1) / threads;
    init_particles_kernel<<<blocks, threads>>>(dptr, dev_vel, num_particles, 1234UL);

    // Unmap resource
    cudaGraphicsUnmapResources(1, vbo_resource, 0);
}

// ------------------------------------------------------------------
// Host Function: Run CUDA Simulation
// Maps VBO, runs kernel, and unmaps. Called every frame.
// ------------------------------------------------------------------
void runCuda(cudaGraphicsResource* vbo_resource, int num_particles, float time) {
    float4* dptr;
    size_t num_bytes;

    cudaGraphicsMapResources(1, &vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, vbo_resource);

    int threads = 256;
    int blocks = (num_particles + threads - 1) / threads;

    boids_kernel<<<blocks, threads>>>(dptr, dev_vel, num_particles, 0.01f);

    cudaGraphicsUnmapResources(1, &vbo_resource, 0);
}

// ------------------------------------------------------------------
// Host Function: Cleanup
// Frees resources on exit.
// ------------------------------------------------------------------
void cleanupCuda(cudaGraphicsResource* vbo_resource) {
    cudaGraphicsUnregisterResource(vbo_resource);
    if (dev_vel) cudaFree(dev_vel);
}