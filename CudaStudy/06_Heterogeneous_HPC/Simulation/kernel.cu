#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_gl_interop.h>
#include "kernel.cuh"
#include <math_constants.h>
#include <math.h>
#include <stdio.h>

// [Thrust] CUDA Standard Template Library for Sorting
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

// ------------------------------------------------------------------
// Simulation Parameters & Grid Settings
// ------------------------------------------------------------------
#define GRID_SIZE 64        // 64 x 64 Grid
#define NUM_CELLS (GRID_SIZE * GRID_SIZE)
#define WORLD_SIZE 2.0f     // -1.0 to 1.0

// [NOTE] Visual range must be <= Cell Size for 3x3 search to work correctly.
// Cell Size = 2.0 / 64 = 0.03125
__constant__ float c_visualRange = 0.03f; 
__constant__ float c_protectedRange = 0.01f;
__constant__ float c_centeringFactor = 0.005f;
__constant__ float c_avoidFactor = 0.05f;
__constant__ float c_alignFactor = 0.05f;
__constant__ float c_maxSpeed = 0.002f;

// ------------------------------------------------------------------
// Global Memory Buffers (Device)
// ------------------------------------------------------------------
float4* dev_vel = nullptr;          // Particle Velocity
float4* dev_pos_sorted = nullptr;   // Sorted Position (for Cache Coherency)
float4* dev_vel_sorted = nullptr;   // Sorted Velocity

// Grid Data Structures (Changed uint -> unsigned int for Windows Compatibility)
unsigned int* dev_gridParticleHash = nullptr; // Particle's Cell ID
unsigned int* dev_gridParticleIndex = nullptr;// Particle's Original Index
unsigned int* dev_cellStart = nullptr;        // Start index of each cell in sorted list
unsigned int* dev_cellEnd = nullptr;          // End index of each cell in sorted list

// ------------------------------------------------------------------
// Helper Functions
// ------------------------------------------------------------------
__device__ float random(unsigned int seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return (float)(seed % 10000) / 10000.0f;
}

__device__ float4 add(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, 0.0f); }

__device__ float4 sub(float4 a, float4 b) { return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0.0f); }

__device__ float4 mult(float4 a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, 0.0f); }

__device__ float length(float4 v) { return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z); }


// Calculates a repulsive vector (Potential Field) applied by surrounding static obstacles.
// Prevents agents from crossing physical boundaries by generating a force inversely proportional to the distance.
__device__ float2 computeMapRepulsion(float4 pos) {
    // Convert current world coordinates [-1.0, 1.0] to map grid indices [0, 127]
    int gridPos_x = (int)((pos.x + 1.0f) / WORLD_SIZE * MAP_WIDTH);
    int gridPos_y = (int)((pos.y + 1.0f) / WORLD_SIZE * MAP_HEIGHT);

    gridPos_x = max(0, min(gridPos_x, MAP_WIDTH - 1));
    gridPos_y = max(0, min(gridPos_y, MAP_HEIGHT - 1));

    float2 total_repulsion = make_float2(0.0f, 0.0f);

    float repulse_radius = 0.025f;
    float repulse_force_scalar = 0.002f;

    // Search 3x3 neighborhood for wall cells
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int neigh_x = gridPos_x + dx;
            int neigh_y = gridPos_y + dy;

            if (neigh_x < 0 || neigh_x >= MAP_WIDTH || neigh_y < 0 || neigh_y >= MAP_HEIGHT) continue;

            int idx = neigh_y * MAP_WIDTH + neigh_x;
            if (d_map[idx] == 1) // If this cell is a wall
            {
                // Unproject grid indices back to the center of the cell in World Space
                float wall_world_x = ((neigh_x + 0.5f) / MAP_WIDTH) * WORLD_SIZE - 1.0f;
                float wall_world_y = ((neigh_y + 0.5f) / MAP_HEIGHT) * WORLD_SIZE - 1.0f;

                // Calculate directional push vector (from wall to agent)
                float push_x = pos.x - wall_world_x;
                float push_y = pos.y - wall_world_y;

                float dist = sqrtf(push_x * push_x + push_y * push_y);

                // Apply linear falloff force if within the repulsion radius
                if (dist > 0.0001f && dist < repulse_radius) {
                    push_x /= dist; // Normalize direction
                    push_y /= dist;

                    float force_mag = (repulse_radius - dist) / repulse_radius;

                    total_repulsion.x += push_x * force_mag * repulse_force_scalar;
                    total_repulsion.y += push_y * force_mag * repulse_force_scalar;
                }
            }
        }
    }

    return total_repulsion;
}

// Computes the Vector Flow Field force.
// Samples the local 8-way distance gradient to steer the agent towards the global goal in O(1) time.
__device__ float2 computeFlowFieldForce(float4 pos) {
    int gridPos_x = (int)((pos.x + 1.0f) / WORLD_SIZE * MAP_WIDTH);
    int gridPos_y = (int)((pos.y + 1.0f) / WORLD_SIZE * MAP_HEIGHT);

    gridPos_x = max(0, min(gridPos_x, MAP_WIDTH - 1));
    gridPos_y = max(0, min(gridPos_y, MAP_HEIGHT - 1));

    unsigned short current_dist = d_dist_map[gridPos_y * MAP_WIDTH + gridPos_x];

    if (current_dist == 65535 || current_dist == 0) return make_float2(0.0f, 0.0f);

    float2 flow_dir = make_float2(0.0f, 0.0f);

    int min_dist = current_dist;

    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;

            int nx = gridPos_x + dx;
            int ny = gridPos_y + dy;

            if (nx < 0 || nx >= MAP_WIDTH || ny < 0 || ny >= MAP_HEIGHT) continue;

            int idx = ny * MAP_WIDTH + nx;

            if (d_map[idx] == 1) continue;

            if (d_dist_map[idx] < current_dist) {
                float weight = (float)(current_dist - d_dist_map[idx]);

                flow_dir.x += dx * weight;
                flow_dir.y += dy * weight;

                if (d_dist_map[idx] < min_dist) min_dist = d_dist_map[idx];
            }
        }
    }

    if (min_dist == current_dist) return make_float2(0.0f, 0.0f);

    float length = sqrtf(flow_dir.x * flow_dir.x + flow_dir.y * flow_dir.y);
    if (length > 0.0001f) {
        flow_dir.x /= length;
        flow_dir.y /= length;

        float flow_strength = 0.005f;
        flow_dir.x *= flow_strength;
        flow_dir.y *= flow_strength;
    }

    return flow_dir;
}

// Calculate Cell ID (Hash) from Position
__device__ int calcGridHash(float4 pos) {
    // Convert range [-1.0, 1.0] to [0, GRID_SIZE-1]
    int gridPos_x = (int)((pos.x + 1.0f) / WORLD_SIZE * GRID_SIZE);
    int gridPos_y = (int)((pos.y + 1.0f) / WORLD_SIZE * GRID_SIZE);
    
    // Clamp to boundaries
    gridPos_x = max(0, min(gridPos_x, GRID_SIZE - 1));
    gridPos_y = max(0, min(gridPos_y, GRID_SIZE - 1));

    return gridPos_y * GRID_SIZE + gridPos_x; // 1D Index
}

// ------------------------------------------------------------------
// Kernel 1: Calculate Hash & Initialize Index
// ------------------------------------------------------------------
__global__ void calcHashD_kernel(float4* pos, unsigned int* gridParticleHash, unsigned int* gridParticleIndex, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // 1. Calculate Cell ID
    int hash = calcGridHash(pos[idx]);

    // 2. Store Hash and Original Index
    gridParticleHash[idx] = hash;
    gridParticleIndex[idx] = idx;
}

// ------------------------------------------------------------------
// Kernel 2: Reorder Data
// ------------------------------------------------------------------
__global__ void reorderDataD_kernel(unsigned int* sortedIndex, float4* oldPos, float4* oldVel, 
                                    float4* sortedPos, float4* sortedVel, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Get the original index from the sorted array
    unsigned int originalIdx = sortedIndex[idx];

    // Copy data to sorted arrays (Coalesced write)
    sortedPos[idx] = oldPos[originalIdx];
    sortedVel[idx] = oldVel[originalIdx];
}

// ------------------------------------------------------------------
// Kernel 3: Find Cell Start/End
// ------------------------------------------------------------------
__global__ void findCellStartEndD_kernel(unsigned int* gridParticleHash, unsigned int* cellStart, unsigned int* cellEnd, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    unsigned int myHash = gridParticleHash[idx];

    // Load prev hash (handle idx=0 case)
    unsigned int prevHash = (idx == 0) ? 999999 : gridParticleHash[idx - 1];

    // If I'm the first particle in this cell
    if (myHash != prevHash) {
        cellStart[myHash] = idx;
        if (idx > 0) cellEnd[prevHash] = idx; // End of previous cell
    }

    // If I'm the very last particle
    if (idx == num_particles - 1) {
        cellEnd[myHash] = idx + 1;
    }
}

// ------------------------------------------------------------------
// Kernel 4: Boids Update (Optimized with Uniform Grid)
// ------------------------------------------------------------------
__global__ void boids_grid_kernel(float4* pos_sorted, float4* vel_sorted, 
                                  float4* pos_out, float4* vel_out, // Output to original buffers
                                  unsigned int* cellStart, unsigned int* cellEnd, 
                                  unsigned int* gridParticleIndex, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    float4 my_pos = pos_sorted[idx];
    float4 my_vel = vel_sorted[idx];

    float4 cohesion = make_float4(0,0,0,0);
    float4 separation = make_float4(0,0,0,0);
    float4 alignment = make_float4(0,0,0,0);
    int neighbor_count = 0;

    // 1. Identify my grid cell coordinates
    int gridPos_x = (int)((my_pos.x + 1.0f) / WORLD_SIZE * GRID_SIZE);
    int gridPos_y = (int)((my_pos.y + 1.0f) / WORLD_SIZE * GRID_SIZE);
    gridPos_x = max(0, min(gridPos_x, GRID_SIZE - 1));
    gridPos_y = max(0, min(gridPos_y, GRID_SIZE - 1));

    // 2. Iterate over 3x3 neighbor cells
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            
            // Neighbor cell coordinates with boundary check
            int neigh_x = gridPos_x + x;
            int neigh_y = gridPos_y + y;

            if (neigh_x < 0 || neigh_x >= GRID_SIZE || neigh_y < 0 || neigh_y >= GRID_SIZE) continue;

            int neighHash = neigh_y * GRID_SIZE + neigh_x;

            // Get start/end indices for this cell from the directory
            unsigned int start_index = cellStart[neighHash];
            unsigned int end_index = cellEnd[neighHash];

            // 3. Iterate particles in this neighbor cell
            for (unsigned int i = start_index; i < end_index; i++) {
                if (i == idx) continue; // Skip self

                float4 other_pos = pos_sorted[i];
                float4 other_vel = vel_sorted[i];
                float dist = length(sub(my_pos, other_pos));

                // Standard Boids Logic
                if (dist < c_visualRange && dist > 0.0001f) {
                    cohesion = add(cohesion, other_pos);
                    alignment = add(alignment, other_vel);
                    neighbor_count++;

                    if (dist < c_protectedRange) {
                        float4 push = sub(my_pos, other_pos);
                        separation.x += push.x / dist;
                        separation.y += push.y / dist;
                    }
                }
            }
        }
    }

    // Apply Boids Rules
    if (neighbor_count > 0) {
        cohesion.x /= neighbor_count; cohesion.y /= neighbor_count;
        cohesion = sub(cohesion, my_pos);
        cohesion = mult(cohesion, c_centeringFactor);

        alignment.x /= neighbor_count; alignment.y /= neighbor_count;
        alignment = mult(alignment, c_alignFactor);

        separation = mult(separation, c_avoidFactor);
    }

    // Apply environmental repulsion to prevent agents from penetrating physical walls
    float2 repulsion = computeMapRepulsion(my_pos);
    my_vel = add(my_vel, make_float4(repulsion.x, repulsion.y, 0, 0));

    // Apply flow field force
    float2 flowFieldForce = computeFlowFieldForce(my_pos);
    my_vel = add(my_vel, make_float4(flowFieldForce.x, flowFieldForce.y, 0, 0));

    // Friction
    my_vel = mult(my_vel, 0.98f);

    my_vel = add(my_vel, cohesion);
    my_vel = add(my_vel, separation);
    my_vel = add(my_vel, alignment);

    // Limit Speed
    float current_max_speed = c_maxSpeed * 1.5f;
    float speed = length(my_vel);
    if (speed > current_max_speed) {
        my_vel = mult(my_vel, current_max_speed / speed);
    }

    my_pos = add(my_pos, my_vel);
    my_pos.w = 1.0f; // Critical for OpenGL

    // Boundary constraint
    if (my_pos.x > 1.0f) { my_pos.x = 1.0f; my_vel.x *= -1.0f; }
    if (my_pos.x < -1.0f) { my_pos.x = -1.0f; my_vel.x *= -1.0f; }
    if (my_pos.y > 1.0f) { my_pos.y = 1.0f; my_vel.y *= -1.0f; }
    if (my_pos.y < -1.0f) { my_pos.y = -1.0f; my_vel.y *= -1.0f; }

    // Store in original buffer (mapped to VBO)
    // We use the 'gridParticleIndex' to write back to the original slot?
    // Actually, writing to 'idx' (which corresponds to sorted order) basically shuffles the VBO.
    // For point rendering, order doesn't matter. So simple write is fine.
    pos_out[idx] = my_pos;
    vel_out[idx] = my_vel;
}

__global__ void init_particles_kernel(float4* pos, float4* vel, int num_particles, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;

    // Random Init
    float rx = random(idx * seed) * 2.0f - 1.0f;
    float ry = random(idx * seed + 1234) * 2.0f - 1.0f;
    pos[idx] = make_float4(rx, ry, 0.0f, 1.0f); 

    float vx = (random(idx * seed + 5678) - 0.5f) * 0.01f;
    float vy = (random(idx * seed + 9999) - 0.5f) * 0.01f;
    vel[idx] = make_float4(vx, vy, 0.0f, 0.0f);
}

// ------------------------------------------------------------------
// Host Functions
// ------------------------------------------------------------------
void initCuda(cudaGraphicsResource** vbo_resource, unsigned int vbo, int num_particles) {
    cudaSetDevice(0);
    cudaGraphicsGLRegisterBuffer(vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);

    // 1. Allocate main velocity buffer
    cudaMalloc((void**)&dev_vel, num_particles * sizeof(float4));

    // 2. Allocate Grid Acceleration Structures
    cudaMalloc((void**)&dev_gridParticleHash, num_particles * sizeof(unsigned int));
    cudaMalloc((void**)&dev_gridParticleIndex, num_particles * sizeof(unsigned int));
    cudaMalloc((void**)&dev_cellStart, NUM_CELLS * sizeof(unsigned int));
    cudaMalloc((void**)&dev_cellEnd, NUM_CELLS * sizeof(unsigned int));
    cudaMalloc((void**)&dev_pos_sorted, num_particles * sizeof(float4));
    cudaMalloc((void**)&dev_vel_sorted, num_particles * sizeof(float4));

    // Init Particles
    float4* dptr;
    size_t num_bytes;
    cudaGraphicsMapResources(1, vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, *vbo_resource);

    int threads = 256;
    int blocks = (num_particles + threads - 1) / threads;
    init_particles_kernel<<<blocks, threads>>>(dptr, dev_vel, num_particles, 1234UL);

    cudaGraphicsUnmapResources(1, vbo_resource, 0);
}

void runCuda(cudaGraphicsResource* vbo_resource, int num_particles, float time, int sensorValue) {
    // Normalize Sensor Value (0 ~ 1023 -> 0.0 ~ 1.0)
    float t = sensorValue / 1023.0f;

    // Map Sensor Value to Simulation Parameters
    // Mode 0 (Gas-like): High Separation, Low Cohesion -> Dispersed
    // Mode 1 (Liquid-like): Low Separation, High Cohesion -> Clustered

    // Cohesion: 0.0 -> 0.08f (Stronger force for clustering)
    float targetCohesion = t * 0.08f;

    // Separation: 0.04f -> 0.002f (Reduce repulsion when clustering)
    float targetSeparation = 0.04f * (1.0f - t) + 0.002f;

    // Update GPU Constant Memory
    cudaMemcpyToSymbol(c_centeringFactor, &targetCohesion, sizeof(float));
    cudaMemcpyToSymbol(c_avoidFactor, &targetSeparation, sizeof(float));

    // Visual Range is fixed to match Grid Size limit
    float fixedRange = 0.03f;
    cudaMemcpyToSymbol(c_visualRange, &fixedRange, sizeof(float));

    float4* dptr; // Mapped VBO (Position)
    size_t num_bytes;

    cudaGraphicsMapResources(1, &vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, vbo_resource);

    int threads = 256;
    int blocks = (num_particles + threads - 1) / threads;

    // ---------------------------------------------------------
    // Step 1: Calculate Hash & Index
    // ---------------------------------------------------------
    calcHashD_kernel<<<blocks, threads>>>(dptr, dev_gridParticleHash, dev_gridParticleIndex, num_particles);

    // ---------------------------------------------------------
    // Step 2: Sort Particles by Hash (using Thrust)
    // ---------------------------------------------------------
    thrust::device_ptr<unsigned int> t_hash(dev_gridParticleHash);
    thrust::device_ptr<unsigned int> t_index(dev_gridParticleIndex);
    thrust::sort_by_key(t_hash, t_hash + num_particles, t_index);

    // ---------------------------------------------------------
    // Step 3: Reorder Data (Pos/Vel) based on sorted Index
    // ---------------------------------------------------------
    reorderDataD_kernel<<<blocks, threads>>>(dev_gridParticleIndex, dptr, dev_vel, 
                                             dev_pos_sorted, dev_vel_sorted, num_particles);

    // ---------------------------------------------------------
    // Step 4: Find Start/End of each Cell
    // ---------------------------------------------------------
    // Reset start/end arrays first
    cudaMemset(dev_cellStart, 0xff, NUM_CELLS * sizeof(unsigned int)); // 0xff... means empty/invalid
    cudaMemset(dev_cellEnd, 0, NUM_CELLS * sizeof(unsigned int));

    findCellStartEndD_kernel<<<blocks, threads>>>(dev_gridParticleHash, dev_cellStart, dev_cellEnd, num_particles);

    // ---------------------------------------------------------
    // Step 5: Solve Boids (Neighbor Search)
    // ---------------------------------------------------------
    boids_grid_kernel<<<blocks, threads>>>(dev_pos_sorted, dev_vel_sorted, 
                                           dptr, dev_vel, 
                                           dev_cellStart, dev_cellEnd, 
                                           dev_gridParticleIndex, num_particles);

    cudaGraphicsUnmapResources(1, &vbo_resource, 0);
}

void cleanupCuda(cudaGraphicsResource* vbo_resource) {
    cudaGraphicsUnregisterResource(vbo_resource);

    if (dev_vel) cudaFree(dev_vel);
    if (dev_pos_sorted) cudaFree(dev_pos_sorted);
    if (dev_vel_sorted) cudaFree(dev_vel_sorted);
    
    // Free Grid Resources
    if (dev_gridParticleHash) cudaFree(dev_gridParticleHash);
    if (dev_gridParticleIndex) cudaFree(dev_gridParticleIndex);
    if (dev_cellStart) cudaFree(dev_cellStart);
    if (dev_cellEnd) cudaFree(dev_cellEnd);
}

// Host function to upload the thresholded CPU map directly into the GPU's constant memory symbol.
void initMapData(unsigned char* cpu_map, unsigned short* cpu_dist_map) {
    cudaMemcpyToSymbol(d_map, cpu_map, sizeof(unsigned char) * MAP_HEIGHT * MAP_WIDTH);
    cudaMemcpyToSymbol(d_dist_map, cpu_dist_map, sizeof(unsigned short) * MAP_HEIGHT * MAP_WIDTH);
}