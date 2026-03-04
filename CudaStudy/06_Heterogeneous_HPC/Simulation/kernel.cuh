#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// The logical resolution of the warehouse grid map (128x128).
#define MAP_WIDTH 128
#define MAP_HEIGHT 128

// Device constant memory for the grid map (0 = path, 1 = obstacle).
__constant__ unsigned char d_map[MAP_WIDTH * MAP_HEIGHT];
// Distance/Integration field for O(1) Vector Flow Field pathfinding.
__constant__ unsigned short d_dist_map[MAP_WIDTH * MAP_HEIGHT];

// CUDA Host Functions
void initCuda(struct cudaGraphicsResource** vbo_resource, unsigned int vbo, int num_particles);
void runCuda(struct cudaGraphicsResource* vbo_resource, int num_particles, float time, int sensorValue);
void cleanupCuda(struct cudaGraphicsResource* vbo_resource);
void initMapData(unsigned char* cpu_map, unsigned short* cpu_dist_map);