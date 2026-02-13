#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA Host Functions
void initCuda(struct cudaGraphicsResource** vbo_resource, unsigned int vbo, int num_particles);
void runCuda(struct cudaGraphicsResource* vbo_resource, int num_particles, float time, int sensorValue);
void cleanupCuda(struct cudaGraphicsResource* vbo_resource);