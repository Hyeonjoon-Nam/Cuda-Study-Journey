# CUDA Learning Journey

This repository documents my progress in mastering CUDA programming and High-Performance Computing (HPC).
My goal is to understand the hardware architecture deeply and write highly optimized kernels.

## Environment
- **GPU:** NVIDIA GeForce RTX 3070 Laptop GPU
- **IDE:** Visual Studio 2022
- **Toolkit:** CUDA 13.1
- **Profiler:** NVIDIA Nsight Compute / Nsight Systems

## Project List

| # | Project | Key Concepts | Status |
|:-:|:---|:---|:---|
| 01 | [Vector Addition](./CudaStudy/01_Add) | Grid-Stride Loop, Unified Memory, Profiling | Done |
| 02 | [Matrix Multiplication](./CudaStudy/02_MatrixMultiplication) | Shared Memory, Tiling, Vectorized Access (float4) | Done |
| 03 | Parallel Reduction | Synchronization, Warp Divergence, Bank Conflicts | Planned |
| 04 | N-Body Simulation | Physics Integration, All-Pairs Interaction | Planned |
| 05 | Spatial Partitioning | Uniform Grid, Atomic Operations | Planned |
| 06 | Massive Boids | Flocking Behavior, Optimization, Visualization | Planned |