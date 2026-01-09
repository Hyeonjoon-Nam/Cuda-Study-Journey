# Project 01: Vector Addition Optimization

## Overview
This project implements element-wise vector addition using CUDA.
The goal was to move from a naive CPU-like implementation to a fully optimized, scalable GPU kernel. I profiled each version to analyze the performance impact of parallelization strategies.

**Target Data Size:** 1,000,000 elements (float)

## Implementation Details

I iterated through three versions of the kernel:

### 1. v1_single_thread (Naive)
- **Configuration:** `<<<1, 1>>>`
- Runs effectively as a sequential code on a single GPU thread.
- **Purpose:** Establishing a baseline to measure pure overhead and lack of parallelism.

### 2. v2_single_block (Basic Parallelism)
- **Configuration:** `<<<1, 256>>>`
- Utilizes multiple threads but is limited to a single Streaming Multiprocessor (SM).
- **Limitation:** Cannot scale beyond the max threads per block (1024) or utilize the full GPU.

### 3. v3_grid_stride (Optimized)
- **Configuration:** `<<<numBlocks, 256>>>`
- Implements the **Grid-Stride Loop** pattern.
- **Advantage:** Decouples grid size from data size, allowing the kernel to scale to any input size while fully saturating the GPU hardware.

## Performance Analysis

Results captured using **NVIDIA Nsight Compute** (Release Build).

| Version | Execution Time | Speedup (vs v1) | Note |
| :--- | :--- | :--- | :--- |
| v1 (Single Thread) | ~164.39 ms | 1x | Baseline |
| v2 (Single Block) | ~2.24 ms | ~73x | Limited by 1 Block |
| **v3 (Grid-Stride)** | **~37.92 us** | **~4,335x** | **Full Utilization** |

### Nsight Compute Screenshot
![Profiling Result](./assets/nsight_profiling_result.png)
*(Comparision of execution time showing drastic improvement in v3)*

## Key Takeaways
- **Grid-Stride Loops are essential:** They make kernels robust against varying data sizes and hardware configurations.
- **Launch Configuration matters:** Simply adding threads isn't enough; calculating the correct number of blocks is crucial for utilizing all SMs.
- **Unified Memory:** Simplifies memory management for prototyping, though manual `cudaMemcpy` might be preferred for explicit control in future projects.
