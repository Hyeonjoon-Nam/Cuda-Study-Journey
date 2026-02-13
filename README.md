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
| 03 | [Parallel Reduction](./CudaStudy/03_ParallelReduction) | Warp Divergence, Loop Unrolling, Volatile, Bank Conflicts | Done |
| 04 | [N-Body Simulation](./CudaStudy/04_NBodySimulation) | Compute vs Memory Bound, Tiling, Thread Coarsening, Occupancy | Done |
| 05 | Spatial Partitioning | Uniform Grid, Atomic Operations | Integrated into Project 06 |
| 06 | [Heterogeneous HPC System](./CudaStudy/06_Heterogeneous_HPC) | CUDA-OpenGL Interop, Multi-threading, Serial I/O, Atomic Sync | **Phase 2 Complete** |

---

## Current Focus: Project 06 - Heterogeneous HPC Simulation System

Moving beyond standalone simulations, this project aims to build a comprehensive control pipeline that bridges **Low-level Hardware**, **System Programming**, and **High-Performance Computing**.

### System Architecture
The system simulates an **Edge Computing** environment where an external input node controls a massive particle simulation in real-time via a dedicated I/O thread.

```mermaid
graph LR
    A[Input Node: Arduino] -- UART/Serial --> B[HPC App: IO Thread]
    B -- std::atomic --> C[HPC Core: CUDA Kernel]
    C -- Zero-Copy Interop --> D[Render: OpenGL]
```

**(Text Representation)**
`[Input Node: Arduino]` --(UART)--> `[IO Thread: Serial Reader]` --(Atomic Memory)--> `[HPC Core: CUDA Kernel]` --(Interop)--> `[Render: OpenGL]`

### Key Technical Objectives

1.  **HPC Core (CUDA & OpenGL):**
    * Optimizing Massive Boids simulation (16k+ particles) using **Spatial Partitioning (Uniform Grid)** and **Thrust** sort (Radix Sort).
    * Implementing **CUDA-OpenGL Interoperability** to achieve zero-copy rendering, eliminating CPU-GPU bandwidth bottlenecks.

2.  **System Integration (C++):**
    * **Multi-threaded Architecture:** Decoupling the Hardware I/O (Serial) from the Rendering Loop using `std::thread` to ensure stable FPS.
    * **Lock-free Synchronization:** Utilizing `std::atomic` to safely transfer sensor data between the I/O thread and CUDA kernel without mutex overhead.
    * **Dynamic Parameter Mapping:** Mapping physical input (0~1023) to simulation physics (Cohesion/Separation forces) in real-time.

3.  **Embedded Interface (Bare-metal C):**
    * (Planned Phase 3) Programming ATmega328P using **Raw Register Access** (adhering to bare-metal constraints) to control simulation parameters via physical hardware inputs.