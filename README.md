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
| 06 | [Heterogeneous HPC System](./CudaStudy/06_Heterogeneous_HPC) | CUDA-OpenGL Interop, Multi-threading, Serial I/O, Atomic Sync | **Phase 3: Bare-Metal Integrated** |

---

## Current Focus: Project 06 - Heterogeneous HPC Simulation System

Moving beyond standalone simulations, this project aims to build a comprehensive control pipeline that bridges **Low-level Hardware**, **System Programming**, and **High-Performance Computing**.

### System Architecture
The system simulates an **Edge Computing** environment where an external input node controls a massive particle simulation ($N=16,384$) in real-time via a dedicated I/O thread.

```mermaid
graph LR
    A[Input Node: Arduino] -- UART/Serial --> B[HPC App: IO Thread]
    B -- std::atomic --> C[HPC Core: CUDA Kernel]
    C -- Zero-Copy Interop --> D[Render: OpenGL]
```

**(Text Representation)**
`[Input Node: Arduino]` --(UART)--> `[IO Thread: Serial Reader]` --(Atomic Memory)--> `[HPC Core: CUDA Kernel]` --(Interop)--> `[Render: OpenGL]`

### Key Technical Objectives & Results
- **HPC Core (CUDA & OpenGL):** Zero-copy rendering with Spatial Partitioning (Uniform Grid) for real-time performance.
- **System Integration (C++):** Decoupled Hardware I/O from Rendering Loop using `std::thread` & `std::atomic` (Lock-free sync).
- **Embedded Interface (Bare-metal):** **Phase 3 Implemented.** Direct register manipulation (`ADMUX`, `UBRR0`) replacing standard Arduino libraries.

**[View Full Project & Code](./CudaStudy/06_Heterogeneous_HPC)**

---

## Future Roadmap: Phase 2
Moving towards **3D Graphics, Texture Memory, and Vision AI**.

I have categorized my next goals into two parallel tracks to balance software depth and system breadth.

### Track A: Simulation Engine (Software Depth)
Focusing on advanced CUDA memory patterns and 3D graphics.
- **Conway's Game of Life:**
    - Implement Cellular Automata using **CUDA Texture Memory** to optimize non-coalesced memory access patterns.
- **3D Simulation:**
    - Expand the kernel to 3D space (`float4`) and implement **Camera Matrices (View/Projection)** in OpenGL.

### Track B: Physical Interaction (System Breadth)
Focusing on Edge AI and Wireless Networking using **ESP32-S3 CAM**.
- **Networked Architecture:**
    - Transition from USB Serial to **UDP/Wi-Fi Communication**, implementing a C++ Socket Server.
- **Vision AI Control:**
    - Replace the analog potentiometer with **Computer Vision**.
    - Implement **TinyML** on the ESP32 to recognize Hand Gestures (e.g., Open Palm = Scatter, Fist = Gather).