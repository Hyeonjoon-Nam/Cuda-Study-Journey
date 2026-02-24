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

## Future Roadmap: From Simulation to Solution

Having pushed the limits of hardware optimization with 4M+ particle simulations, my next phase pivots from "Tech Demos" to **"Real-World Problem Solving."** Drawing inspiration from real-world warehouse logistics (e.g., Amazon, Coupang), the upcoming roadmap focuses on massive-scale pathfinding, wireless telemetry, and framework architecture.

### Goal 1: Wireless Edge Integration (Project 06 Polish)
Breaking the physical constraints of USB serial communication by introducing **UDP/Wi-Fi Telemetry**.
- **Hardware:** ESP32 (SoftAP Mode) serving as an independent wireless control node.
- **Software:** Implementing a Native C++ WinSock2 UDP Receiver on a dedicated thread.
- **Objective:** Maintain a strict 60FPS lock-free simulation pipeline despite inherent network jitter and packet loss.

### Goal 2: Logistics Swarm Simulator
A massive Multi-Agent Pathfinding (MAPF) simulation mimicking thousands of AGVs (Automated Guided Vehicles) in a warehouse environment.
- **HPC Routing:** Transitioning from individual A* to **Vector Flow Fields** (Reverse BFS) to achieve $O(1)$ path lookup for massive agent counts.
- **Local Avoidance:** Implementing GPU-accelerated collision avoidance (Separation / Tangential Forces) to resolve traffic deadlocks in narrow corridors.
- **Objective:** Visually demonstrate real-time throughput optimization for thousands of autonomous units on a single GPU.

### Goal 3: Unified HPC Sandbox Architecture
Consolidating standalone projects into a single, cohesive engine framework.
- **Framework:** Integrating **Dear ImGui** over the GLFW/OpenGL pipeline.
- **System Design:** Abstracting simulations into a `Scene` management system for runtime hot-swapping.
- **Objective:** Build a recruiter-ready interactive sandbox featuring real-time performance metrics (Throughput, Latency) and parameter tuning.

### Goal 4: Cross-Platform HPC Deployment (AMD ROCm)
Expanding the system's hardware abstraction by porting the CUDA-based simulation to the **AMD ROCm (HIP)** ecosystem.
- **Environment:** AMD Developer Cloud (Instinct Accelerators).
- **Porting:** Utilizing `HIPIFY` to translate CUDA kernels into HIP, ensuring architectural compatibility (Warp vs. Wavefront).
- **Objective:** Cross-validate the simulation's throughput across different GPU architectures, proving the algorithm's platform-independent efficiency.
