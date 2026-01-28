# Project 06: Heterogeneous HPC Simulation System

## Overview
This project builds a **Full-Pipeline Control System** that integrates Bare-metal Hardware, a Network Gateway, and a CUDA HPC Simulation.
Unlike standalone simulations, this system mimics an **Edge Computing** architecture where an external input node controls a massive particle simulation ($N \ge 100,000$) in real-time via a custom UDP protocol.

**Target Hardware:**
- **Host:** NVIDIA GeForce RTX 3070 Laptop GPU (46 SMs)
- **Input Node:** Arduino Uno (ATmega328P) - *Bare-metal Mode*

## System Architecture

The system consists of three distinct layers connected via an optimized data pipeline.

```mermaid
graph LR
    A[Input Node: Arduino] -- UART/Serial --> B[Gateway: C++ WinSock]
    B -- UDP Socket --> C[HPC Core: CUDA Kernel]
    C -- Zero-Copy Interop --> D[Render: OpenGL]
```

**(Text Representation)**
`[Input Node: Arduino (Bare-metal)]` --(UART)--> `[Gateway: C++ App]` --(UDP)--> `[HPC Core: CUDA]` --(Interop)--> `[Render: OpenGL]`

## Implementation Goals

### 1. HPC Core (Simulation Layer)
- **Objective:** Compute Bound Optimization.
- **Strategy:** Transition from $O(N^2)$ to $O(N)$ using **Spatial Partitioning (Uniform Grid)**.
- **Optimization:**
    - **Thrust Sort:** Reordering particles to maximize memory coalescence.
    - **OpenGL Interop:** Zero-copy rendering to eliminate CPU-GPU bandwidth overhead.

### 2. Middleware (Gateway Layer)
- **Objective:** Asynchronous Data Pipeline.
- **Strategy:** Decouple hardware polling from simulation rendering.
- **Mechanism:**
    - **Multi-threading:** Separate threads for Serial I/O (Input) and UDP transmission.
    - **Thread-safe Queue:** Utilizing `std::mutex` and `std::condition_variable` to prevent race conditions.

### 3. Hardware (Input Layer)
- **Objective:** Low-level Control.
- **Strategy:** **Bare-metal Programming** (No Arduino Library).
- **Mechanism:** Direct register manipulation of `UBRR` (UART) and `ADCSRA` (ADC) to demonstrate embedded proficiency.

## 📂 Directory Structure

```text
06_Heterogeneous_HPC/
├── Firmware/           # [Input Node] Bare-metal C code for ATmega328P
├── Gateway/            # [Middleware] C++ WinSock UDP Gateway
└── Simulation/         # [HPC Core] CUDA & OpenGL Visualization
    ├── kernel.cu       # Spatial Partitioning & Physics Kernels
    └── main.cpp        # Rendering Loop & Network Receiver
```

## 📅 Development Roadmap

### Phase 1: Core Engine (Current Focus)
- [ ] **Step 1:** OpenGL Interop Setup (Zero-copy Visualization)
- [ ] **Step 2:** Naive Boids Implementation ($O(N^2)$)
- [ ] **Step 3:** Spatial Partitioning Optimization (Uniform Grid)

### Phase 2: System Integration
- [ ] **Step 4:** C++ UDP Gateway (Serial <-> Network Bridge)
- [ ] **Step 5:** Thread-safe Network Receiver in Simulation

### Phase 3: Hardware Control
- [ ] **Step 6:** Bare-metal Firmware Implementation (Register Level)

## Performance Analysis
*(To be updated upon project completion. Expected metrics: FPS comparison between O(N^2) vs Grid, and Latency measurement from Arduino to Render.)*