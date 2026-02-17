# Project 06: Heterogeneous HPC Simulation System

![Demo](./assets/demo.gif)
*(Real-time Boids Simulation controlled by Bare-metal Arduino Input)*

## Overview
This project builds a **Full-Pipeline Control System** that integrates Bare-metal Hardware and a CUDA HPC Simulation.
The system mimics an **Embedded/HPC** architecture where an external input node controls a massive particle simulation ($N \ge 16,384$) in real-time via a dedicated hardware thread.

**Target Hardware:**
- **Host:** NVIDIA GeForce RTX 3070 Laptop GPU (46 SMs)
- **Input Node:** Arduino Uno (ATmega328P) - *Bare-metal Mode*

## System Architecture

The system consists of three layers connected via a **Direct Memory Access (DMA) style** pipeline using multi-threading.

```mermaid
graph LR
    A[Input Node: Arduino] -- UART/Serial --> B[HPC App: IO Thread]
    B -- Atomic Memory --> C[HPC Core: CUDA Kernel]
    C -- Zero-Copy Interop --> D[Render: OpenGL]
```

**(Text Representation)**
`[Input Node: Arduino]` --(UART)--> `[IO Thread: Serial Reader]` --(Atomic Sync)--> `[HPC Thread: CUDA Physics]` --(Interop)--> `[Render: OpenGL]`

## Implementation Goals

### 1. HPC Core (Simulation Layer)
- **Objective:** Compute Bound Optimization.
- **Strategy:** Transition from $O(N^2)$ to $O(N)$ using **Spatial Partitioning (Uniform Grid)**.
- **Optimization:**
    - **Thrust Sort:** Reordering particles to maximize memory coalescence.
    - **OpenGL Interop:** Zero-copy rendering to eliminate CPU-GPU bandwidth overhead.

### 2. System Integration (I/O Layer)
- **Objective:** Asynchronous Data Pipeline.
- **Strategy:** **Multi-threading & Atomic Synchronization**.
- **Mechanism:**
    - **Thread Separation:** Decoupled `Serial I/O` thread from the `Rendering` thread to prevent blocking.
    - **Variable Mapping:** Mapped physical sensor input (0~1023) to simulation physics (Cohesion/Separation forces).

### 3. Hardware (Input Layer)
- **Objective:** Low-level Control.
- **Strategy:** **Bare-metal Programming** (Direct Register Access).
- **Mechanism:** **Implemented (Phase 3)** Direct register manipulation of `UBRR` (UART) and `ADCSRA` (ADC) without standard Arduino libraries.

## Directory Structure

```text
06_Heterogeneous_HPC/
├── Firmware/
│   └── BareMetal_Potentiometer.ino  # [Phase 3] Register-level AVR Firmware
├── Simulation/                      # [HPC Core] Main Application
│   ├── kernel.cu                    # CUDA Physics Kernels
│   ├── main.cpp                     # OpenGL Loop & Thread Management
│   ├── SerialPort.h                 # Win32 Serial Header
│   └── SerialPort.cpp               # Win32 Serial Implementation
└── README.md                        # Documentation
```

## Development Roadmap

### Phase 1: Core Engine (Complete)
- [x] **Step 1: OpenGL Interop Setup (Zero-Copy Visualization)**
- [x] **Step 2: Naive Boids Implementation ($O(N^2)$)**
- [x] **Step 3: Spatial Partitioning Optimization (Uniform Grid)**
    - **Performance Achieved:** 262,144 particles @ 60 FPS (RTX 3070).

### Phase 2: System Integration (Complete)
- [x] **Step 4: Serial Communication Module**
    - Implemented `SerialPort` class using Win32 API.
- [x] **Step 5: Multi-threaded Integration**
    - Implemented `std::thread` worker for non-blocking I/O.
    - Real-time mapping of sensor data to CUDA constant memory.
    - **Result:** Dynamic Cohesion/Separation control via external hardware input.

### Phase 3: Hardware Control (Complete)
- [x] **Step 6: Bare-metal Firmware Implementation (Register Level)**
    - Replaced `analogRead` with `ADMUX`/`ADCSRA` register control.
    - Replaced `Serial.print` with `UBRR0`/`UDR0` UART control.


### Performance Analysis (Validated via Nsight Compute)

To verify the efficiency of the **Heterogeneous System Architecture**, I profiled the application while the **Arduino was actively sending data** via UART.

![Nsight Profiling Result](./assets/nsight_profiling_active_io.png)
*(Profiling Data: Kernel execution during active Serial I/O)*

**Key Findings:**
* **Zero I/O Overhead:** The `boids_grid_kernel` execution time remained consistent at **~60µs** (compare Loop 1 vs Loop 2), proving that the asynchronous I/O thread (`std::thread`) successfully decoupled UART communication from the CUDA/OpenGL rendering loop.
* **Stable Latency:** Despite the continuous hardware interrupts from the microcontroller, the GPU simulation pipeline maintained a steady frame rate without stalling.
* **Compute Bound:** The Uniform Grid optimization successfully shifted the bottleneck from global memory access to compute, achieving high throughput even with 16,384 particles.