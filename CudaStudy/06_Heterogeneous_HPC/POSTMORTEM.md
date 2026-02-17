# Postmortem: The Journey from CUDA Hello-World to Heterogeneous System

## 1. Project Overview & The "Aha!" Moment
This project concludes the first chapter of my CUDA learning journey. Starting from basic vector addition, it evolved into a full-stack system integrating Bare-metal Firmware, System Software, and GPU Acceleration.

**The most thrilling moment:**
When I finally connected the Arduino potentiometer, rotated the dial, and saw 16,000+ particles instantly react on the screen. It wasn't just code anymore; it was a physical interaction. I immediately called my roommate to show off this "living" simulation. That tangible connection between hardware and software was the highlight of this project.

## 2. Technical Milestones (Fact Check)

People often say "GPU is 1000x faster," but through these projects, I learned exactly **when** and **why** it is faster.

- **Project 01 (Vector Add):** Achieved **~4000x speedup** compared to single-threaded CPU. This showed the raw power of massive parallelism for simple tasks.
- **Project 02 (MatMul):** Speedup settled to **~1.35x - 4x** depending on optimization. Learned that simply throwing threads at a problem isn't enough; **Memory Coalescing** and **Shared Memory Tiling** are crucial to overcome bandwidth bottlenecks.
- **Project 03 (Reduction):** Tackled **Warp Divergence** and **Bank Conflicts**. Learned that hardware architecture dictates code structure.
- **Project 06 (System):** The focus shifted from raw speed to **System Stability**. Using `std::thread` and `std::atomic` prevented the slow serial I/O (9600bps) from stuttering the 60FPS rendering loop.

## 3. Theory vs. Reality (Hardware Constraints)
The most valuable lesson came from **Project 04 (N-Body)**.
- **Theory:** "Thread Coarsening" reduces instruction overhead and should be faster.
- **Reality:** On my RTX 3070 Laptop GPU, it performed **worse** than Shared Memory Tiling.
- **Analysis:** Coarsening increased register usage per thread, which limited **Occupancy** (the number of active warps).
- **Conclusion:** Theoretical optimization techniques are not silver bullets. One must profile the specific hardware constraints (Register File size, Cache size) to find the sweet spot. This sparked my interest in **Embedded System Engineering**—optimizing within strict resource limits.

## 4. Bare-metal & Hardware Integration
- **Direct Register Access:** Instead of easy Arduino libraries, I manipulated `ADMUX`, `ADCSRA`, and `UBRR0` directly. It was painful to debug (especially bitwise operators and timing), but I now understand how the MCU actually works.
- **Troubleshooting:**
    - **Floating Input:** Debugged erratic sensor values caused by missing pull-down resistors/connections.
    - **Serial Fragmentation:** Solved the issue of fragmented data packets (e.g., `10`, `23` vs `1023`) by implementing a state-machine parser in C++.

## 5. Conclusion: Ready for Phase 2

Through Project 06, I successfully integrated CUDA simulation with physical hardware using bare-metal programming. However, I also realized the physical limitations of "Wired" (USB) and "Analog" (Potentiometer) controls.

To overcome these constraints and push the boundaries of Heterogeneous Computing, I am now moving forward to **Phase 2**, focusing on **Computer Vision (Edge AI)** and **Wireless Networking**. 

*(See the root README for the detailed roadmap.)*