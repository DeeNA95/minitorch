# Building a C++ Deep Learning Framework from Scratch: My Journey with CUDA

Building a deep learning framework like PyTorch from the ground up is often seen as a rite of passage for AI engineers. Over the past few weeks, I’ve been heads-down in the weeds of C++, CUDA kernels, and memory management to build **MiniTorch**, my own GPU-accelerated library.

## The Performance Gap: Beyond Naive Kernels

When you start, you write naive kernels. They work, but they’re slow. The real magic happens when you start thinking about how the GPU actually "breathes" data.

### 1. The Caching Allocator (Memory Pool)
In a training loop with hundreds of thousands of iterations, calling `cudaMalloc` and `cudaFree` every step is a death sentence for performance. I implemented a custom **GPU Memory Pool** that caches and reuses allocations. Instead of asking the OS for memory every time, the framework manages its own slab, making tensor operations feel instantaneous.

### 2. Tiled Matrix Multiplication
Optimizing MatMul is where you truly understand GPU architecture. By using **Shared Memory Tiling**, I moved computation from slow Global Memory to blazingly fast SM-local memory. 

The intuition? It's exactly like a **CNN**. Just as a CNN slides a filter over an image to process local patches, a tiled kernel slides small windows of data into the GPU's registers, reducing memory redundant fetches by orders of magnitude. 

## Design Choice: Why Modules over Tensor Autograd?

One question I get is: *"Why didn't you build it like PyTorch where `loss.backward()` just works?"*

PyTorch uses **Tensor-based Autograd**, where every tensor remembers its history via a dynamic computational graph. For MiniTorch, I consciously chose an **Object-Oriented Module System**.

In this architecture, layers like `Linear` or `Sigmoid` are explicitly responsible for their own `forward` and `backward` passes. 

**Why?**
1.  **Transparency**: It forced me to manually cache the exact intermediate states (like Sigmoid's output or ReLU's mask) needed for backprop. 
2.  **Simplicity & Speed**: At this stage, building a complex logic for dynamic graph construction adds overhead that distracts from the core goal: writing the most efficient CUDA kernels possible.
3.  **The "Lego" Feel**: Using a `Sequential` container that explicitly iterates through modules makes the flow of gradients incredibly clear.

## The Results
Running on a dataset of **350,000+ sales samples**, the framework now trains a multi-layer DNN with **Adam optimization** completely on the GPU. Seeing the MSE loss drop in real-time on a framework I wrote from scratch is one of the most rewarding moments in my engineering journey.

## What's Next?
The foundation is solid. Next up:
*   **CNNs**: Moving beyond MLPs.
*   **Infrastructure**: Decoupling the backend to support CPU/GPU dispatch.
*   **The Big One**: Building a Transformer block entirely in this C++ ecosystem.

Stay tuned—I'll be sharing more deep dives into the kernels soon!
