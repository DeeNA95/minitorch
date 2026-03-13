---
title: "From CNNs to CUDA: An Intuitive Guide to Tiled Matrix Multiplication"
date: 2026-03-06
description: "How understanding Convolutional Neural Networks helps build intuition for GPU architecture and optimized Matrix Multiplication."
series: Learning CUDA
---

# From CNNs to CUDA: An Intuitive Guide to Tiled Matrix Multiplication

When writing a deep learning framework from scratch, optimizing Matrix Multiplication (MatMul) is a rite of passage. If you've ever written a naive MatMul kernel in CUDA, you know the feeling: it's correct, but painfully slow. The standard advice is to use "tiled matrix multiplication" with shared memory.

For a long time, the mechanics of tiling—moving blocks of data into shared memory, syncing threads, and calculating partial sums—felt mechanical. I understood the *how*, but the *why* didn't truly click until I thought about it through a lens I already understood: **Convolutional Neural Networks (CNNs).**

Let's explore how the core intuition behind CNNs maps perfectly onto GPU memory architecture, and how recognizing this connection demystifies optimizations like tiled MatMul.

## Global vs. Local Understanding

At the heart of a CNN is the convolution operation: sliding a small filter (the kernel) over an image to detect local patterns like edges or textures.

A Multi-Layer Perceptron (MLP) or fully connected layer tries to understand the *entire* image at once. Every input pixel connects to every neuron. This is a **global** operation. It's expensive, loses spatial context, and struggles to scale.

A CNN, conversely, relies on **local understanding**. It looks at a small neighborhood of pixels (e.g., a 3x3 patch) at a time. By focusing on local patches, it shares weights and builds up a complex understanding hierarchically.

This exact dichotomy—global vs. local—exists in GPU memory architecture.

## The GPU Memory Hierarchy

If we look closely, the GPU architecture heavily favors local operations, much like a CNN.

1. **Global Memory (The VRAM):** This is huge (16GB, 24GB, or more) but incredibly slow. Every read from global memory takes hundreds of clock cycles.
2. **Shared Memory (The Local Patch):** This is tiny (usually 48KB or 96KB per Streaming Multiprocessor) but blazingly fast. It acts as a user-managed cache.

A naive MatMul kernel is like an MLP trying to understand an entire image at once.

When computing $C = A \times B$, every thread computes one element of the output matrix $C$. To do this, it needs to read an entire row of $A$ and an entire column of $B$.

```cpp
// Naive MatMul (Global Understanding)
float sum = 0.0f;
for (int i = 0; i < N; i++) {
    // Reading from slow global memory every single time!
    sum += A[row * N + i] * B[i * N + col];
}
C[row * N + col] = sum;
```

Because adjacent threads in a block are computing adjacent elements of $C$, they end up re-reading the exact same elements of $A$ and $B$ from global memory multiple times. It's wildly inefficient. It's trying to solve a local problem by repeatedly querying the global state.

## Enter Tiling (The Convolutional Intuition)

Tiled MatMul fixes this by forcing the GPU to behave more like a CNN. Instead of operating on the whole matrix, we break the problem down into **local patches** (tiles).

Just as a CNN slides a 3x3 filter across an image, tiled MatMul slides a "tile" (e.g., 16x16 or 32x32) across the input matrices.

Here is the intuition step-by-step:

### 1. Identify the Local Region

A thread block (a group of threads) is assigned a small tile of the output matrix $C$ to compute. Let's say a 16x16 block.

### 2. Load the Patch into Fast Memory

Before doing any math, the threads collaborate to load the corresponding 16x16 patch of $A$ and $B$ from slow Global Memory into fast Shared Memory.

This is the crucial step. Instead of hundreds of threads individually fetching the same data from global memory, they work together: each thread fetches exactly one element into the shared cache.

```cpp
// 1. Threads collaborate to load a local patch into Shared Memory
__shared__ float tile_A[TILE_SIZE][TILE_SIZE];
__shared__ float tile_B[TILE_SIZE][TILE_SIZE];

tile_A[threadIdx.y][threadIdx.x] = A[...];
tile_B[threadIdx.y][threadIdx.x] = B[...];

// Wait for all threads to finish loading the patch
__syncthreads();
```

### 3. Process Locally

Now that the data patch is in fast shared memory, the threads perform the dot products required for their portion of the computation.

Because Shared Memory is close to the processing cores, these reads are incredibly fast. We've converted a slow global operation into a fast local one.

```cpp
// 2. Perform the math using the locally cached patch
for (int i = 0; i < TILE_SIZE; i++) {
    sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
}
__syncthreads(); // Wait before loading the next patch
```

### 4. Slide the Window

Just as the convolution filter slides to the next part of the image, the thread block slides to the next tile in matrices $A$ and $B$, accumulating the partial sums until the entire row/column dot product is complete.

## Why This Intuition Matters

When learning CUDA, the mechanisms of `__shared__` memory and `__syncthreads()` can feel esoteric. But when framed conceptually, they answer a very familiar structural question: **How do we localize computation?**

- **CNNs localize computation in space:** They recognize that adjacent pixels are related and process them together.
- **Tiled MatMul localizes computation in memory:** It recognizes that adjacent threads require the same data, so it caches that data locally to avoid redundant fetches to global state.

Once that clicked, writing the tiled kernel no longer felt like memorizing a hardware trick. It felt like applying the core architectural principle of deep learning—exploiting locality—to the hardware itself.

By applying the logic of reading from a local, shared context rather than a vast, slow global state, we unlock massive performance gains on the GPU. Understanding the problem through the lens of convolution doesn't just make the code faster; it makes the hardware architecture itself profoundly intuitive.
