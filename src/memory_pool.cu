#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include "minitorch/memory_pool.cuh"

using namespace minitorch;

namespace minitorch {

MemoryPool::~MemoryPool() {
    for (auto &pair : cache) {
        for (float *ptr : pair.second) {
            cudaFree(ptr);
        }
    }
    cache.clear();
}

float *MemoryPool::allocate(size_t bytes) {

    auto it = cache.find(bytes);

    if (it != cache.end() && !it->second.empty()) {
        float *ptr = it->second.back();
        it->second.pop_back();
        return ptr;
    }

    float *ptr = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&ptr), bytes);

    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
    }

    return ptr;
}

void MemoryPool::deallocate(float *ptr, size_t bytes) {
    if (!ptr)
        return;

    cache[bytes].push_back(ptr);
}
} // namespace minitorch
