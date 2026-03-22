#pragma once
// uniform init
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math.h"
#include "minitorch/utils.cuh"
using namespace minitorch;
__global__ void fill_mat(float *data, float value, int n_cols, int n_rows) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < n_cols && y < n_rows) {
        data[get_idx_2d(y, x, n_cols)] = value;
    } else
        return;
}
// overload for matrix
__global__ void uniform_init(float *data, int cols, int rows, float scale) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < cols && y < rows) {
        data[get_idx_2d(y, x, cols)] =
            data[get_idx_2d(y, x, cols)] * scale * sqrt(6.0f / (cols + rows));
    }
}
// overload for tensor
__device__ __inline__ void uniform_init(float *data, float scale, auto warp, int n_cols,
                                        int n_rows) {
    for (int tid = warp.thread_rank(); tid < n_cols * n_rows; tid += warp.size()) {
        data[tid] = data[tid] * scale * sqrt(6.0f / (n_cols + n_rows));
    }
}

__global__ void scale_fill(float *data, float low, float high, int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    if (x < n) {
        data[x] = (high - low) * data[x] + low;
    }
}
