#pragma once
#include "matrix.cuh"
#include "tensor.cuh"
#include "cooperative_groups/reduce.h"
#include <cooperative_groups>
#define TILE_SIZE 16
#include "minitorch/utils.cuh"

namespace cg = cooperative_groups;
/*
 * This file contains device functions that can be used for operations on matrices and tensors
 * operations will be moved from ops.cuh/cu to individual tensor/matrix files
 */
namespace minitorch {

// addition
inline void __device__ dev_add(const float *__restrict__ a, const float *__restrict__ b, float *c,
                               int n_cols, int n_rows) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= n_cols || y >= n_rows)
        return;

    c[get_idx_2d(y, x, n_cols)] = a[get_idx_2d(y, x, n_cols)] + b[get_idx_2d(y, x, n_cols)];
}

inline void __device__ dev_bias_add(const float *__restrict__ a, const float *__restrict__ b,
                                    float *c, int n_cols, int n_rows) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= n_cols || y >= n_rows)
        return;

    c[get_idx_2d(y, x, n_cols)] = a[get_idx_2d(y, x, n_cols)] + b[x];
}

// sub
inline void __device__ dev_sub(const float *__restrict__ a, const float *__restrict__ b, float *c,
                               int n_cols, int n_rows) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= n_cols || y >= n_rows)
        return;
    c[get_idx_2d(y, x, n_cols)] = a[get_idx_2d(y, x, n_cols)] - b[get_idx_2d(y, x, n_cols)];
}
inline void __device__ dev_elem_mul(const float *__restrict__ a, const float *__restrict__ b,
                                    float *c, int n_cols, int n_rows) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= n_cols || y >= n_rows)
        return;
    c[get_idx_2d(y, x, n_cols)] = a[get_idx_2d(y, x, n_cols)] * b[get_idx_2d(y, x, n_cols)];
}
inline void __device__ dev_scalar_mul(const float *__restrict__ a, const float b, float *c,
                                      int n_cols, int n_rows) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= n_cols || y >= n_rows)
        return;
    c[get_idx_2d(y, x, n_cols)] = a[get_idx_2d(y, x, n_cols)] * b;
}

inline void __device__ dev_transpose(const float *A, float *C, int n_cols, int n_rows) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= n_cols || y >= n_rows)
        return;

    C[x * n_rows + y] = A[y * n_cols + x];
}
inline void __device__ dev_dot_product(const float *__restrict__ A, const float *__restrict__ B,
                                       float *C, int a_rows, int b_rows, int b_cols) {

    __shared__ float tile_a[TILE_SIZE][TILE_SIZE], tile_b[TILE_SIZE][TILE_SIZE];

    int x = threadIdx.x + blockDim.x * blockIdx.x;    // device col index
    int y = threadIdx.y + blockDim.y * blockIdx.y;    // row index
    int x_local = threadIdx.x, y_local = threadIdx.y; // local index

    float sum = 0.0f;

    // A has row number A_rows and col number b_rows hence a stride is b_rows
    // B has row number b_rows and col number b_cols hence b stride is b_cols
    //
    // fill the shared mem tiles
    for (int i = 0; i < (b_rows + TILE_SIZE - 1) / TILE_SIZE; i++) {
        if (i * TILE_SIZE + x_local >= b_rows) {
            tile_a[y_local][x_local] = 0.0f;
            tile_b[y_local][x_local] = 0.0f;
        } else {
            tile_a[y_local][x_local] = A[y * b_rows + (i * TILE_SIZE + x_local)];
            tile_b[y_local][x_local] = B[(i * TILE_SIZE + y_local) * b_cols + x];
        }
        __syncthreads();
        // perform the dot product
        for (int j = 0; j < TILE_SIZE; j++) {
            sum += tile_a[y_local][j] * tile_b[j][x_local];
        }
        __syncthreads();
    }
    if (x >= b_cols || y >= a_rows) {
        return;
    }
    C[y * b_cols + x] = sum;
}

inline __device__ float dev_sum(const float* __restrict__ start, int reduce_size, int stride, auto warp ){
    float sum = 0.0f;

    for (int i = warp.thread_rank(); i < reduce_size; i += warp.size()){
        sum += start[i * stride];
    }

    // warp reduce
    sum = cg::reduce(warp, sum, cg::plus<float>());


    return sum;

}

} // namespace minitorch
