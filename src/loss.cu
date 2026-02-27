#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include "cooperative_groups.h"
#include "cooperative_groups/reduce.h"
#include "minitorch/loss.cuh"
#include "minitorch/utils.cuh"

namespace cg = cooperative_groups;

namespace minitorch {

__global__ void average(float *__restrict__ sums, int n) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    float s = 0.0f;

    for (int tid = grid.thread_rank(); tid < n; tid += grid.size()) {
        s += sums[tid];
    }

    s = cg::reduce(warp, s, cg::plus<float>());

    if (warp.thread_rank() == 0) // relies on only one warp being launches ie no more than 32
                                 // threads
        sums[0] = s / n;
}

__global__ void ker_mse_forward(const float *__restrict__ preds_data,
                                const float *__restrict__ actuals_data, float *__restrict__ sums,
                                int n) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    float s = 0.0f;

    for (int tid = grid.thread_rank(); tid < n; tid += grid.size()) {
        float diff = (actuals_data[tid] - preds_data[tid]);
        s += diff * diff;
    }
    warp.sync();

    s = cg::reduce(warp, s, cg::plus<float>());
    if (warp.thread_rank() == 0) {
        atomicAdd(&sums[block.group_index().x], s);
    }
};

float mse_forward(Matrix &preds, Matrix &actual) {
    float *preds_data = preds.getdata();
    float *actuals_data = actual.getdata();
    int n_cols = preds.getcols();
    int n_rows = preds.getrows();

    assert(preds.getcols() == actual.getcols() && "Matrices must have the same number of columns");
    assert(preds.getrows() == actual.getrows() && "Matrices must have the same number of rows");

    int n = preds.getcols() * preds.getrows();

    dim3 threads(256);
    dim3 blocks((n_cols + threads.x - 1) / threads.x);

    float *sums;
    cudaMalloc(&sums, sizeof(float) * blocks.x);
    cudaMemset(sums, 0, sizeof(float) * blocks.x);

    ker_mse_forward<<<blocks, threads>>>(preds_data, actuals_data, sums, n);
    average<<<1, 32>>>(sums, blocks.x); // blocks.x to get width of block
    cudaDeviceSynchronize();
    float loss;
    cudaMemcpy(&loss, sums, (std::size_t)(sizeof(float)), cudaMemcpyDeviceToHost);
    cudaFree(sums);

    return loss;
}

__global__ void ker_mse_backward(const float *__restrict__ preds, const float *__restrict__ actuals,
                                 float *__restrict__ derivs, int n, int n_cols) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= n_cols || y >= n / n_cols)
        return;

    derivs[get_idx_2d(y, x, n_cols)] =
        (2.0f / n) * (preds[get_idx_2d(y, x, n_cols)] - actuals[get_idx_2d(y, x, n_cols)]);
}

Matrix mse_backward(Matrix &preds, Matrix &actual) {
    float *preds_data = preds.getdata();
    float *actuals_data = actual.getdata();
    int n_cols = preds.getcols();
    int n_rows = preds.getrows();

    assert(preds.getcols() == actual.getcols() && "Matrices must have the same number of columns");
    assert(preds.getrows() == actual.getrows() && "Matrices must have the same number of rows");

    int n = preds.getcols() * preds.getrows();

    dim3 threads(16, 16);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y);

    Matrix derivs = Matrix(n_rows, n_cols);

    ker_mse_backward<<<blocks, threads>>>(preds.getdata(), actual.getdata(), derivs.getdata(), n,
                                          n_cols);

    return derivs;
}
} // namespace minitorch
