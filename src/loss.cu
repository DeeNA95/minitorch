#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include "cooperative_groups.h"
#include "cooperative_groups/reduce.h"
#include "math_constants.h"
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

float mse_forward(Tensor &preds, Tensor &actual) {
    float *preds_data = preds.getdata();
    float *actuals_data = actual.getdata();
    int n_cols = preds.get_shape()[1];
    int n_rows = preds.get_shape()[0];

    assert(preds.get_shape()[1] == actual.get_shape()[1] &&
           "Matrices must have the same number of columns");
    assert(preds.get_shape()[0] == actual.get_shape()[0] &&
           "Matrices must have the same number of rows");

    int n = preds.get_shape()[1] * preds.get_shape()[0];

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

Tensor mse_backward(Tensor &preds, Tensor &actual) {
    float *preds_data = preds.getdata();
    float *actuals_data = actual.getdata();
    int n_cols = preds.get_shape()[1];
    int n_rows = preds.get_shape()[0];

    assert(preds.get_shape()[1] == actual.get_shape()[1] &&
           "Matrices must have the same number of columns");
    assert(preds.get_shape()[0] == actual.get_shape()[0] &&
           "Matrices must have the same number of rows");

    int n = preds.get_shape()[1] * preds.get_shape()[0];

    dim3 threads(16, 16);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y);

    Tensor derivs = Tensor(n_rows, n_cols);

    ker_mse_backward<<<blocks, threads>>>(preds.getdata(), actual.getdata(), derivs.getdata(), n,
                                          n_cols);

    return derivs;
}

/*
 * CROSS ENTROPY
 *
 * */

CrossEntropyLoss::CrossEntropyLoss(int n_classes) : n_classes(n_classes) {}

__global__ void ker_cross_entropy_forward(const float *__restrict__ preds,
                                          const float *__restrict__ actuals,
                                          float *__restrict__ loss, int batch_size, int n_classes) {
    __shared__ float s_max[32];
    __shared__ float s_sum[32];
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    int b = blockIdx.x;
    if (b >= batch_size)
        return;

    float local_max = -CUDART_INF_F;

    for (int c = threadIdx.x; c < n_classes; c += blockDim.x) {
        float p = preds[b * n_classes + c];
        if (p > local_max)
            local_max = p;
    }
    float warp_max;
    warp_max = cg::reduce(warp, local_max, cg::greater<float>());
    if (warp.thread_rank() == 0)
        s_max[warp.meta_group_rank()] = warp_max;
    block.sync();

    float local_sum = 0.0f;
    for (int c = threadIdx.x; c < n_classes; c += blockDim.x) {
        float p = preds[b * n_classes + c];
        float e = expf(p - warp_max);
        local_sum += e;
    }
    float warp_sum;
    warp_sum = cg::reduce(warp, local_sum, cg::plus<float>());
    if (warp.thread_rank() == 0)
        s_sum[warp.meta_group_rank()] = warp_sum;
    block.sync();

    for (int c = threadIdx.x; c < n_classes; c += blockDim.x) {
        if (fabsf(actuals[b * n_classes + c] - 1.0f) < 1e-6) {
            float x_true = preds[b * n_classes + c];
            loss[b] = -(x_true - warp_max) + logf(warp_sum);
        }
    }
}

Tensor CrossEntropyLoss::forward(const Tensor &preds, const Tensor &actuals) {
    int n = actuals.get_size();
    std::vector<int> shape = actuals.get_shape();
    Tensor loss = Tensor(shape[0]);
    Tensor batch_loss(1);
    int threads = 256;
    int blocks = shape[0];

    ker_cross_entropy_forward<<<blocks, threads>>>(preds.getdata(), actuals.getdata(),
                                                   loss.getdata(), shape[0], n_classes);
    cudaDeviceSynchronize();
    average<<<1, 32>>>(loss.getdata(), blocks);
    loss.reshape({1});

    return loss;
}
} // namespace minitorch
