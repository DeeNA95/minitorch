#include <cstddef>
#include "cooperative_groups.h"
#include "cuda_runtime.h"
#include "minitorch/optim.cuh"
namespace cg = cooperative_groups;
using namespace minitorch;

namespace minitorch {

__global__ void sgd(float *__restrict__ weights, const float *__restrict__ grad_weights, float lr,
                    int n) {

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    for (int tid = grid.thread_rank(); tid < n; tid += grid.size()) {
        weights[tid] -= lr * grad_weights[tid];
    }
}

void sgd_update(Matrix &weights, const Matrix &grad_weights, float lr) {
    float *w = weights.getdata();
    const float *gw = grad_weights.getdata();
    int n_rows = weights.getrows(), n_cols = weights.getcols();

    int threads = 256;

    sgd<<<1, threads>>>(w, gw, lr, n_rows * n_cols);
    cudaDeviceSynchronize();
}

} // namespace minitorch
