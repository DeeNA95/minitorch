#include <cuda_runtime.h>
#include "minitorch/activations.cuh"

using namespace minitorch;

namespace minitorch {

__global__ void ker_sigmoid(float *__restrict__ w, int n_cols) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // obtain matrix cell and run sigmoid on it
    auto idx = [&n_cols](int y, int x) { return (y * n_cols + x); };
    w[idx(y, x)] = (1 / (1 + expf(-w[idx(y, x)])));
}

void sigmoid(Matrix &weights) {
    float *weight = weights.getdata();
    int n_cols = weights.getcols(), n_rows = weights.getrows();

    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);

    ker_sigmoid<<<blocks, threads>>>(weight, n_cols);
}

__global__ void ker_relu(float *__restrict__ w, int n_cols) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // obtain matrix cell and run relu on it
    auto idx = [&n_cols](int y, int x) { return (y * n_cols + x); };
    w[idx(y, x)] = max(0.0f, w[idx(y, x)]);
}

void relu(Matrix &weights) {
    float *weight = weights.getdata();
    int n_cols = weights.getcols(), n_rows = weights.getrows();

    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);

    ker_relu<<<blocks, threads>>>(weight, n_cols);
}

} // namespace minitorch
