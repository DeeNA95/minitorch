#include <cuda_runtime.h>
#include "minitorch/activations.cuh"
#include "minitorch/utils.cuh"

using namespace minitorch;
// for forward pass, INPUT goes in and OUTPUT goes out
// for backward pass, OUTPUT goes in and INPUT goes out
namespace minitorch {

// SIGMOID

__global__ void ker_sigmoid_forward(float *__restrict__ w, int n_cols) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // obtain matrix cell and run sigmoid_forward on it
    w[get_idx_2d(y, x, n_cols)] = (1 / (1 + expf(-w[get_idx_2d(y, x, n_cols)])));
}

void sigmoid_forward(Matrix &weights) {
    float *weight = weights.getdata();
    int n_cols = weights.getcols(), n_rows = weights.getrows();

    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);

    ker_sigmoid_forward<<<blocks, threads>>>(weight, n_cols);
}

__global__ void ker_sigmoid_backward(const float *__restrict__ grad_mat,
                                     const float *__restrict__ out, float *__restrict__ grads_in,
                                     int n_rows, int n_cols) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= n_cols || y >= n_rows)
        return;

    grads_in[get_idx_2d(y, x, n_cols)] = grad_mat[get_idx_2d(y, x, n_cols)] *
                                         out[get_idx_2d(y, x, n_cols)] *
                                         (1.0f - out[get_idx_2d(y, x, n_cols)]);
}

Matrix sigmoid_backward(Matrix &grad_mat, Matrix &out) {
    float *grads = grad_mat.getdata(), *outs = out.getdata();
    int n_cols = grad_mat.getcols(), n_rows = grad_mat.getrows();

    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);

    Matrix grad_in = Matrix(n_rows, n_cols);

    ker_sigmoid_backward<<<blocks, threads>>>(grads, outs, grad_in.getdata(), n_rows, n_cols);
    cudaDeviceSynchronize();

    return grad_in;
}

// RELU

__global__ void ker_relu_forward(float *__restrict__ w, int n_cols) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    // obtain matrix cell and run relu_forward on it
    w[get_idx_2d(y, x, n_cols)] = max(0.0f, w[get_idx_2d(y, x, n_cols)]);
}

void relu_forward(Matrix &weights) {
    float *weight = weights.getdata();
    int n_cols = weights.getcols(), n_rows = weights.getrows();

    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);

    ker_relu_forward<<<blocks, threads>>>(weight, n_cols);
}

__global__ void ker_relu_backward(const float *__restrict__ grad_mat,
                                  const float *__restrict__ relu_in, float *__restrict__ grads_in,
                                  int n_rows, int n_cols) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= n_cols || y >= n_rows)
        return;

    grads_in[get_idx_2d(y, x, n_cols)] =
        (relu_in[get_idx_2d(y, x, n_cols)] > 0.0f) ? grad_mat[get_idx_2d(y, x, n_cols)] : 0.0f;
}

Matrix relu_backward(Matrix &grad_mat, Matrix &relu_in) {
    float *grads = grad_mat.getdata(), *relu_ins = relu_in.getdata();
    int n_cols = grad_mat.getcols(), n_rows = grad_mat.getrows();

    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);

    Matrix grad_in = Matrix(n_rows, n_cols);

    ker_relu_backward<<<blocks, threads>>>(grads, relu_ins, grad_in.getdata(), n_rows, n_cols);
    cudaDeviceSynchronize();

    return grad_in;
}
} // namespace minitorch
