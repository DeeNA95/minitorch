#include <cmath>
#include <cuda_runtime.h>
#include "minitorch/activations.cuh"
#include "minitorch/module.hh"
#include "minitorch/utils.cuh"
#define _USE_MATH_DEFINES // Required for M_PI to be defined

using namespace minitorch;
// for forward pass, INPUT goes in and OUTPUT goes output_matrixput_matrix
// for backward pass, OUTPUT goes in and INPUT goes output_matrixput_matrix
namespace minitorch {

// SIGMOID

__global__ void ker_sigmoid_forward(float *__restrict__ input, int n_cols, int n_rows) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= n_cols || y >= n_rows)
        return;

    // obtain matrix cell and run Sigmoid::forward on it
    input[get_idx_2d(y, x, n_cols)] = (1 / (1 + expf(-input[get_idx_2d(y, x, n_cols)])));
}

Matrix Sigmoid::forward(const Matrix &inputs) {
    Matrix output_matrix = inputs.copy();

    float *input = output_matrix.getdata();
    int n_cols = output_matrix.getcols(), n_rows = output_matrix.getrows();

    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);

    ker_sigmoid_forward<<<blocks, threads>>>(input, n_cols, n_rows);

    this->output_cache = output_matrix.copy();

    return output_matrix;
}

__global__ void ker_sigmoid_backward(const float *__restrict__ gradients_matrix,
                                     const float *__restrict__ output_matrix,
                                     float *__restrict__ grads_in, int n_rows, int n_cols) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= n_cols || y >= n_rows)
        return;

    grads_in[get_idx_2d(y, x, n_cols)] = gradients_matrix[get_idx_2d(y, x, n_cols)] *
                                         output_matrix[get_idx_2d(y, x, n_cols)] *
                                         (1.0f - output_matrix[get_idx_2d(y, x, n_cols)]);
}

Matrix Sigmoid::backward(const Matrix &gradients_matrix) {
    float *grads = gradients_matrix.getdata(), *output_matrix = this->output_cache.getdata();
    int n_cols = gradients_matrix.getcols(), n_rows = gradients_matrix.getrows();

    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);

    Matrix grad_in(n_rows, n_cols);

    ker_sigmoid_backward<<<blocks, threads>>>(grads, output_matrix, grad_in.getdata(), n_rows,
                                              n_cols);
    cudaDeviceSynchronize();

    return grad_in;
}

// RELU

__global__ void ker_relu_forward(float *__restrict__ input, int n_cols, int n_rows) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= n_cols || y >= n_rows)
        return;

    // obtain matrix cell and run Relu::forward on it
    input[get_idx_2d(y, x, n_cols)] = max(0.0f, input[get_idx_2d(y, x, n_cols)]);
}

Matrix Relu::forward(const Matrix &inputs) {
    Matrix output_matrix = inputs.copy();

    float *input = output_matrix.getdata();
    int n_cols = output_matrix.getcols(), n_rows = output_matrix.getrows();

    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);

    ker_relu_forward<<<blocks, threads>>>(input, n_cols, n_rows);
    this->output_cache = output_matrix.copy();
    return output_matrix;
}

__global__ void ker_relu_backward(const float *__restrict__ gradients_matrix,
                                  const float *__restrict__ relu_in, float *__restrict__ grads_in,
                                  int n_rows, int n_cols) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= n_cols || y >= n_rows)
        return;

    grads_in[get_idx_2d(y, x, n_cols)] = (relu_in[get_idx_2d(y, x, n_cols)] > 0.0f)
                                             ? gradients_matrix[get_idx_2d(y, x, n_cols)]
                                             : 0.0f;
}

Matrix Relu::backward(const Matrix &gradients_matrix) {
    float *grads = gradients_matrix.getdata(), *relu_ins = this->output_cache.getdata();
    int n_cols = gradients_matrix.getcols(), n_rows = gradients_matrix.getrows();

    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);

    Matrix grad_in = Matrix(n_rows, n_cols);

    ker_relu_backward<<<blocks, threads>>>(grads, relu_ins, grad_in.getdata(), n_rows, n_cols);
    cudaDeviceSynchronize();

    return grad_in;
}

// GELU

__device__ float dev_tanh(float x) {
    float numerator = expf(2 * x) - 1;
    float denominator = expf(2 * x) + 1;

    return numerator / denominator;
}

__global__ void ker_gelu_forward(float *input, int n_cols, int n_rows) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= n_cols || y >= n_rows)
        return;

    // obtain matrix cell and run Gelu::forward on it
    auto q = input[get_idx_2d(y, x, n_cols)];
    auto tanh_arg = sqrtf(2.0 / M_PI) * (q + 0.044715 * q * q * q);

    input[get_idx_2d(y, x, n_cols)] = 0.5 * q * (1 + dev_tanh(tanh_arg));
}

Matrix Gelu::forward(const Matrix &inputs) {
    Matrix output_matrix = inputs.copy();
    float *input = output_matrix.getdata();
    int n_cols = output_matrix.getcols(), n_rows = output_matrix.getrows();

    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);

    ker_gelu_forward<<<blocks, threads>>>(input, n_cols, n_rows);

    this->input_cache = inputs.copy();
    return output_matrix;
}

__global__ void ker_gelu_backward(const float *__restrict__ gradients_matrix,
                                  const float *__restrict__ gelu_in, float *__restrict__ grads_in,
                                  int n_rows, int n_cols) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= n_cols || y >= n_rows)
        return;

    auto q = gelu_in[get_idx_2d(y, x, n_cols)];
    // estimating sech^2 with 1 - tanh^2
    auto z = sqrtf(2.0 / M_PI) * (q + 0.044715 * q * q * q);
    auto tanhed = dev_tanh(z);
    auto seched2 = (1 - tanhed * tanhed);

    grads_in[get_idx_2d(y, x, n_cols)] =
        (0.5 * (1 + tanhed) + 0.5 * q * seched2 * sqrtf(2.0f / M_PI) * (1 + 3 * 0.044715 * q * q)) *
        gradients_matrix[get_idx_2d(y, x, n_cols)];
}

Matrix Gelu::backward(const Matrix &gradients_matrix) {
    float *grads = gradients_matrix.getdata(), *gelu_ins = this->input_cache.getdata();
    int n_cols = gradients_matrix.getcols(), n_rows = gradients_matrix.getrows();

    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);

    Matrix grad_in = Matrix(n_rows, n_cols);

    ker_gelu_backward<<<blocks, threads>>>(grads, gelu_ins, grad_in.getdata(), n_rows, n_cols);
    cudaDeviceSynchronize();

    return grad_in;
}

} // namespace minitorch
