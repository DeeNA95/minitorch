#include <cuda_runtime.h>
#include <iostream>
#include "minitorch/pool/maxpool1d.cuh"
#include "minitorch/pool/pool_utils.cuh"

namespace minitorch {

MaxPool1d::MaxPool1d(int kernel_size, int stride, int padding, int dilation)
    : kernel_size(kernel_size), stride(stride), padding(padding), dilation(dilation) {
    this->arg_max = Tensor(1, 1);
}

Tensor MaxPool1d::forward(const Tensor &inputs) {
    this->input_shape = inputs.get_shape();
    int batch = input_shape[0], in_channels = input_shape[1], in_length = input_shape[2];
    int out_length = (in_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    Tensor output = Tensor(batch, in_channels, out_length);

    this->arg_max = Tensor(batch, in_channels, out_length);

    int threads = 256;
    dim3 blocks((out_length + threads - 1) / threads, batch * in_channels);

    maxpool1d_forward<<<blocks, threads>>>(inputs.getdata(), output.getdata(),
                                           this->arg_max.getdata(), in_channels, in_length,
                                           out_length, kernel_size, stride, padding, dilation);
    cudaDeviceSynchronize();
    return output;
}

Tensor MaxPool1d::backward(const Tensor &gradients_tensor) {
    Tensor grad_input(this->input_shape);
    grad_input.fill(0.0f);
    std::vector<int> shape = gradients_tensor.get_shape();
    int out_length = shape[2];
    int threads = 256;
    dim3 blocks((out_length + threads - 1) / threads, this->input_shape[0] * this->input_shape[1]);
    maxpool1d_backward<<<blocks, threads>>>(gradients_tensor.getdata(), grad_input.getdata(),
                                            this->arg_max.getdata(), this->input_shape[1],
                                            out_length);

    cudaDeviceSynchronize();
    return grad_input;
}
} // namespace minitorch
