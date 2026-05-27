#include <cuda_runtime.h>
#include <vector>
#include "minitorch/conv/conv1d.cuh"
#include "minitorch/conv/conv_utils.cuh"
#include "minitorch/module.hh"

namespace minitorch {

Conv1d::Conv1d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
               int dilation, int groups, bool bias)
    : weight(out_channels, in_channels, kernel_size),
      grad_weight(out_channels, in_channels, kernel_size), kernel_size(kernel_size), stride(stride),
      padding(padding), dilation(dilation), groups(groups), bias_bool(bias),
      in_channels(in_channels), out_channels(out_channels) {

    if (bias) {
        this->bias = Tensor(out_channels);
        this->bias.uniform_initialisation(0.01f);
        this->grad_bias = Tensor(out_channels);
        this->grad_bias.uniform_initialisation(0.01f);
    }
    this->weight.uniform_initialisation(0.01f);
    this->grad_weight.uniform_initialisation(0.01f);
}

std::vector<Parameter> Conv1d::parameters() {
    std::vector<Parameter> params;
    params.push_back({&this->weight, &this->grad_weight});
    if (this->bias_bool) {
        params.push_back({&this->bias, &this->grad_bias});
    }
    return params;
}

Tensor Conv1d::forward(const Tensor &inputs) { // takes and input of (batch,in_channel,in_len)
    this->input_shape = inputs.get_shape();
    int batch = inputs.get_shape()[0];
    int output_len =
        (inputs.get_shape()[2] + 2 * this->padding - this->dilation * (this->kernel_size - 1) - 1) /
            this->stride +
        1;

    Tensor output(batch, this->out_channels, output_len); // batch, out_channels, output_len
    Tensor col_matrix(batch, this->in_channels * this->kernel_size, output_len);

    // making sure 1thread to 1output
    int threads = 256;
    // writing blocks to be such that each dim in blocks handles, x: output being
    //  calculated of len output_len divided by n_threads, 2:each filter or out_channel being run of
    //  len out_channels and 3: each batch being handled

    dim3 blocks((output_len + threads - 1) / threads, this->in_channels * this->kernel_size, batch);

    // reshape wrights for tensormatmul with inputs
    Tensor reshaped_weight = weight.copy();
    reshaped_weight.reshape({1, this->out_channels, this->in_channels * this->kernel_size});

    im2col1d<<<blocks, threads>>>(inputs.getdata(), col_matrix.getdata(), this->kernel_size,
                                  this->in_channels, this->stride, this->padding, this->dilation,
                                  inputs.get_shape()[2], output_len);

    output = reshaped_weight * col_matrix;

    if (this->bias_bool) {
        output = bias_add(output, this->bias);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "im2col1d kernel error " << cudaGetErrorString(err) << '\n';
    }

    this->col_matrix_cache = std::move(col_matrix.copy());

    // conv1d_forward<<<blocks,threads>>>()

    return output;
}

Tensor Conv1d::backward(const Tensor &gradients_tensor) { // shaped (batch, out_channels,
                                                          // out_length)
    std::vector<int> shape = gradients_tensor.get_shape();
    int in_len = this->input_shape[2];
    int batch = shape[0];
    int out_channels = shape[1];
    int output_len = shape[2];

    // forsward pass went, input -> [im2col] -> col_mattrix -> [matmul] with weights -> output
    // so for backward pass, gradients_tensor -> [matmul backprop] -> grad_col_matrix -> [col2im] ->
    // grad_inputs
    // gradients tensor is from the loss/upper layer hence dL/dY. dL/dW is calculated by
    //
    // dL/dY * Y^T * dL/dX
    Tensor reshaped_weight = this->weight.copy();
    // does not change layout of tensor in memory so use of tensor_transpose is needed during may
    // mul
    reshaped_weight.reshape({1,  this->out_channels,this->in_channels * this->kernel_size});

    Tensor grad_col_matrix = tensor_transpose(reshaped_weight) *
                             gradients_tensor; // shape batch, in_channels * kernel_size, out_len

    Tensor grad_input(batch, this->in_channels, in_len);

    int threads = 256;
    // writing blocks to be such that each dim in blocks handles, x: output being
    //  calculated of len output_len divided by n_threads, 2:each filter or out_channel being run of
    //  len out_channels and 3: each batch being handled

    dim3 blocks((output_len + threads - 1) / threads, this->in_channels * this->kernel_size, batch);

    col2im1d<<<blocks, threads>>>(grad_col_matrix.getdata(), grad_input.getdata(),
                                  this->kernel_size, this->in_channels, this->stride, this->padding,
                                  this->dilation, in_len, output_len);

    // step 2, grad_weight
    Tensor batch_grad_weights =
        tensor_matmul(gradients_tensor, tensor_transpose(this->col_matrix_cache));
    this->grad_weight = batch_grad_weights.sum(0);
    this->grad_weight.reshape({this->out_channels, this->in_channels, this->kernel_size});

    // step 3, grad_bias
    if (this->bias_bool) {
        Tensor grad_sum = gradients_tensor.copy();
        this->grad_bias = grad_sum.sum(0, true).sum(2, true);
        this->grad_bias.reshape({this->out_channels});
    }

    return grad_input;
}
} // namespace minitorch
