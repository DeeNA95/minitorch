#include <cuda_runtime.h>
#include "minitorch/conv/conv1d.cuh"
#include "minitorch/conv/conv_utils.cuh"

namespace minitorch {

Conv1d::Conv1d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
               int dilation, int groups, bool bias)
    : weight(out_channels, in_channels, kernel_size),
      grad_weight(out_channels, in_channels, kernel_size), kernel_size(kernel_size), stride(stride),
      padding(padding), dilation(dilation), groups(groups), bias_bool(bias) , in_channels(in_channels), out_channels(out_channels){

    if (bias) {
        this->bias = Tensor(out_channels, 1);
        this->bias.uniform_initialisation(0.01f);
        this->grad_bias = Tensor(out_channels, 1);
        this->grad_bias.uniform_initialisation(0.01f);
    }
    this->weight.uniform_initialisation(0.01f);
    this->grad_weight.uniform_initialisation(0.01f);
}

Tensor Conv1d::forward(const Tensor &inputs) { // takes and input of (batch,channel,len)
    this->input_cache = std::move(inputs.copy());
    int batch = inputs.get_shape()[0];
    int output_len = ((inputs.get_shape()[2] + 2*this->padding - this->dilation*(this->kernel_size-1))/this->stride+1) + 1;

    Tensor output(batch, this->out_channels, output_len); //batch, out_channels, output_len
    Tensor col_matrix(batch, this->in_channels * this->kernel_size, output_len);

    //making sure 1thread to 1output
    int threads = 256;
    //writing blocks to be such that each dim in blocks handles, x: output being
    // calculated of len output_len divided by n_threads, 2:each filter or out_channel being run of
    // len out_channels and 3: each batch being handled

    dim3 blocks((output_len+threads-1)/threads, this->in_channels * this->kernel_size, batch);

    //reshape wrights for tensormatmul with inputs
    Tensor reshaped_weight = weight.copy();
    reshaped_weight.reshape({1,this->out_channels, this->in_channels*this->kernel_size});

    im2col1d<<<blocks,threads>>>(
        inputs.getdata(),
        col_matrix.getdata(),
        this->kernel_size,
        this->in_channels,
        this->stride,
        this->padding,
        this->dilation,
        inputs.get_shape()[2],
        output_len
    );

    output = reshaped_weight*col_matrix;

    if (this->bias_bool) {
        output = bias_add(output, this->bias);
    }



    // conv1d_forward<<<blocks,threads>>>()

    return output;
}

Tensor Conv1d::backward(const Tensor &gradients_tensor) {
    if (this->bias_bool == true) {
        Tensor grad_copy = gradients_tensor.copy();
        this->grad_bias  = grad_copy.sum(0, true).copy();
    }
    //transpose the tensor
    std::vector<int> shape = gradients_tensor.get_shape();
    int s1 = shape[shape.size()-1];
    int s2 = shape[shape.size()-2];
    shape[shape.size()-1] = s2;
    shape[shape.size()-2] = s1;

    Tensor reshaped_weight = this->weight.copy();
    reshaped_weight.reshape(shape);

    Tensor grad_inputs = gradients_tensor * reshaped_weight;

    Tensor reshaped_input_cache = this->input_cache.copy();
    reshaped_input_cache.reshape(shape);

    this->grad_weight = std::move(reshaped_input_cache * gradients_tensor);

    return grad_inputs;



}
} // namespace minitorch
