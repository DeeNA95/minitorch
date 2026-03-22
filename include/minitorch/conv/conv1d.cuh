#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "minitorch/memory_pool.cuh"
#include "minitorch/module.hh"
#include "minitorch/tensor.cuh"

namespace minitorch {

class Conv1d : public Module {
    Tensor input_cache;
    Tensor weight;
    Tensor bias;
    Tensor grad_weight;
    Tensor grad_bias;
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int dilation;
    int groups;
    bool bias_bool;

public:
    // out channel is number of filters
    Conv1d(int in_channels, int out_channels, int kernel_size, int stride, int padding,
               int dilation, int groups, bool bias=false);

    Tensor forward(const Tensor &inputs);
    Tensor backward(const Tensor &gradients_tensor);
    std::vector<Parameter> parameters();
};

} // namespace minitorch
