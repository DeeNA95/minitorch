#pragma once
#include <cuda_runtime.h>
#include <vector>
#include "minitorch/module.hh"
#include "minitorch/tensor.cuh"

namespace minitorch {

class MaxPool1d : public Module {
private:
    Tensor arg_max;
    std::vector<int> input_shape;
    int kernel_size;
    int stride;
    int padding;
    int dilation;

public:
    MaxPool1d(int kernel_size, int stride, int padding, int dilation);
    Tensor forward(const Tensor &inputs);
    Tensor backward(const Tensor &gradients_tensor);
};
} // namespace minitorch
