#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include "minitorch/tensor.cuh"
#include "minitorch/module.hh"
namespace minitorch {

class Sigmoid : public Module {
private:
    Tensor output_cache; // output from farward will be cached here for backward pass here

public:
    Sigmoid() : output_cache(0, 0) {};

    Tensor forward(const Tensor &inputs) ;
    Tensor backward(const Tensor &gradients_matrix) ;
};

class Relu : public Module {
private:
    Tensor output_cache;

public:
    Relu() : output_cache(0, 0) {};

    Tensor forward(const Tensor &inputs) ;

    Tensor backward(const Tensor &gradients_matrix) ;
};

class Gelu : public Module {
private:
    Tensor input_cache;

public:
    Gelu() : input_cache(0, 0) {};

    Tensor forward(const Tensor &inputs) ;

    Tensor backward(const Tensor &gradients_matrix) ;
};

// void sigmoid_forward(Tensor &inputs);
// void relu_forward(Tensor &inputs);
// void gelu_forward(Tensor &inputs);
// Tensor relu_backward(Tensor &grad_mat, Tensor &out);
// Tensor gelu_backward(Tensor &grad_mat, Tensor &out);
//
// Tensor sigmoid_backward(Tensor &grad_mat, Tensor &out_pre_act);

__device__ float dev_tanh(float x);
} // namespace minitorch
