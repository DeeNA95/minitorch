#pragma once
#include <algorithm>
#include <cuda_runtime.h>
#include "minitorch/matrix.cuh"
#include "minitorch/module.hh"
namespace minitorch {

class Sigmoid : public Module {
private:
    Matrix output_cache; // output from farward will be cached here for backward pass here

public:
    Sigmoid() : output_cache(0, 0) {};

    Matrix forward(const Matrix &inputs) override;
    Matrix backward(const Matrix &gradients_matrix) override;
};

class Relu : public Module {
private:
    Matrix output_cache;

public:
    Relu() : output_cache(0, 0) {};

    Matrix forward(const Matrix &inputs) override;

    Matrix backward(const Matrix &gradients_matrix) override;
};

class Gelu : public Module {
private:
    Matrix input_cache;

public:
    Gelu() : input_cache(0, 0) {};

    Matrix forward(const Matrix &inputs) override;

    Matrix backward(const Matrix &gradients_matrix) override;
};

// void sigmoid_forward(Matrix &inputs);
// void relu_forward(Matrix &inputs);
// void gelu_forward(Matrix &inputs);
// Matrix relu_backward(Matrix &grad_mat, Matrix &out);
// Matrix gelu_backward(Matrix &grad_mat, Matrix &out);
//
// Matrix sigmoid_backward(Matrix &grad_mat, Matrix &out_pre_act);

__device__ float dev_tanh(float x);
} // namespace minitorch
