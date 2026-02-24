#pragma once
#include <cuda_runtime.h>
#include <string>
#include "minitorch/matrix.cuh"
#include "minitorch/ops.cuh"

namespace minitorch {

class Linear {
private:
    int n_weights;
    Matrix weights;
    Matrix bias;
    Matrix grad_weights; // gradients for weights
    Matrix grad_bias;
    Matrix input_cache;

public:
    Linear(int in_features, int out_features);
    ~Linear();

    Matrix forward(const Matrix &inputs);
    Matrix backward(Matrix &grad_outputs);
    Matrix &get_weights();
    Matrix &get_bias();
    const Matrix &get_grad_weights() const;
    const Matrix &get_grad_bias() const;
    void fix_weights();
};

} // namespace minitorch
