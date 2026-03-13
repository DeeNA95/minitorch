#pragma once
#include <cuda_runtime.h>
#include <string>
#include "minitorch/matrix.cuh"
#include "minitorch/module.hh"
#include "minitorch/ops.cuh"

namespace minitorch {

class Linear : public Module {
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

    Matrix forward(const Matrix &inputs) override;
    Matrix backward(const Matrix &grad_outputs) override;
    std::vector<Parameter> parameters() override;
    Matrix &get_weights();
    Matrix &get_bias();
    const Matrix &get_grad_weights() const;
    const Matrix &get_grad_bias() const;
    void fix_weights();
};

} // namespace minitorch
