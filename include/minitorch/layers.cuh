#pragma once
#include <cuda_runtime.h>
#include <string>
#include "minitorch/tensor.cuh"
#include "minitorch/module.hh"
#include "minitorch/ops.cuh"

namespace minitorch {

class Linear : public Module {
private:
    int n_weights;
    Tensor weights;
    Tensor bias;
    Tensor grad_weights; // gradients for weights
    Tensor grad_bias;
    Tensor input_cache;

public:
    Linear(int in_features, int out_features);
    ~Linear();

    Tensor forward(const Tensor &inputs) override;
    Tensor backward(const Tensor &grad_outputs) override;
    std::vector<Parameter> parameters() override;
    Tensor &get_weights();
    Tensor &get_bias();
    const Tensor &get_grad_weights() const;
    const Tensor &get_grad_bias() const;
    void fix_weights();
};

} // namespace minitorch
