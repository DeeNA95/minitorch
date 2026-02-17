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

public:
    Linear(int in_features, int out_features);
    ~Linear();

    Matrix forward(const Matrix &inputs, std::string act_fn);
    Matrix &get_weights() const;
    Matrix &get_bias() const;
    void fix_weights();
};

} // namespace minitorch
