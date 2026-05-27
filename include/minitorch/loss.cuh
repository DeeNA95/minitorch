#pragma once
#include <cuda_runtime.h>
#include "minitorch/tensor.cuh"

namespace minitorch {
// TODO: convert to class
float mse_forward(Tensor &preds, Tensor &actual);
Tensor mse_backward(Tensor &preds, Tensor &actual);

class CrossEntropyLoss {
public:
    CrossEntropyLoss(int n_classes);
    Tensor forward(const Tensor &preds, const Tensor &actuals);
    Tensor backward(const Tensor &preds, const Tensor &actuals);

private:
    int n_classes;
};
} // namespace minitorch
