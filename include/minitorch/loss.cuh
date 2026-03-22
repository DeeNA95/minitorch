#pragma once
#include <cuda_runtime.h>
#include "minitorch/tensor.cuh"

namespace minitorch {

float mse_forward(Tensor &preds, Tensor &actual);
Tensor mse_backward(Tensor &preds, Tensor &actual);
} // namespace minitorch
