#pragma once
#include <cuda_runtime.h>
#include "minitorch/matrix.cuh"

namespace minitorch {

float mse_forward(Matrix &preds, Matrix &actual);
Matrix mse_backward(Matrix &preds, Matrix &actual);
} // namespace minitorch
