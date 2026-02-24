#pragma once
#include <cuda_runtime.h>
#include "minitorch/matrix.cuh"

namespace minitorch {

void sigmoid_forward(Matrix &weights);
void relu_forward(Matrix &weights);
Matrix relu_backward(Matrix &grad_mat, Matrix &out);
Matrix sigmoid_backward(Matrix &grad_mat, Matrix &out_pre_act);
} // namespace minitorch
