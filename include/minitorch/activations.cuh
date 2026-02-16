#pragma once
#include <cuda_runtime.h>
#include "minitorch/matrix.cuh"

namespace minitorch {

void sigmoid(Matrix &weights);
void relu(Matrix &weights);
} // namespace minitorch
