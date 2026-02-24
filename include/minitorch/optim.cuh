#pragma once
#include "minitorch/matrix.cuh"

using namespace minitorch;

namespace minitorch {
void sgd_update(Matrix &weights, const Matrix &grad_weights, float lr);
}
