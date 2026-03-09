#pragma once
#include <vector>
#include "minitorch/matrix.cuh"

using namespace minitorch;

namespace minitorch {

void sgd_update(Matrix &weights, const Matrix &grad_weights, float lr);

// adam requires keeping a set of matrices to store optim weights
class Adam {
private:
    float lr, beta1, beta2, epsilon, weight_decay;
    int t; // timestep

    // for each param, we store m and v buffers
    std::vector<Matrix *> params; // ponter to weights & bias
    std::vector<const Matrix *> grads;
    std::vector<Matrix> m_buffers; // first moment (momentum)
    std::vector<Matrix> v_buffers; // second moment )velocity)

public:
    Adam(float lr, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-9f,
         float weight_decay = 0.01f);

    void add_parameter(Matrix &param, const Matrix &grad);

    void step();

    void zero_grad();
};

} // namespace minitorch
