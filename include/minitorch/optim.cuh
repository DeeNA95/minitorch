#pragma once
#include <vector>
#include "minitorch/tensor.cuh"

using namespace minitorch;

namespace minitorch {

void sgd_update(Tensor &weights, const Tensor &grad_weights, float lr);

// adam requires keeping a set of matrices to store optim weights
class Adam {
private:
    float lr, beta1, beta2, epsilon, weight_decay;
    int t; // timestep

    // for each param, we store m and v buffers
    std::vector<Tensor *> params; // ponter to weights & bias
    std::vector<const Tensor *> grads;
    std::vector<Tensor> m_buffers; // first moment (momentum)
    std::vector<Tensor> v_buffers; // second moment )velocity)

public:
    Adam(float lr, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-9f,
         float weight_decay = 0.01f);

    void add_parameter(Tensor &param, const Tensor &grad);

    void step();

    void zero_grad();
};

} // namespace minitorch
