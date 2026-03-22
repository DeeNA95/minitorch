#pragma once
#include <vector>
#include "minitorch/tensor.cuh"
using namespace minitorch;
namespace minitorch {

struct Parameter
{
    Tensor *weight;
    const Tensor *grad;
};

class Module {
public:
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor &inputs) = 0;
    virtual Tensor backward(const Tensor &gradients_Tensor) = 0;
    virtual std::vector<Parameter> parameters() {
        return {};
    }
};
} // namespace minitorch
