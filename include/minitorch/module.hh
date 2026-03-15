#pragma once
#include <vector>
#include "minitorch/matrix.cuh"
using namespace minitorch;
namespace minitorch {

struct Parameter
{
    Matrix *weight;
    const Matrix *grad;
};

class Module {
public:
    virtual ~Module() = default;
    virtual Matrix forward(const Matrix &inputs) = 0;
    virtual Matrix backward(const Matrix &gradients_matrix) = 0;
    virtual std::vector<Parameter> parameters() {
        return {};
    }
};
} // namespace minitorch
