#include <vector>
#include "minitorch/activations.cuh"
#include "minitorch/layers.cuh"
#include "minitorch/matrix.cuh"
#include "minitorch/module.hh"

using namespace minitorch;

namespace minitorch {

class Sequential : public Module {
private:
    std::vector<Module *> layers;

public:
    Sequential(std::vector<Module *> layers) : layers(layers) {};

    Matrix forward(const Matrix &inputs) override {
        Matrix out = inputs.copy();

        for (auto layer : layers) {
            out = layer->forward(out);
        }
        return out;
    }

    Matrix backward(const Matrix &gradients_matrix) override {
        Matrix current_grad = gradients_matrix.copy();

        // Backward pass must iterate in reverse!
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            current_grad = (*it)->backward(current_grad);
        }

        return current_grad;
    }

    std::vector<Matrix *> parameters() override {
        std::vector<Matrix *> params;
        for (auto layer : layers) {
            auto layer_params = layer->parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }
};

} // namespace minitorch
