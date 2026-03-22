#include <vector>
#include "minitorch/activations.cuh"
#include "minitorch/layers.cuh"
#include "minitorch/tensor.cuh"
#include "minitorch/module.hh"

using namespace minitorch;

namespace minitorch {

class Sequential : public Module {
private:
    std::vector<Module *> layers;

public:
    Sequential(std::vector<Module *> layers) : layers(layers) {};

    Tensor forward(const Tensor &inputs) override {
        Tensor out = inputs.copy();

        for (auto layer : layers) {
            out = layer->forward(out);
        }
        return out;
    }

    Tensor backward(const Tensor &gradients_matrix) override {
        Tensor current_grad = gradients_matrix.copy();

        // Backward pass must iterate in reverse!
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            current_grad = (*it)->backward(current_grad);
        }

        return current_grad;
    }

    std::vector<Parameter> parameters() override {
        std::vector<Parameter> params;
        for (auto layer : layers) {
            auto layer_params = layer->parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }
};

} // namespace minitorch
