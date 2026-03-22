#include <cuda_runtime.h>
#include <string>
#include "iostream"
#include "minitorch/activations.cuh"
#include "minitorch/layers.cuh"
#include "minitorch/tensor.cuh"
#include "minitorch/ops.cuh"

using namespace minitorch;

namespace minitorch {
Linear::Linear(int in_features, int out_features)
    : n_weights(in_features * out_features), weights(in_features, out_features),
      bias(1, out_features), grad_weights(in_features, out_features), grad_bias(1, out_features),
      input_cache(1, 1) /* will be overwritten by move op*/ {

    Linear::weights.rand_fill(-0.01f, 0.01f);
    Linear::bias.rand_fill(-0.01f, 0.01f);
}

Linear::~Linear() {
    Linear::n_weights = 0;
    // since linear own the matrices, the destructors will be run automagically
}

// bad idea to
Tensor Linear::forward(const Tensor &inputs /*, std::string act_fn*/) {
    Linear::input_cache = std::move(inputs.copy());
    Tensor weighted = tensor_matmul(inputs, weights);

    Tensor output = Tensor(weighted.get_shape()[0], weighted.get_shape()[1]);
    b_add(weighted, bias, output);

    /*if (act_fn == "relu") {
        relu_forward(output);
    } else if (act_fn == "sigmoid") {
        sigmoid_forward(output);
    } else {
        std::cout << "NO AVAILABLE ACTIVATION CHOSEN" << '\n';
    }
    */

    return output;
}

Tensor Linear::backward(const Tensor &grad_output) {
    // takes grads from upper layer and computes 3 things
    // -grad_weights the error attributable to weight
    // -grad_inputs
    // -grad_bias
    /* forward pass is (I * W) + B
     * so by chain rule for each grad_inputs = grad_outputs * weights;
     * grad_weights = grad_outputs * Inputs
     * bias = grad_outputs
     */
    Tensor grad_inputs = tensor_matmul(
        grad_output,
        tensor_transpose(this->weights)); // this will be passed as grad_output to the lower layer
    this->grad_weights = std::move(tensor_matmul(tensor_transpose(this->input_cache), grad_output));

    // grad_bias should be the sum of grad_ouput along rows to each batch, but since we are
    // currently batchless its just equal to grad_output,
    this->grad_bias = grad_output.copy();

    return grad_inputs;
}

void Linear::fix_weights() {
    float *weights_buffer = new float[weights.get_shape()[1] * weights.get_shape()[0]];
    weights.to_host(weights_buffer);

    for (int i = 0; i < weights.get_shape()[0]; i++) {
        for (int j = 0; j < weights.get_shape()[1]; j++) {
            if (i == j) {
                weights_buffer[i * weights.get_shape()[1] + j] = 1.0f;
            } else {
                weights_buffer[i * weights.get_shape()[1] + j] = 0.0f;
            }
        }
    }
    weights.to_device(weights_buffer);
    delete[] weights_buffer;
    float *bias_buffer = new float[bias.get_shape()[1]];
    bias.to_host(bias_buffer);

    for (int i = 0; i < bias.get_shape()[1]; i++) {
        bias_buffer[i] = 0.0f;
    };

    bias.to_device(bias_buffer);
    delete[] bias_buffer;
}

//
//
Tensor &Linear::get_weights() {
    return weights;
}

std::vector<Parameter> Linear::parameters() {
    return {{&weights, &grad_weights}, {&bias, &grad_bias}};
}
//
Tensor &Linear::get_bias() {
    return bias;
}
const Tensor &Linear::get_grad_bias() const {
    return grad_bias;
}
const Tensor &Linear::get_grad_weights() const {
    return grad_weights;
}
} // namespace minitorch
