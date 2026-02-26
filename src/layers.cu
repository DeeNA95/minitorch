#include <cuda_runtime.h>
#include <string>
#include "iostream"
#include "minitorch/activations.cuh"
#include "minitorch/layers.cuh"
#include "minitorch/matrix.cuh"
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
Matrix Linear::forward(const Matrix &inputs /*, std::string act_fn*/) {
    Linear::input_cache = std::move(inputs.copy());
    Matrix weighted = mat_matmul(inputs, weights);

    Matrix output = Matrix(weighted.getrows(), weighted.getcols());
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

Matrix Linear::backward(Matrix &grad_output) {
    // takes grads from upper layer and computes 3 things
    // -grad_weights the error attributable to weight
    // -grad_inputs
    // -grad_bias
    /* forward pass is (I * W) + B
     * so by chain rule for each grad_inputs = grad_outputs * weights;
     * grad_weights = grad_outputs * Inputs
     * bias = grad_outputs
     */
    Matrix grad_inputs = mat_matmul(
        grad_output,
        mat_transpose(this->weights)); // this will be passed as grad_output to the lower layer
    this->grad_weights = std::move(mat_matmul(mat_transpose(this->input_cache), grad_output));

    // grad_bias should be the sum of grad_ouput along rows to each batch, but since we are
    // currently batchless its just equal to grad_output,
    this->grad_bias = std::move(grad_output);

    return grad_inputs;
}

void Linear::fix_weights() {
    float *weights_buffer = new float[weights.getcols() * weights.getrows()];
    weights.to_host(weights_buffer);

    for (int i = 0; i < weights.getrows(); i++) {
        for (int j = 0; j < weights.getcols(); j++) {
            if (i == j) {
                weights_buffer[i * weights.getcols() + j] = 1.0f;
            } else {
                weights_buffer[i * weights.getcols() + j] = 0.0f;
            }
        }
    }
    weights.to_device(weights_buffer);
    delete[] weights_buffer;
    float *bias_buffer = new float[bias.getcols()];
    bias.to_host(bias_buffer);

    for (int i = 0; i < bias.getcols(); i++) {
        bias_buffer[i] = 0.0f;
    };

    bias.to_device(bias_buffer);
    delete[] bias_buffer;
}

//
Matrix &Linear::get_weights() {
    return weights;
}
//
Matrix &Linear::get_bias() {
    return bias;
}
const Matrix &Linear::get_grad_bias() const {
    return grad_bias;
}
const Matrix &Linear::get_grad_weights() const {
    return grad_weights;
}
} // namespace minitorch
