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
      bias(1, out_features) {

    Linear::weights.rand_fill(-5, 5);
    Linear::bias.rand_fill(-1, 1);
}

Linear::~Linear() {
    Linear::n_weights = 0;
    // since linear own the matrices, the destructors will be run automagically
}

Matrix Linear::forward(const Matrix &inputs, std::string act_fn) {

    Matrix weighted = mat_matmul(inputs, weights);

    Matrix output = Matrix(weighted.getrows(), weighted.getcols());
    b_add(weighted, bias, output);

    if (act_fn == "relu") {
        relu(output);
    } else if (act_fn == "sigmoid") {
        sigmoid(output);
    } else {
        std::cout << "NO AVAILABLE ACTIVATION CHOSEN" << '\n';
    }

    return output;
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

    for (int i = 0; i < bias.getcols(); i++) { bias_buffer[i] = 0.0f; };

    bias.to_device(bias_buffer);
    delete[] bias_buffer;
}

//
// Matrix &Linear::get_weights() const {
//     return weights;
// }
//
// Matrix &Linear::get_bias() const {
//     return bias;
// }

} // namespace minitorch
