#include <iostream>
#include "cuda_runtime.h"
#include "minitorch/activations.cuh"
#include "minitorch/layers.cuh"
#include "minitorch/loss.cuh"
#include "minitorch/optim.cuh"
#define LR 0.0003

#include <cstdlib>

using namespace minitorch;

int main(int argc, char *argv[]) {
    int epochs = 5000;
    float lr = LR;
    if (argc > 1) {
        epochs = std::atoi(argv[1]);
    }
    if (argc > 2) {
        lr = std::atof(argv[2]);
    }
    std::cout << "Starting training for " << epochs << " epochs with LR=" << lr << '\n';
    // an xor test using 2 layers
    Matrix input = Matrix(4, 2);
    Matrix target = Matrix(4, 1);

    float *in_data = new float[8];
    float *targ_data = new float[4];

    for (int i = 0; i < 8; i++) {
        if (i == 0 || i == 1 || i == 5 || i == 2) {
            in_data[i] = 0.0f;
        } else {
            in_data[i] = 1.0f;
        }
    }

    for (int i = 0; i < 4; i++) {
        if (i == 0 || i == 3) {
            targ_data[i] = 0.0f;
        } else {
            targ_data[i] = 1.0f;
        }
    }

    input.to_device(in_data);
    target.to_device(targ_data);

    // network
    Linear l1 = Linear(2, 4);
    Linear l2 = Linear(4, 1);
    float loss;

    for (int epoch = 0; epoch < epochs; epoch++) {
        // forward
        Matrix out1 = l1.forward(input);
        Matrix a1 = out1.copy();
        sigmoid_forward(a1);
        Matrix out2 = l2.forward(a1);
        Matrix a2 = out2.copy();
        sigmoid_forward(a2);

        // loss this just for printing the one that goes into the backward is handled in
        // mse_backward
        if (epoch % (epochs / 10 > 0 ? epochs / 10 : 1) == 0 || epoch == epochs - 1) {
            loss = mse_forward(a2, target);
            std::cout << "EPOCH " << epoch << " LOSS " << loss << '\n';
        }
        // backward
        Matrix mse_back = mse_backward(a2, target);
        Matrix a2_back = sigmoid_backward(mse_back, a2);
        Matrix l2_back = l2.backward(a2_back);
        Matrix a1_back = sigmoid_backward(l2_back, a1);
        Matrix l1_back = l1.backward(a1_back);

        // update weights
        sgd_update(l1.get_weights(), l1.get_grad_weights(), lr);
        sgd_update(l2.get_weights(), l2.get_grad_weights(), lr);
        sgd_update(l1.get_bias(), l1.get_grad_bias(), lr);
        sgd_update(l2.get_bias(), l2.get_grad_bias(), lr);
    }

    std::cout << "TRAINING COMPLETE" << '\n' << "LAST LOSS: " << loss << '\n';
}
