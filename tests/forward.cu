#include <cuda_runtime.h>
#include <iostream>
#include "minitorch/activations.cuh"
#include "minitorch/layers.cuh"
#include "minitorch/loss.cuh"
#include "minitorch/tensor.cuh"
#include "minitorch/ops.cuh"

using namespace minitorch;
int main() {
    Tensor input = Tensor(12, 6);
    input.uniform_initialisation(10.0f);

    Linear lin_layer = Linear(input.get_shape()[1], 3);

    Tensor output = lin_layer.forward(input);
    Relu relu;
    output = relu.forward(output);

    std::cout << "FORWARD PASS COMPLETE" << '\n';
    std::cout << "INPUT" << '\n';
    input.print();
    std::cout << '\n';
    std::cout << "OUTPUT" << '\n';
    output.print();

    std::cout << '\n' << '\n' << "TESTING ACTIVATIONS" << '\n';
    Tensor relu_test = Tensor(5, 5);
    relu_test.uniform_initialisation(10.0f); // skewed negative to get more negs
    relu_test.print();
    relu_test = relu.forward(relu_test);
    relu_test.print();

    Tensor sigmoid_test = Tensor(5, 5);
    sigmoid_test.uniform_initialisation(10.0f); // should fit to 0,1
    sigmoid_test.print();
    Sigmoid sigmoid;
    sigmoid_test = sigmoid.forward(sigmoid_test);
    sigmoid_test.print();

    // fixed weights test

    Linear lin = Linear(3, 12);
    Linear lin2 = Linear(12, 1);
    Tensor in = Tensor(3, 3);
    in.fill(2);
    lin2.fix_weights();
    lin.fix_weights();
    Tensor out1 = lin.forward(in);
    out1 = sigmoid.forward(out1);
    Tensor out2 = lin2.forward(out1);
    out2 = sigmoid.forward(out2);
    std::cout << "FIXED WEIGHTS TEST" << '\n';
    out2.print();

    // simulated actuals to test mse_loss forward
    Tensor actuals = Tensor(3, 1);
    actuals.uniform_initialisation(1.0f);
    float loss = mse_forward(out2, actuals);
    std::cout << "LOSS: " << loss << '\n';

    // backward
    Tensor grads = mse_backward(out2, actuals);
    std::cout << "GRADS:" << '\n';
    grads.print();

    return 0;
}
