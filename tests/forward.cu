#include <cuda_runtime.h>
#include <iostream>
#include "minitorch/activations.cuh"
#include "minitorch/layers.cuh"
#include "minitorch/loss.cuh"
#include "minitorch/matrix.cuh"
#include "minitorch/ops.cuh"

using namespace minitorch;
int main() {
    Matrix input = Matrix(12, 6);
    input.rand_fill(-10, 10);

    Linear lin_layer = Linear(input.getcols(), 3);

    Matrix output = lin_layer.forward(input);
    relu_forward(output);

    std::cout << "FORWARD PASS COMPLETE" << '\n';
    std::cout << "INPUT" << '\n';
    input.print();
    std::cout << '\n';
    std::cout << "OUTPUT" << '\n';
    output.print();

    std::cout << '\n' << '\n' << "TESTING ACTIVATIONS" << '\n';
    Matrix relu_test = Matrix(5, 5);
    relu_test.rand_fill(-10, 2); // skewed negative to get more negs
    relu_test.print();
    relu_forward(relu_test);
    relu_test.print();

    Matrix sigmoid_test = Matrix(5, 5);
    sigmoid_test.rand_fill(-10, -1); // should fit to 0,1
    sigmoid_test.print();
    sigmoid_forward(sigmoid_test);
    sigmoid_test.print();

    // fixed weights test

    Linear lin = Linear(3, 12);
    Linear lin2 = Linear(12, 1);
    Matrix in = Matrix(3, 3);
    in.fill(2);
    lin2.fix_weights();
    lin.fix_weights();
    Matrix out1 = lin.forward(in);
    sigmoid_forward(out1);
    Matrix out2 = lin2.forward(out1);
    sigmoid_forward(out2);
    std::cout << "FIXED WEIGHTS TEST" << '\n';
    out2.print();

    // simulated actuals to test mse_loss forward
    Matrix actuals = Matrix(3, 1);
    actuals.rand_fill(-1, 1);
    float loss = mse_forward(out2, actuals);
    std::cout << "LOSS: " << loss << '\n';

    // backward
    Matrix grads = mse_backward(out2, actuals);
    std::cout << "GRADS:" << '\n';
    grads.print();

    return 0;
}
