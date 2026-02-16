#include <cuda_runtime.h>
#include <iostream>
#include "minitorch/activations.cuh"
#include "minitorch/layers.cuh"
#include "minitorch/matrix.cuh"
#include "minitorch/ops.cuh"

using namespace minitorch;
int main() {
    Matrix input = Matrix(12, 6);
    input.rand_fill(-10, 10);

    Linear lin_layer = Linear(input.getcols(), 3);

    Matrix output = lin_layer.forward(input, "relu");

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
    relu(relu_test);
    relu_test.print();

    Matrix sigmoid_test = Matrix(5, 5);
    sigmoid_test.rand_fill(-10, -1); // should fit to 0,1
    sigmoid_test.print();
    sigmoid(sigmoid_test);
    sigmoid_test.print();

    // fixed weights test

    Linear lin = Linear(3000, 12000);
    Linear lin2 = Linear(12000, 1);
    Matrix in = Matrix(300, 3000);
    in.fill(2);
    lin2.fix_weights();
    lin.fix_weights();
    Matrix out1 = lin.forward(in, "sigmoid");
    Matrix out2 = lin2.forward(out1, "sigmoid");
    std::cout << "FIXED WEIGHTS TEST" << '\n';
    out2.print();

    return 0;
}
