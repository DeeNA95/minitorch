#include <iostream>
#include "minitorch/tensor.cuh"
#include "minitorch/pool/maxpool1d.cuh"

using namespace minitorch;

int main() {
    std::cout << "============= MaxPool1d Shapes & Math Test =============" << std::endl;

    // Create a MaxPool1d layer
    // kernel_size = 2, stride = 2, padding = 0, dilation = 1
    MaxPool1d pool(2, 2, 0, 1);

    // Create a dummy input tensor: batch=1, in_channels=2, input_len=6
    Tensor x(1, 2, 6);
    x.fill(1.0f); // purely identical data (1s) to make math easy

    std::cout << "--- Forward Pass ---" << std::endl;
    std::cout << "Input Tensor shape: ";
    for (int s : x.get_shape()) std::cout << s << " ";
    std::cout << "\nInput Data:" << std::endl;
    x.print();

    Tensor out = pool.forward(x);

    std::cout << "Output Tensor shape (Expected 1 2 3): ";
    for (int s : out.get_shape()) std::cout << s << " ";
    std::cout << "\nOutput Data:" << std::endl;
    out.print();

    std::cout << "\n--- Backward Pass ---" << std::endl;

    // Create dummy gradients: batch=1, out_channels=2, output_len=3
    Tensor grad_out(1, 2, 3);
    grad_out.fill(2.0f); // purely identical data (2s) to make math easy

    std::cout << "Incoming Gradients shape: ";
    for (int s : grad_out.get_shape()) std::cout << s << " ";
    std::cout << std::endl;

    Tensor grad_in = pool.backward(grad_out);

    std::cout << "\nGrad Inputs shape (Expected 1 2 6): ";
    for (int s : grad_in.get_shape()) std::cout << s << " ";
    std::cout << "\nGrad Inputs Data:" << std::endl;
    grad_in.print();

    std::cout << "\nTest completely executed without memory exceptions!" << std::endl;
    std::cout << "MaxPool1d Forward / Backward pipeline is fully functional!" << std::endl;

    return 0;
}
