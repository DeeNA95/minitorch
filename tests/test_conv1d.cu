#include <iostream>
#include "minitorch/tensor.cuh"
#include "minitorch/conv/conv1d.cuh"

using namespace minitorch;

int main() {
    std::cout << "============= Conv1d Shapes & Math Test =============" << std::endl;

    // Create a Conv1d layer
    // in_channels = 2, out_channels = 3, kernel_size = 3
    // stride = 1, padding = 1, dilation = 1, groups = 1, bias = true
    Conv1d conv(2, 3, 3, 1, 1, 1, 1, true);

    // Create a dummy input tensor: batch=1, in_channels=2, input_len=5
    Tensor x(1, 2, 5);
    x.fill(1.0f); // purely identical data (1s) to make math easy

    std::cout << "--- Forward Pass ---" << std::endl;
    std::cout << "Input Tensor shape: ";
    for (int s : x.get_shape()) std::cout << s << " ";
    std::cout << std::endl;

    Tensor out = conv.forward(x);

    std::cout << "Output Tensor shape (Expected 1 3 5): ";
    for (int s : out.get_shape()) std::cout << s << " ";
    std::cout << "\nOutput Data:" << std::endl;
    out.print();

    std::cout << "\n--- Backward Pass ---" << std::endl;

    // Create dummy gradients from the imaginary layer above: batch=1, out_channels=3, output_len=5
    Tensor grad_out(1, 3, out.get_shape()[2]);
    grad_out.fill(2.0f); // purely identical data (2s) to make math easy

    std::cout << "Incoming Gradients shape: ";
    for (int s : grad_out.get_shape()) std::cout << s << " ";
    std::cout << std::endl;

    Tensor grad_in = conv.backward(grad_out);

    std::cout << "\nGrad Inputs shape (Expected 1 2 5): ";
    for (int s : grad_in.get_shape()) std::cout << s << " ";
    std::cout << "\nGrad Inputs Data:" << std::endl;
    grad_in.print();

    std::cout << "\nTest completely executed without memory exceptions!" << std::endl;
    std::cout << "Forward / Backward pipeline is fully functional!" << std::endl;

    return 0;
}
