#include <cuda_runtime.h>
#include <iostream>
#include "minitorch/tensor.cuh"

using namespace minitorch;

int main() {
    std::cout << "STARTING TENSOR TEST" << '\n';
    Tensor mat(20, 14, 1, 2, 3, 4);
    Tensor mat2(20, 14, 1, 2, 4, 5);

    mat.uniform_initialisation(0.1f);
    mat2.uniform_initialisation(0.1f);
    Tensor mat3 = mat * mat2;

    mat3.reshape({20, 14, 1, 2, 4, 5});
    std::cout << "END OF TENSOR TEST" << '\n';

    return 0;
}
