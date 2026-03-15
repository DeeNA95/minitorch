#include <cuda_runtime.h>
#include <iostream>
#include "minitorch/matrix.cuh"

using namespace minitorch;

int main() {
    std::cout << "STARTING MATRIX TEST" << '\n';
    Matrix mat = Matrix(20, 14);
    mat.fill(3.14);
    mat.print();
    std::cout << "END OF MATRIX TEST" << '\n';

    std::cout << "UNIFORM XAVIER HE" << '\n';
    Matrix mat2 = Matrix(20, 14);
    mat2.uniform_initialisation(0.01f);
    mat2.print();
    return 0;
}
