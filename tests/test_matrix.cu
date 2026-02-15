#include <cuda_runtime.h>
#include <iostream>
#include "minitorch/matrix.cuh"

using namespace minitorch;

int main() {
    std::cout << "STARTING MATRIX TEST" << '\n';
    Matrix mat = Matrix(200, 140);
    mat.fill(3.14);
    mat.print();
    std::cout << "END OF MATRIX TEST" << '\n';
    return 0;
}
