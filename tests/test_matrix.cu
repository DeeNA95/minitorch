#include <cuda_runtime.h>
#include <iostream>
#include "minitorch/tensor.cuh"
#include "minitorch/ops.cuh"

using namespace minitorch;

int main() {
    std::cout << "STARTING MATRIX TEST" << '\n';
    Tensor mat = Tensor(20, 14);
    mat.fill(3.14);
    mat.print();
    std::cout << "END OF MATRIX TEST" << '\n';

    std::cout << "UNIFORM XAVIER HE" << '\n';
    Tensor mat2 = Tensor(14, 14);
    mat2.uniform_initialisation(0.01f);
    mat2.print();

    std::cout << "TESTING OPERATORS" << '\n';

    mat = mat + mat;
    mat.print();

    mat = mat * mat2;
    mat.print();

    mat = mat * 3.14f;
    mat.print();

    mat = mat - mat * mat2;
    mat.print();

    mat = elementwise_multiplication(mat, mat);
    mat.print();
    return 0;
}
