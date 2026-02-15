#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include "minitorch/matrix.cuh"
#include "minitorch/ops.cuh"

using namespace minitorch;

// Assuming matrix A is n x k, matrix B is k x m, and result C is n x m
void multiplyMatrices(int n, int k, int m, const float *A, const float *B, float *C) {
    // Outer loop iterates over the rows of the result matrix C (and matrix A)

    for (int i = 0; i < n; ++i) {
        // Middle loop iterates over the columns of the result matrix C (and matrix B)
        for (int j = 0; j < m; ++j) {
            // Initialize the element C[i][j] to 0 before the inner loop
            C[i * m + j] = 0;
            // Inner loop calculates the dot product of the i-th row of A and j-th column of B
            for (int q = 0; q < k; ++q) { C[i * m + j] += A[i * k + q] * B[q * m + j]; }
        }
    }
}

int main() {
    // simple 4x4 matrix
    Matrix A = Matrix(4, 4);
    Matrix B = Matrix(4, 4);

    Matrix C = Matrix(4, 4);

    A.fill(3.14);
    B.fill(6.86);

    mat_add(A, B, C);
    C.print();
    std::cout << "AFTER ADD" << '\n';
    mat_sub(A, B, C);
    C.print();
    std::cout << "AFTER SUB" << '\n';

    mat_elem_mul(A, B, C);
    C.print();
    std::cout << "AFTER ELEMENT WISE MULTIPLICATION" << '\n';

    mat_scalar_mul(A, 1.7321, C);
    C.print();
    std::cout << "AFTER SCALAR MULTIPLE BY 1.7321" << '\n';

    Matrix D = Matrix(12, 3);
    D.fill(1.21);
    std::cout << "BEFORE TRANSPOSE" << '\n';
    D.print();

    Matrix transposed = mat_transpose(D);
    std::cout << "TRANSPOSED" << '\n';
    transposed.print();

    Matrix X = Matrix(4, 8);
    X.fill(2);

    Matrix Y = mat_matmul(A, X);
    std::cout << "MATMUL" << '\n';
    Y.print();

    std::cout << '\n' << '\n';

    // test of matmul correctness vs cpu
    Matrix matmul_test1(128, 64);
    float *host_buffer1 = new float[128 * 64];
    Matrix matmul_test2(64, 128);
    float *host_buffer2 = new float[64 * 128];

    // copy to device and fill with random
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-5, 5);

    // matmul_test1.to_host(host_buffer1);
    // matmul_test2.to_host(host_buffer2);

    for (int i = 0; i < matmul_test1.getcols() * matmul_test1.getrows(); i++) {
        host_buffer1[i] = dist(gen);
    }
    for (int i = 0; i < matmul_test2.getcols() * matmul_test2.getrows(); i++) {
        host_buffer2[i] = dist(gen);
    }

    // back to deevice mem
    matmul_test1.to_device(host_buffer1);
    matmul_test2.to_device(host_buffer2);

    Matrix matmul_test3 = mat_matmul(matmul_test1, matmul_test2);
    float *host_buffer3 = new float[128 * 128];
    float *host_ans3 = new float[128 * 128];
    matmul_test3.to_host(host_ans3);

    // manual test
    multiplyMatrices(128, 64, 128, host_buffer1, host_buffer2, host_buffer3);

    for (int i = 0; i < 128 * 128; i++) {
        float min_diff = std::abs(host_ans3[i] - host_buffer3[i]);
        if (min_diff >= 1e-4) {
            // #warning "MATMUL INCORRECT";
            std::cout << "FAIL at pos " << i << " | GPU: " << host_ans3[i]
                      << " CPU: " << host_buffer3[i] << " diff: " << min_diff << '\n';
        } else {
            std::cout << "Correct for Pos " << i << '\n';
        }
    }
    delete[] host_buffer1;
    delete[] host_buffer2;
    delete[] host_buffer3;
    delete[] host_ans3;
    return 0;
}
