#include <cassert>
#include <cuda_runtime.h>
#include "minitorch/matrix.cuh"
#include "minitorch/ops.cuh"
#include "minitorch/utils.cuh"
#define TILE_SIZE 16

using namespace minitorch;

namespace minitorch {

// addition
void __global__ matrix_add(const float *__restrict__ a, const float *__restrict__ b, float *c,
                           int n_cols, int n_rows) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= n_cols || y >= n_rows)
        return;

    c[get_idx_2d(y, x, n_cols)] = a[get_idx_2d(y, x, n_cols)] + b[get_idx_2d(y, x, n_cols)];
}

void mat_add(const Matrix &A, const Matrix &B, Matrix &C) {
    const float *a_data = A.getdata();
    const float *b_data = B.getdata();
    float *c_data = C.getdata();

    assert(A.getcols() == B.getcols() && "Columns must be equal");
    assert(A.getrows() == B.getrows() && "Rows must be equal");

    int n_cols = A.getcols(); // ncols is also the witdth of the mat
    int n_rows = A.getrows();

    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);

    matrix_add<<<blocks, threads>>>(a_data, b_data, c_data, n_cols, n_rows);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { /* kernel launch failed */
    }
}

// practically an addition of a column vector to a matrix
void __global__ bias_add(const float *__restrict__ a, const float *__restrict__ b, float *c,
                         int n_cols, int n_rows) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= n_cols || y >= n_rows)
        return;

    c[get_idx_2d(y, x, n_cols)] = a[get_idx_2d(y, x, n_cols)] + b[x];
}

void b_add(const Matrix &A, const Matrix &B, Matrix &C) {
    const float *a_data = A.getdata();
    const float *b_data = B.getdata();
    float *c_data = C.getdata();

    assert(A.getcols() == B.getcols() && "Columns must be equal");

    int n_cols = B.getcols(); // ncols is also the witdth of the mat
    int n_rows = A.getrows();

    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);

    bias_add<<<blocks, threads>>>(a_data, b_data, c_data, n_cols, n_rows);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { /* kernel launch failed */
    }
}

// sub
void __global__ matrix_sub(const float *__restrict__ a, const float *__restrict__ b, float *c,
                           int n_cols, int n_rows) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= n_cols || y >= n_rows)
        return;
    c[get_idx_2d(y, x, n_cols)] = a[get_idx_2d(y, x, n_cols)] - b[get_idx_2d(y, x, n_cols)];
}
// TODO: add overload to alow for matrix sub without specifying the matrix ie the function will
// return a mat
void mat_sub(const Matrix &A, const Matrix &B, Matrix &C) {
    const float *a_data = A.getdata();
    const float *b_data = B.getdata();
    float *c_data = C.getdata();

    assert(A.getcols() == B.getcols() && "Columns must be equal");
    assert(A.getrows() == B.getrows() && "Rows must be equal");

    int n_cols = A.getcols(); // ncols is also the witdth of the mat
    int n_rows = A.getrows();

    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);

    matrix_sub<<<blocks, threads>>>(a_data, b_data, c_data, n_cols, n_rows);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { /* kernel launch failed */
    }
}

// element wise multiplication
void __global__ matrix_elem_mul(const float *__restrict__ a, const float *__restrict__ b, float *c,
                                int n_cols, int n_rows) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= n_cols || y >= n_rows)
        return;
    c[get_idx_2d(y, x, n_cols)] = a[get_idx_2d(y, x, n_cols)] * b[get_idx_2d(y, x, n_cols)];
}

void mat_elem_mul(const Matrix &A, const Matrix &B, Matrix &C) {
    const float *a_data = A.getdata();
    const float *b_data = B.getdata();
    float *c_data = C.getdata();

    assert(A.getcols() == B.getcols() && "Columns must be equal");
    assert(A.getrows() == B.getrows() && "Rows must be equal");

    int n_cols = A.getcols(); // ncols is also the witdth of the mat
    int n_rows = A.getrows();

    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);

    matrix_elem_mul<<<blocks, threads>>>(a_data, b_data, c_data, n_cols, n_rows);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { /* kernel launch failed */
    }
}

// scalar mul
void __global__ matrix_scalar_mul(const float *__restrict__ a, const float b, float *c, int n_cols,
                                  int n_rows) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= n_cols || y >= n_rows)
        return;
    c[get_idx_2d(y, x, n_cols)] = a[get_idx_2d(y, x, n_cols)] * b;
}

void mat_scalar_mul(const Matrix &A, float B, Matrix &C) {
    const float *a_data = A.getdata();
    float *c_data = C.getdata();

    int n_cols = A.getcols(); // ncols is also the witdth of the mat
    int n_rows = A.getrows();

    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);

    matrix_scalar_mul<<<blocks, threads>>>(a_data, B, c_data, n_cols, n_rows);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { /* kernel launch failed */
    }
}

void __global__ matrix_transpose(const float *A, float *C, int n_cols, int n_rows) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x >= n_cols || y >= n_rows)
        return;

    C[x * n_rows + y] = A[y * n_cols + x];
}

Matrix mat_transpose(Matrix &A) {
    int r = A.getrows();
    int c = A.getcols();

    Matrix C = Matrix(c, r);

    dim3 threads(16, 16, 1);
    dim3 blocks((c + threads.x - 1) / threads.x, (r + threads.y - 1) / threads.y, 1);

    matrix_transpose<<<blocks, threads>>>(A.getdata(), C.getdata(), c, r);

    return C;
}

void __global__ matrix_matmul(const float *__restrict__ A, const float *__restrict__ B, float *C,
                              int a_rows, int b_rows, int b_cols) {

    __shared__ float tile_a[TILE_SIZE][TILE_SIZE], tile_b[TILE_SIZE][TILE_SIZE];

    int x = threadIdx.x + blockDim.x * blockIdx.x;    // global col index
    int y = threadIdx.y + blockDim.y * blockIdx.y;    // row index
    int x_local = threadIdx.x, y_local = threadIdx.y; // local index

    float sum = 0.0f;

    // A has row number A_rows and col number b_rows hence a stride is b_rows
    // B has row number b_rows and col number b_cols hence b stride is b_cols
    //
    // fill the shared mem tiles
    for (int i = 0; i < (b_rows + TILE_SIZE - 1) / TILE_SIZE; i++) {
        if (i * TILE_SIZE + x_local >= b_rows) {
            tile_a[y_local][x_local] = 0.0f;
            tile_b[y_local][x_local] = 0.0f;
        } else {
            tile_a[y_local][x_local] = A[y * b_rows + (i * TILE_SIZE + x_local)];
            tile_b[y_local][x_local] = B[(i * TILE_SIZE + y_local) * b_cols + x];
        }
        __syncthreads();
        // perform the dot product
        for (int j = 0; j < TILE_SIZE; j++) {
            sum += tile_a[y_local][j] * tile_b[j][x_local];
        }
        __syncthreads();
    }
    if (x >= b_cols || y >= a_rows) {
        return;
    }
    C[y * b_cols + x] = sum;

    /*
     *NAIVE AND UNOPTIMISED.
     * for (int i = 0; i < b_rows; i++) { // total stride is b_rows as that is shared dim
        sum += A[y * b_rows + i] * B[i * b_cols + x];
    };
    C[get_idx_2d(y, x, b_cols)] = sum;
    */
}

Matrix mat_matmul(const Matrix &A, const Matrix &B) {
    int a_rows = A.getrows();
    int a_cols = A.getcols();
    int b_rows = B.getrows();
    int b_cols = B.getcols();

    assert(a_cols == b_rows &&
           "Number of columns in matrix A must be equal to number of rows in Matrix B");

    Matrix C = Matrix(a_rows, b_cols);
    dim3 threads(TILE_SIZE, TILE_SIZE, 1);
    dim3 blocks((b_cols + threads.x - 1) / threads.x, (a_rows + threads.y - 1) / threads.y, 1);

    matrix_matmul<<<blocks, threads>>>(A.getdata(), B.getdata(), C.getdata(), a_rows, b_rows,
                                       b_cols);

    return C;
}

} // namespace minitorch
