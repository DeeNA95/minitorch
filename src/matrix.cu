#include <cstddef>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <random>
#include <cassert>
#include "minitorch/matrix.cuh"
#include "minitorch/memory_pool.cuh"
#include "minitorch/ops.cuh"
#include "minitorch/ops_utils.cuh"
#include "minitorch/random.cuh"
#include "minitorch/utils.cuh"
namespace minitorch {


/* KERNELS */

// fills matrix
__global__ void fill_mat(float *data, float value, int n_cols, int n_rows) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < n_cols && y < n_rows) {
        data[get_idx_2d(y, x, n_cols)] = value;
    } else
        return;
}

// extract_batch
__global__ void ker_extract_batch(const float *source, float *dest, const int *indices,
                                  int start_idx, int batch_size, int num_cols) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= batch_size * num_cols)
        return;

    int dest_row = i / num_cols, dest_col = i % num_cols;

    auto orig_row = indices[start_idx + dest_row];

    dest[i] = source[orig_row * num_cols + dest_col];
}
// uniform init
__global__ void uniform_init(float *data, int cols, int rows, float scale) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < cols && y < rows) {
        data[get_idx_2d(y, x, cols)] =
            data[get_idx_2d(y, x, cols)] * scale * sqrt(6.0f / (cols + rows));
    }
}

__global__ void scale_fill(float *data, float low, float high, int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;

    if (x < n) {
        data[x] = (high - low) * data[x] + low;
    }
}

// addition
void __global__ matrix_add(const float *__restrict__ a, const float *__restrict__ b, float *c,
                           int n_cols, int n_rows) {
    dev_add(a, b, c, n_cols, n_rows);
}
// subtraction
void __global__ matrix_sub(const float *__restrict__ a, const float *__restrict__ b, float *c,
                           int n_cols, int n_rows) {
    dev_sub(a, b, c, n_cols, n_rows);
}

void __global__ matrix_scalar_multiplication(const float *__restrict__ a, float b, float *c,
                                            int n_cols, int n_rows) {
    dev_scalar_mul(a, b, c, n_cols, n_rows);
}

void __global__ matrix_elementwise_multiplication(const float *__restrict__ a, const float *__restrict__ b, float *c,
                                            int n_cols, int n_rows) {
    dev_elem_mul(a, b, c, n_cols, n_rows);
}

void __global__ matrix_multiplication(const float *__restrict__ a, const float *__restrict__ b, float *c,
                                            int a_rows, int b_rows, int b_cols) {
    dev_dot_product(a, b, c, a_rows, b_rows, b_cols);
}

void __global__ bias_add_ker(const float *__restrict__ a, const float *__restrict__ b, float *c,
                         int n_cols, int n_rows) {
    dev_bias_add(a, b, c, n_cols, n_rows);
}

void __global__ matrix_transpose_ker(const float *A, float *C, int n_cols, int n_rows) {
    dev_transpose(A, C, n_cols, n_rows);
}
/* CLASS MEMBER FUNCTIONS */

// constructor
Matrix::Matrix(int row, int col) : rows(row), cols(col) {
    std::size_t bytes_size = sizeof(float) * row * col;
    // cudaMalloc(&data, bytes_size);
    data = MemoryPool::instance().allocate(bytes_size);
}

Matrix::~Matrix() {
    // cudaFree(data);
    if (data)
        MemoryPool::instance().deallocate(data, sizeof(float) * rows * cols);
}

int Matrix::getrows() const {
    return rows;
}

int Matrix::getcols() const {
    return cols;
}
float *Matrix::getdata() const {
    return data;
}
void Matrix::to_device(float *host_data) {
    std::size_t bytes_size = sizeof(float) * rows * cols;
    cudaMemcpy(Matrix::data, host_data, bytes_size, cudaMemcpyHostToDevice);
}

void Matrix::to_host(float *host_buffer) {
    std::size_t bytes_size = sizeof(float) * rows * cols;
    cudaMemcpy(host_buffer, data, bytes_size, cudaMemcpyDeviceToHost);
}

void Matrix::fill(float value) {
    int w = Matrix::getcols();
    int h = Matrix::getrows();
    // int n = w * h;
    // integer ceil calculation for threads and blocks
    dim3 threads(16, 16, 1);
    dim3 blocks((w + threads.x - 1) / threads.x, (h + threads.y - 1) / threads.y);
    // std::cout << blocks.x << ' ' << blocks.y << '\n';
    fill_mat<<<blocks, threads>>>(Matrix::data, value, w, h);

    // std::cout << data << '\n';
}

void Matrix::print() {
    auto d = new float[Matrix::getcols() * Matrix::getrows()];
    Matrix::to_host(d);
    for (int k = 0; k < Matrix::getrows(); k++) {
        for (int i = 0; i < Matrix::getcols(); i++) {
            std::cout << d[i + k * Matrix::getcols()] << ' ';
        }
        std::cout << '\n';
    }
    delete[] d;
}

Matrix::Matrix(Matrix &&other) noexcept {
    data = other.data;
    rows = other.rows;
    cols = other.cols;

    other.data = nullptr;
    other.rows = 0;
    other.cols = 0;
}

Matrix &Matrix::operator=(Matrix &&other) noexcept {
    if (this != &other) {
        // cudaFree(data);

        if (data)
            MemoryPool::instance().deallocate(data, sizeof(float) * rows * cols);
        data = other.data;
        rows = other.rows;
        cols = other.cols;

        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }
    return *this;
}

// copy
Matrix Matrix::copy() const {
    int rows = this->rows, cols = this->cols;

    Matrix out(rows, cols);
    // this refers to the current matrix as it is a memeber function
    cudaMemcpy(out.data, this->data, (std::size_t)(sizeof(float) * rows * cols),
               cudaMemcpyDeviceToDevice);

    return out;
}

// random fill
void Matrix::rand_fill(float low, float high) {
    int total_elements = rows * cols;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    Random_Manager::instance().uniform(this->data, total_elements);
    scale_fill<<<blocks, threads>>>(this->data, low, high, total_elements);
}

Matrix Matrix::extract_batch(int *d_indices, int start_idx, int batch_size) const {
    Matrix mini_batch(batch_size, this->cols);
    int tot_elems = batch_size * this->cols;
    int threads = 256;
    int blocks = (tot_elems + 255) / 256;

    ker_extract_batch<<<blocks, threads>>>(this->data, mini_batch.getdata(), d_indices, start_idx,
                                           batch_size, this->cols);

    return mini_batch;
}

// xavier_he

void Matrix::uniform_initialisation(float scale) {
    float *data = this->data;
    int rows = this->rows;
    int cols = this->cols;

    dim3 threads(16, 16, 1);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);

    Random_Manager::instance().uniform(data, rows * cols);
    uniform_init<<<blocks, threads>>>(data, cols, rows, scale);
}

// overload operators

Matrix add(const Matrix &A, const Matrix &B) {
    int rows = A.getrows(), cols = A.getcols();
    Matrix out(rows, cols);
    dim3 threads(16, 16, 1);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);
    matrix_add<<<blocks, threads>>>(A.getdata(), B.getdata(), out.getdata(), cols, rows);
    return out;
}
Matrix Matrix::operator+(const Matrix &other) const {
    return add(*this, other);
}

Matrix mat_matmul(const Matrix &A, const Matrix &B) {
    int a_rows = A.getrows();
    int a_cols = A.getcols();
    int b_rows = B.getrows();
    int b_cols = B.getcols();

    assert(a_cols == b_rows &&
           "Number of columns in matrix A must be equal to number of rows in Matrix B");

    Matrix out(a_rows, b_cols);
    dim3 threads(TILE_SIZE, TILE_SIZE, 1);
    dim3 blocks((b_cols + threads.x - 1) / threads.x, (a_rows + threads.y - 1) / threads.y);
    matrix_multiplication<<<blocks, threads>>>(A.getdata(), B.getdata(), out.getdata(), a_rows, b_rows, b_cols);
    return out;
}

Matrix Matrix::operator*(const Matrix &other) const {
    return mat_matmul(*this, other);
}

Matrix Matrix::operator*(float scalar) const {
    int rows = this->rows, cols = this->cols;
    Matrix out(rows, cols);
    dim3 threads(16, 16, 1);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);
    matrix_scalar_multiplication<<<blocks, threads>>>(this->data, scalar, out.getdata(), cols, rows);
    return out;
}

Matrix sub(const Matrix &A, const Matrix &B) {
    int rows = A.getrows(), cols = A.getcols();
    Matrix out(rows, cols);
    dim3 threads(16, 16, 1);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);
    matrix_sub<<<blocks, threads>>>(A.getdata(), B.getdata(), out.getdata(), cols, rows);
    return out;
}

Matrix Matrix::operator-(const Matrix &other) const {
    return sub(*this, other);
}

Matrix Matrix::elem_mul(const Matrix &other) const {
    int rows = this->rows, cols = this->cols;
    Matrix out(rows, cols);
    dim3 threads(16, 16, 1);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);
    matrix_elementwise_multiplication<<<blocks, threads>>>(this->data, other.getdata(), out.getdata(), cols, rows);
    return out;
}

// Implement mat_* operations from ops.cuh
void mat_add(const Matrix &A, const Matrix &B, Matrix &C) {
    C = add(A, B);
}

void mat_sub(const Matrix &A, const Matrix &B, Matrix &C) {
    C = sub(A, B);
}

void mat_elem_mul(const Matrix &A, const Matrix &B, Matrix &C) {
    C = A.elem_mul(B);
}

void mat_scalar_mul(const Matrix &A, float B, Matrix &C) {
    C = A * B;
}

Matrix mat_transpose(Matrix &A) {
    int r = A.getrows();
    int c = A.getcols();
    Matrix C = Matrix(c, r);
    dim3 threads(16, 16, 1);
    dim3 blocks((c + threads.x - 1) / threads.x, (r + threads.y - 1) / threads.y, 1);
    matrix_transpose_ker<<<blocks, threads>>>(A.getdata(), C.getdata(), c, r);
    return C;
}

void b_add(const Matrix &A, const Matrix &B, Matrix &C) {
    const float *a_data = A.getdata();
    const float *b_data = B.getdata();
    float *c_data = C.getdata();
    int n_cols = B.getcols();
    int n_rows = A.getrows();
    dim3 threads(16, 16, 1);
    dim3 blocks((n_cols + threads.x - 1) / threads.x, (n_rows + threads.y - 1) / threads.y, 1);
    bias_add_ker<<<blocks, threads>>>(a_data, b_data, c_data, n_cols, n_rows);
}

} // namespace minitorch
