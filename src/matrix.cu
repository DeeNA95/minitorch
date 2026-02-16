#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include "minitorch/matrix.cuh"

using namespace minitorch;

__global__ void fill_mat(float *data, float value, int n_cols, int n_rows) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    auto idx = [&n_cols](int y, int x) { return (y * n_cols + x); };
    if (x < n_cols && y < n_rows) {
        data[idx(y, x)] = value;
    } else
        return;
}

Matrix::Matrix(int row, int col) : rows(row), cols(col) {
    std::size_t bytes_size = sizeof(float) * row * col;
    cudaMalloc(&data, bytes_size);
}

Matrix::~Matrix() {
    cudaFree(data);
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
        cudaFree(data);
        data = other.data;
        rows = other.rows;
        cols = other.cols;

        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }
    return *this;
}

// random fill
void Matrix::rand_fill(float low, float high) {
    float *buffer = new float[Matrix::getcols() * Matrix::getrows()];

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(low, high);

    for (int i = 0; i < Matrix::getcols() * Matrix::getrows(); i++) { buffer[i] = dist(gen); }

    Matrix::to_device(buffer);

    delete[] buffer;
}
