#pragma once

namespace minitorch {

class Matrix {

private:
    float *data;
    int rows;
    int cols;

public:
    // constructor
    Matrix(int rows, int cols);
    // Destructor
    ~Matrix();

    // move constructor ( ie takes from one matrix and give to this one)
    Matrix(Matrix &&other) noexcept;
    Matrix &operator=(Matrix &&other) noexcept;
    // deleting copy constructor to prevent copy accidents
    Matrix(const Matrix &) = delete;
    Matrix &operator=(const Matrix &) = delete;
    Matrix copy() const;

    // get item
    // float &at(int r, int c);

    // sends to device
    void to_device(float *host_data);

    // to hosts
    void to_host(float *host_buffer);

    void fill(float value);

    void print();

    int getrows() const;
    int getcols() const;
    float *getdata() const;

    void rand_fill(float low, float high);

    // for mini-batches, takes a `list` of indices and a start index and batchsize
    Matrix extract_batch(int *d_indices, int start_idx, int batch_size) const;

    void uniform_initialisation(float scale);

    // overloads
    Matrix operator+(const Matrix &other) const;

    Matrix operator*(const Matrix &other) const;
    Matrix operator*(float scalar) const;
    Matrix operator-(const Matrix &other) const;
    // not an overload but for element-wise multiplication
    Matrix elem_mul(const Matrix &other) const;
};

// Non-member functions for Matrix operations
Matrix add(const Matrix &A, const Matrix &B);
Matrix sub(const Matrix &A, const Matrix &B);
Matrix mat_matmul(const Matrix &A, const Matrix &B);

} // namespace minitorch
