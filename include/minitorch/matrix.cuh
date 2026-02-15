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
};
} // namespace minitorch
