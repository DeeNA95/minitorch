#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "minitorch/memory_pool.cuh"

namespace minitorch {

class Tensor {

private:
    float *data;
    std::vector<int> shape;
    std::vector<int> strides;
    int total_elems;

public:
    template <typename... Args> Tensor(Args... args) {
        this->shape = {static_cast<int>(args)...};
        // stride of first dim (starting from the back is 1) ie n*c, stride of n is c and stride of
        // c is
        // 1
        this->strides.resize(this->shape.size());
        this->strides.back() = 1;
        this->total_elems = 1;
        for (int s : this->shape) {
            this->total_elems *= s;
        }
        // start from second to last dim
        for (int i = this->shape.size() - 2; i >= 0; i--) {
            // stride for current dim is product of previous dims
            this->strides[i] = this->strides[i + 1] * this->shape[i + 1];
        }

        data = MemoryPool::instance().allocate(sizeof(float) * this->total_elems);
        // std::cout << "Tensor of size " << this->total_elems << " created" << '\n';
        // std::cout << '\n' << "SHAPES" << '\n' << '\n';
        //
        // for (auto i = this->shape.begin(); i != this->shape.end(); i++) {
        //     std::cout << *i << " ";
        // }
        // std::cout << '\n' << "STRIDES" << '\n' << '\n';
        //
        // for (auto i = this->strides.begin(); i != this->strides.end(); i++) {
        //     std::cout << *i << " ";
        // }
    }

    // second constructor which takes vectors
    Tensor(std::vector<int> shape) {
        this->shape = shape;
        // stride of first dim (starting from the back is 1) ie n*c, stride of n is c and stride of
        // c is
        // 1
        this->strides.resize(this->shape.size());
        this->strides.back() = 1;
        this->total_elems = 1;
        for (int s : this->shape) {
            this->total_elems *= s;
        }
        // start from second to last dim
        for (int i = this->shape.size() - 2; i >= 0; i--) {
            // stride for current dim is product of previous dims
            this->strides[i] = this->strides[i + 1] * this->shape[i + 1];
        }

        data = MemoryPool::instance().allocate(sizeof(float) * this->total_elems);
        // std::cout << "Tensor of size " << this->total_elems << " created" << '\n';
        // std::cout << '\n' << "SHAPES" << '\n' << '\n';
        //
        // for (auto i = this->shape.begin(); i != this->shape.end(); i++) {
        //
    }

    ~Tensor();

    int indexer();

    Tensor(Tensor &&other) noexcept;
    Tensor operator=(Tensor &&other) noexcept;
    // deleting copy constructor to prevent copy accidents
    Tensor(const Tensor &) = delete;
    Tensor operator=(const Tensor &) = delete;

    // overload operators
    Tensor operator+(const Tensor &other) const;
    Tensor operator-(const Tensor &other) const;
    Tensor operator*(const Tensor &other) const;
    Tensor operator*(float scalar) const;

    std::vector<int> get_shape() const;

    float *getdata() const;

    // reshape
    void reshape(std::vector<int> shape);

    // init
    void uniform_initialisation(float scale);

    Tensor sum(int dim, bool keepdim = true);

    Tensor copy() const;

    void to_host(float *buffer) const;
    void to_device(const float *buffer);


};

// Non-member functions for Tensor operations
Tensor add(const Tensor &A, const Tensor &B);
Tensor sub(const Tensor &A, const Tensor &B);
Tensor elementwise_multiplication(const Tensor &A, const Tensor &B);
Tensor scalar_multiplication(const Tensor &A, float b);
Tensor tensor_matmul(const Tensor &A, const Tensor &B);
Tensor tensor_transpose(const Tensor &A);
Tensor bias_add(const Tensor &A, const Tensor &bias);

} // namespace minitorch
