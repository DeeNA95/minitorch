#include <cassert>
#include <cooperative_groups.h>
#include <vector>
#include "minitorch/initialisation/init.cuh"
#include "minitorch/memory_pool.cuh"
#include "minitorch/ops.cuh"
#include "minitorch/ops_utils.cuh"
#include "minitorch/tensor.cuh"
#include "minitorch/utils.cuh"

namespace cg = cooperative_groups;
namespace minitorch {

Tensor::~Tensor() {
    if (data) {
        MemoryPool::instance().deallocate(data, sizeof(float) * this->total_elems);
    }
}

std::vector<int> Tensor::get_shape() const {
    return this->shape;
}

float *Tensor::getdata() const {
    return this->data;
}

int Tensor::indexer() {
    // Basic implementation for now, can be expanded for N-dim indexing
    return total_elems;
}

// Move constructor
Tensor::Tensor(Tensor &&other) noexcept
    : data(other.data), shape(std::move(other.shape)), strides(std::move(other.strides)),
      total_elems(other.total_elems) {
    other.data = nullptr;
    other.total_elems = 0;
}

// Move assignment
Tensor Tensor::operator=(Tensor &&other) noexcept {
    if (this != &other) {
        if (data) {
            MemoryPool::instance().deallocate(data, sizeof(float) * total_elems);
        }
        data = other.data;
        shape = std::move(other.shape);
        strides = std::move(other.strides);
        total_elems = other.total_elems;

        other.data = nullptr;
        other.total_elems = 0;
    }
    return std::move(*this);
}

// overloads and operations

// addition
void __global__ tensor_add(const float *__restrict__ a, const float *__restrict__ b, float *c,
                           int batch_count, int a_rows, int b_rows, int b_cols) {
    int batch_idx = blockIdx.z;

    // calc offsets
    const float *A_batch = a + batch_idx * a_rows * b_cols; // a_cols == b_cols
    const float *B_batch = b + batch_idx * b_rows * b_cols;
    float *C_batch = c + batch_idx * a_rows * b_cols;

    dev_add(A_batch, B_batch, C_batch, b_cols, a_rows);
}

Tensor add(const Tensor &A, const Tensor &B) {
    std::vector<int> a_shape = A.get_shape();
    std::vector<int> b_shape = B.get_shape();

    assert(a_shape == b_shape && "Tensors should be equal size");

    int a_rows = a_shape[a_shape.size() - 2];
    int b_rows = b_shape[b_shape.size() - 2];
    int b_cols = b_shape.back();

    Tensor C = Tensor(a_shape);

    int batch_count = 1;
    for (int i = 0; i < a_shape.size() - 2; i++) {
        batch_count *= a_shape[i];
    }

    dim3 threads(TILE_SIZE, TILE_SIZE, 1);
    dim3 blocks((b_cols + threads.x - 1) / threads.x, (a_rows + threads.y - 1) / threads.y,
                batch_count);
    tensor_add<<<blocks, threads>>>(A.getdata(), B.getdata(), C.getdata(), batch_count, a_rows,
                                    b_rows, b_cols);
    return C;
}

Tensor Tensor::operator+(const Tensor &other) const {
    return add(*this, other);
}

// bias add — bias shape is (channels,), applied per-channel across all spatial dims
// For input (B, C, H, W) or (B, C, L): bias[c] is added to every element in channel c
void __global__ tensor_bias_add_kernel(const float *__restrict__ a, const float *__restrict__ bias,
                                       float *c, int n_channels, int spatial_size) {
    // blockIdx.z indexes into (batch * channel) space
    int bc_idx = blockIdx.z;
    int channel_idx = bc_idx % n_channels;

    int x = threadIdx.x + blockDim.x * blockIdx.x;

    if (x >= spatial_size)
        return;

    int offset = bc_idx * spatial_size;
    c[offset + x] = a[offset + x] + bias[channel_idx];
}

Tensor bias_add(const Tensor &A, const Tensor &bias) {
    std::vector<int> a_shape = A.get_shape();
    std::vector<int> bias_shape = bias.get_shape();

    // bias must be 1D with size == number of channels (second dim)
    int n_channels = a_shape[1];
    assert(bias_shape.size() == 1 && bias_shape[0] == n_channels && "Bias shape must be (channels,)");

    Tensor C = Tensor(a_shape);

    // spatial_size = product of all dims after channel dim
    int spatial_size = 1;
    for (size_t i = 2; i < a_shape.size(); i++) {
        spatial_size *= a_shape[i];
    }

    // batch_channels = batch_size * n_channels
    int batch_channels = 1;
    for (size_t i = 0; i < 2; i++) {
        batch_channels *= a_shape[i];
    }

    int threads = 256;
    int blocks_x = (spatial_size + threads - 1) / threads;
    dim3 grid(blocks_x, 1, batch_channels);

    tensor_bias_add_kernel<<<grid, threads>>>(A.getdata(), bias.getdata(), C.getdata(),
                                               n_channels, spatial_size);
    return C;
}

// subtraction
void __global__ tensor_sub(const float *__restrict__ a, const float *__restrict__ b, float *c,
                           int batch_count, int a_rows, int b_rows, int b_cols) {
    int batch_idx = blockIdx.z;

    // calc offsets
    const float *A_batch = a + batch_idx * a_rows * b_cols;
    const float *B_batch = b + batch_idx * b_rows * b_cols;
    float *C_batch = c + batch_idx * a_rows * b_cols;

    dev_sub(A_batch, B_batch, C_batch, b_cols, a_rows);
}

Tensor sub(const Tensor &A, const Tensor &B) {
    std::vector<int> a_shape = A.get_shape();
    std::vector<int> b_shape = B.get_shape();

    assert(a_shape == b_shape && "Tensors should be equal size");

    int a_rows = a_shape[a_shape.size() - 2];
    int b_rows = b_shape[b_shape.size() - 2];
    int b_cols = b_shape.back();

    Tensor C = Tensor(a_shape);

    int batch_count = 1;
    for (int i = 0; i < a_shape.size() - 2; i++) {
        batch_count *= a_shape[i];
    }

    dim3 threads(TILE_SIZE, TILE_SIZE, 1);
    dim3 blocks((b_cols + threads.x - 1) / threads.x, (a_rows + threads.y - 1) / threads.y,
                batch_count);
    tensor_sub<<<blocks, threads>>>(A.getdata(), B.getdata(), C.getdata(), batch_count, a_rows,
                                    b_rows, b_cols);
    return C;
}

Tensor Tensor::operator-(const Tensor &other) const {
    return sub(*this, other);
}

// elementwise multiplication
void __global__ tensor_elementwise_multiplication(const float *__restrict__ a,
                                                  const float *__restrict__ b, float *c,
                                                  int batch_count, int a_rows, int b_rows,
                                                  int b_cols) {
    int batch_idx = blockIdx.z;

    // calc offsets
    const float *A_batch = a + batch_idx * a_rows * b_cols;
    const float *B_batch = b + batch_idx * b_rows * b_cols;
    float *C_batch = c + batch_idx * a_rows * b_cols;

    dev_elem_mul(A_batch, B_batch, C_batch, b_cols, a_rows);
}

Tensor elementwise_multiplication(const Tensor &A, const Tensor &B) {
    std::vector<int> a_shape = A.get_shape();
    std::vector<int> b_shape = B.get_shape();

    assert(a_shape == b_shape && "Tensors should be equal size");

    int a_rows = a_shape[a_shape.size() - 2];
    int b_rows = b_shape[b_shape.size() - 2];
    int b_cols = b_shape.back();

    Tensor C = Tensor(a_shape);

    int batch_count = 1;
    for (int i = 0; i < a_shape.size() - 2; i++) {
        batch_count *= a_shape[i];
    }

    dim3 threads(TILE_SIZE, TILE_SIZE, 1);
    dim3 blocks((b_cols + threads.x - 1) / threads.x, (a_rows + threads.y - 1) / threads.y,
                batch_count);
    tensor_elementwise_multiplication<<<blocks, threads>>>(A.getdata(), B.getdata(), C.getdata(),
                                                           batch_count, a_rows, b_rows, b_cols);
    return C;
}

// scalar multipliaction
void __global__ tensor_scalar_multiplication(const float *__restrict__ a, float b, float *c,
                                             int batch_count, int a_rows, int b_rows, int b_cols) {
    int batch_idx = blockIdx.z;

    // calc offsets
    const float *A_batch = a + batch_idx * a_rows * b_cols;
    float *C_batch = c + batch_idx * a_rows * b_cols;

    dev_scalar_mul(A_batch, b, C_batch, b_cols, a_rows);
}

Tensor scalar_multiplication(const Tensor &A, float b) {
    std::vector<int> a_shape = A.get_shape();

    int a_rows = a_shape[a_shape.size() - 2];
    int b_rows = a_rows;
    int b_cols = a_shape.back();

    Tensor C = Tensor(a_shape);

    int batch_count = 1;
    for (int i = 0; i < a_shape.size() - 2; i++) {
        batch_count *= a_shape[i];
    }

    dim3 threads(TILE_SIZE, TILE_SIZE, 1);
    dim3 blocks((b_cols + threads.x - 1) / threads.x, (a_rows + threads.y - 1) / threads.y,
                batch_count);
    tensor_scalar_multiplication<<<blocks, threads>>>(A.getdata(), b, C.getdata(), batch_count,
                                                      a_rows, b_rows, b_cols);
    return C;
}

Tensor Tensor::operator*(float scalar) const {
    return scalar_multiplication(*this, scalar);
}

// matmul
void __global__ tensor_matmul_kernel(const float *__restrict__ A, const float *__restrict__ B,
                                     float *C, int batch_count, int a_rows, int b_rows,
                                     int b_cols) {
    int batch_idx = blockIdx.z;

    // calc offsets
    const float *A_batch = A + batch_idx * a_rows * b_rows; // a_cols == b_rows
    const float *B_batch = B + batch_idx * b_rows * b_cols;
    float *C_batch = C + batch_idx * a_rows * b_cols;

    dev_dot_product(A_batch, B_batch, C_batch, a_rows, b_rows, b_cols);
}

Tensor tensor_matmul(const Tensor &A, const Tensor &B) {
    // get shape
    std::vector<int> a_shape = A.get_shape();
    std::vector<int> b_shape = B.get_shape();

    int a_rows = a_shape[a_shape.size() - 2];
    int a_cols = a_shape.back();
    int b_rows = b_shape[b_shape.size() - 2];
    int b_cols = b_shape.back();

    assert(a_cols == b_rows &&
           "Number of columns in matrix A must be equal to number of rows in Matrix B");

    a_shape.back() = b_cols;

    Tensor C = Tensor(a_shape);

    int batch_count = 1;
    for (int i = 0; i < a_shape.size() - 2; i++) {
        batch_count *= a_shape[i];
    }

    dim3 threads(TILE_SIZE, TILE_SIZE, 1);
    dim3 blocks((b_cols + threads.x - 1) / threads.x, (a_rows + threads.y - 1) / threads.y,
                batch_count);

    tensor_matmul_kernel<<<blocks, threads>>>(A.getdata(), B.getdata(), C.getdata(), batch_count,
                                              a_rows, b_rows, b_cols);

    return C;
}

Tensor Tensor::operator*(const Tensor &other) const {
    return tensor_matmul(*this, other);
}

void Tensor::reshape(std::vector<int> shape) {
    this->shape = shape;
    this->strides.resize(this->shape.size());
    int old_total_elems = this->total_elems;
    this->total_elems = 1;
    for (int s : this->shape) {
        this->total_elems *= s;
    }
    if (old_total_elems != this->total_elems) {
        std::cerr << "SHAPE CHANGE NOT ALLOWED" << '\n';
        exit(1);
    }
    // start from second to last dim
    for (int i = this->shape.size() - 2; i >= 0; i--) {
        // stride for current dim is product of previous dims
        this->strides[i] = this->strides[i + 1] * this->shape[i + 1];
    }
    // consider addition of a contiguous that will change shape in memory
}

__global__ void tensor_init(float *data, int n_cols, int n_rows, float scale, int n_matrices) {
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    auto global_warp_index = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();

    if (global_warp_index >= n_matrices)
        return;

    float *data_ptr = data + global_warp_index * n_cols * n_rows;
    uniform_init(data_ptr, scale, warp, n_cols, n_rows);
}

void Tensor::uniform_initialisation(float scale) {
    // calculates blocks needed for warps
    int n_matrices = 1;
    size_t size = this->shape.size();
    for (int i = 0; i < size - 2; i++) {
        n_matrices *= this->shape[i];
    }
    // using 256 threads per block so 8 warps
    int threads = 256;
    int blocks = (n_matrices + threads / 32 - 1) /
                 (threads / 32); // kept programmatic incase of increase in threads
    int n_cols = this->shape[size - 1], n_rows = this->shape[size - 2];
    tensor_init<<<blocks, threads>>>(this->data, n_cols, n_rows, scale, n_matrices);
}
Tensor Tensor::copy() const {
    std::vector<int> shape = this->shape;

    Tensor out(shape);
    // this refers to the current matrix as it is a memeber function
    cudaMemcpy(out.data, this->data, (std::size_t)(sizeof(float) * this->total_elems),
               cudaMemcpyDeviceToDevice);

    return out;
}

__global__ void sum_reduce(const float* __restrict__ in, float* __restrict__ out, int outer, int inner, int reduce_size){
    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    int global_warp_index = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();

    if (global_warp_index >= outer*inner) return;

    int outer_idx = global_warp_index / inner;
    int inner_idx = global_warp_index % inner;

    const float* in_ptr = in + outer_idx * inner * reduce_size + inner_idx;
    float* out_ptr = out + outer_idx * inner + inner_idx;

    float warp_sum = dev_sum(in_ptr, reduce_size, inner, warp);

    if (warp.thread_rank() == 0){
        *out_ptr = warp_sum;
    }
}

Tensor Tensor::sum(int dim, bool keepdim){
    if (dim < 0){
        dim = this->shape.size()+dim;
    }
    int outer = 1;
    int inner =1;
    int reduce_size = this->shape[dim];

    for (int i=0 ; i < dim; i++){
        outer *= this->shape[i];
    }
    for (int i=this->shape.size()-1 ; i > dim; i--){
        inner *= this->shape[i];
    }

    std::vector<int> out_shape;
    if (keepdim){
        out_shape = this->shape;
        out_shape[dim] = 1;
    } else {
        out_shape = this->shape;
        out_shape.erase(out_shape.begin()+dim);
    }


    if (out_shape.empty()) out_shape = {1};

    Tensor out(out_shape);

    int threads = 256;
    int blocks = (outer*inner + threads/32 -1)/(threads/32);

    sum_reduce<<<blocks, threads>>>(this->data, out.getdata(), outer, inner, reduce_size);

    return out;

}

__global__ void tensor_transpose_kernel(const float *__restrict__ in, float *__restrict__ out, int batch_count, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;

    if (x < cols && y < rows) {
        int in_offset = z * rows * cols;
        int out_offset = z * rows * cols;
        out[out_offset + x * rows + y] = in[in_offset + y * cols + x];
    }
}

Tensor tensor_transpose(const Tensor &A) {
    std::vector<int> shape = A.get_shape();
    if (shape.size() < 2) return A.copy();
    
    int rows = shape[shape.size() - 2];
    int cols = shape.back();
    
    std::vector<int> out_shape = shape;
    out_shape[out_shape.size() - 2] = cols;
    out_shape.back() = rows;
    
    Tensor out(out_shape);
    
    int batch_count = 1;
    for (int i = 0; i < shape.size() - 2; i++) {
        batch_count *= shape[i];
    }
    
    dim3 threads(16, 16);
    dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y, batch_count);
    
    tensor_transpose_kernel<<<blocks, threads>>>(A.getdata(), out.getdata(), batch_count, rows, cols);
    return out;
}

void Tensor::to_host(float *buffer) const {
    cudaMemcpy(buffer, this->data, sizeof(float) * this->total_elems, cudaMemcpyDeviceToHost);
}

void Tensor::to_device(const float *buffer) {
    cudaMemcpy(this->data, buffer, sizeof(float) * this->total_elems, cudaMemcpyHostToDevice);
}

} // namespace minitorch
