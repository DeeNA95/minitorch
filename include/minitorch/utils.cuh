#pragma once
#include <algorithm>
#include <array>
#include <cstdlib>
#include <stdexcept>
#include <utility>
#include <vector>
#define MAX_TENSOR_RANK 6
#define MAX_BATCH_RANK (MAX_TENSOR_RANK - 2)
namespace minitorch {

__host__ __inline__ std::pair<std::vector<int>, std::vector<int>>
get_padded_sizes(std::vector<int> shape_a, std::vector<int> shape_b) {
    if (shape_a.size() != shape_b.size()) {
        int diff = static_cast<int>(shape_a.size()) - static_cast<int>(shape_b.size());
        if (diff < 0) {
            shape_a.insert(shape_a.begin(), std::abs(diff), 1);
        } else {
            shape_b.insert(shape_b.begin(), std::abs(diff), 1);
        }
    }
    return {shape_a, shape_b};
}

__host__ __device__ __forceinline__ int get_idx_2d(int y, int x, int n_cols) {
    return (y * n_cols + x);
}

__host__ __device__ __forceinline__ int get_batch_size(std::vector<int> shape) {
    // take the shape vec, drop the last 2 and calc the prod
    int tot = 1;
    int needed_len = shape.size() - 2;

    for (int i = 0; i < needed_len; i++) {
        tot *= shape[i];
    }
    return tot;
}

__host__ __forceinline__ bool check_broadcasting(const std::vector<int> shape_a,
                                                 const std::vector<int> shape_b) {
    // check if each dim either matches or is 1
    for (int i = 0; i < shape_a.size() - 2; i++) {
        if (shape_a[i] != shape_b[i] && (shape_a[i] != 1 && shape_b[i] != 1))
            return false;
    }
    return true;
}
__host__ __forceinline__ std::vector<int> broadcasted_output(std::vector<int> shape_a,
                                                             std::vector<int> shape_b) {
    // check if can be broadcasted then output shape of final tensor
    std::vector<int> out;
    // if (shape_a.size() != shape_b.size()) {
    //     int diff = static_cast<int>(shape_a.size()) - static_cast<int>(shape_b.size());
    //     if (diff < 0) {
    //         shape_a.insert(shape_a.begin(), std::abs(diff), 1);
    //     } else {
    //         shape_b.insert(shape_b.begin(), std::abs(diff), 1);
    //     }
    // }
    auto [aligned_a, aligned_b] = get_padded_sizes(shape_a, shape_b);

    bool broadcasted = check_broadcasting(aligned_a, aligned_b);
    if (broadcasted) {

        for (int i = 0; i < aligned_a.size() - 2; i++) {
            if (aligned_a[i] >= aligned_b[i]) {
                out.push_back(aligned_a[i]);

            } else {
                out.push_back(aligned_b[i]);
            }
        }
    } else {
        return {};
    }
    int rows = aligned_a[aligned_a.size() - 2];
    int cols = aligned_b.back();
    out.push_back(rows);
    out.push_back(cols);
    return out;
} /*
   * THIS IS THE; STRUCT WHICH WILL CONTAIN ALL THE METADATA NEEDED FOR STRIDE CALCULATIONS FOR
   * MATMULS
   */
struct MatmulMeta
{
    int rank;        // then lengthof the tensor shape 2 <= rank <= 6 because of MAX_TENSOR_RANK
    int batch_rank;  // num dimensions leaving out the matrix ones ie batch dimensions number
    int batch_count; // product of the values of batch_rank dims nb at 0 ie a 2d matmul batch count
                     // will be 1 while batchrank 0.

    std::array<int, MAX_TENSOR_RANK> output_shape;
    std::array<int, MAX_BATCH_RANK>
        a_batch_stride; // will contain the needed stride values for broadcast calculatioins
    std::array<int, MAX_BATCH_RANK> b_batch_stride;
    std::array<int, MAX_BATCH_RANK> c_batch_stride;

    int a_rows;
    int b_rows;
    int a_cols;
    int b_cols;
};

__host__ __inline__ MatmulMeta matmul_metadata(std::vector<int> shape_a, std::vector<int> shape_b) {
    if (shape_a.size() > MAX_TENSOR_RANK || shape_b.size() > MAX_TENSOR_RANK ||
        shape_a.size() < 2 || shape_b.size() < 2)
        throw std::invalid_argument("shape incorrect");

    if (shape_b[shape_b.size() - 2] != shape_a.back())
        throw std::invalid_argument("shape incorrect");

    bool all_positive = std::all_of(shape_a.begin(), shape_a.end(), [](int i) { return i > 0; });
    if (!all_positive)
        throw std::invalid_argument("dim size contains a negative");
    bool all_positive2 = std::all_of(shape_b.begin(), shape_b.end(), [](int i) { return i > 0; });
    if (!all_positive2)
        throw std::invalid_argument("dim size contains a negative2");

    std::vector<int> out = broadcasted_output(shape_a, shape_b);
    if (out.size() == 0) {
        throw std::invalid_argument("no output shape");
    }

    auto [aligned_a, aligned_b] = get_padded_sizes(shape_a, shape_b);
    // contains accurate set for the output dims now we need the stride calcs
}

} // namespace minitorch
