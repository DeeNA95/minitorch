#pragma once

namespace minitorch {

__host__ __device__ __forceinline__ int get_idx_2d(int y, int x, int n_cols) {
    return (y * n_cols + x);
}

} // namespace minitorch
