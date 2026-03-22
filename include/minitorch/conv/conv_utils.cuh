#include "cuda_runtime.h"

// __global__ void conv1d_forward(const float* __restrict__ input, const float* __restrict__ weight, const float* __restrict__ bias, float* __restrict__ output, int in_channels, int out_channels, int kernel_size, int stride, int padding, int dilation, int groups, int batch_size, int in_length, int out_length){
//     // for convolutions, we want 1 thread to 1 output
//     // "Which index of the array of the output am I calculating?"
//     int out_x =
//     // "Which filter am I applying?"
//     int out_c = blockIdx.y;  // Because we set blockDim.y = 1 as threads is not dim3
//     // "Which audio clip am I working on?"
//     int b = blockIdx.z;      // Because we set blockDim.z = 1

//     if (out_x >= out_length) return;



// }

__global__ void im2col1d(const float* __restrict__ input, float* __restrict__ col_mat, int kernel_size, int in_channels, int stride, int padding, int dilation, int in_length, int out_length){
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_c = blockIdx.y;
    int b = blockIdx.z;

    int ic = out_c / kernel_size;  // Which channel? (4 / 3 = 1)
    int k  = out_c % kernel_size;  // Which position in the window? (4 % 3 = 1)

    if (out_x >= out_length) return;

    int in_x = out_x * stride + k * dilation - padding;
    int input_flat_idx = b * (in_channels * in_length) + ic * in_length + in_x;

    if (in_x >= 0 && in_x < in_length) {
        col_mat[b * (in_channels * kernel_size * out_length) + out_c * out_length + out_x] = input[input_flat_idx];
    } else {
        col_mat[b * (in_channels * kernel_size * out_length) + out_c * out_length + out_x] = 0.0f;
    }



}
