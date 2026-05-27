#pragma once
#include <math_constants.h>
#include "cuda_runtime.h"

__global__ void maxpool1d_forward(const float *__restrict__ input, float *__restrict__ output,
                                  float *__restrict__ arg_max, int in_channels, int in_length,
                                  int out_length, int kernel_size, int stride, int padding,
                                  int dilation) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_c = blockIdx.y;

    int b = out_c / in_channels;
    int c = out_c % in_channels;

    if (out_x >= out_length)
        return;

    int output_flat_idx = b * (in_channels * out_length) + c * out_length + out_x;
    int in_x_start = out_x * stride - padding;

    float max_val = -CUDART_INF_F;
    int max_idx = -1;

    for (int k = 0; k < kernel_size; k++) {
        int in_x = in_x_start + k * dilation;
        if (in_x >= 0 && in_x < in_length) {
            int input_flat_idx = b * (in_channels * in_length) + c * in_length + in_x;
            float val = input[input_flat_idx];
            if (val > max_val) {
                max_val = val;
                max_idx = input_flat_idx;
            }
        }
    }

    output[output_flat_idx] = max_val;
    arg_max[output_flat_idx] = static_cast<float>(max_idx);
}

__global__ void maxpool1d_backward(const float *__restrict__ grad_tensor,
                                   float *__restrict__ grad_input,
                                   const float *__restrict__ arg_max, int in_channels,
                                   int out_length) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_c = blockIdx.y;

    int b = out_c / in_channels;
    int c = out_c % in_channels;

    if (out_x >= out_length)
        return;

    int output_flat_idx = b * (in_channels * out_length) + c * out_length + out_x;

    int idx = arg_max[output_flat_idx];
    atomicAdd(&grad_input[idx], grad_tensor[output_flat_idx]);
}
