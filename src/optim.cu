#include <cstddef>
#include <cstdio>
#include <iostream>
#include "cooperative_groups.h"
#include "cuda_runtime.h"
#include "minitorch/optim.cuh"

namespace cg = cooperative_groups;
using namespace minitorch;

namespace minitorch {

__global__ void sgd(float *__restrict__ weights, const float *__restrict__ grad_weights, float lr,
                    int n) {

    auto grid = cg::this_grid();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    for (int tid = grid.thread_rank(); tid < n; tid += grid.size()) {
        weights[tid] -= lr * grad_weights[tid];
    }
}

void sgd_update(Tensor &weights, const Tensor &grad_weights, float lr) {
    float *w = weights.getdata();
    const float *gw = grad_weights.getdata();
    int n_rows = weights.get_shape()[0], n_cols = weights.get_shape()[1];

    int threads = 256;

    sgd<<<1, threads>>>(w, gw, lr, n_rows * n_cols);
    cudaDeviceSynchronize();
}

// adam

Adam::Adam(float lr, float beta1, float beta2, float epsilon, float weight_decay) {
    Adam::lr = lr;
    Adam::beta1 = beta1;
    Adam::beta2 = beta2;
    Adam::epsilon = epsilon;
    Adam::weight_decay = weight_decay;
    this->t = 0;
}

void Adam::add_parameter(Tensor &param, const Tensor &grad) {
    Adam::params.push_back(&param);
    Adam::grads.push_back(&grad);

    Tensor m_buffer(param.get_shape()[0], param.get_shape()[1]);
    m_buffer.fill(0.0f);

    Tensor v_buffer(param.get_shape()[0], param.get_shape()[1]);
    v_buffer.fill(0.0f);

    Adam::m_buffers.push_back(std::move(m_buffer));

    Adam::v_buffers.push_back(std::move(v_buffer));
}

__global__ void ker_adam_update(const float *grad, float *param, float *m_buffer, float *v_buffer,
                                float beta1, float beta2, float epsilon, float weight_decay, int t,
                                float lr, int n) {
    // auto grid = cg::this_grid();
    // auto block = cg::this_thread_block();

    // for (int i = grid.thread_rank(); i < n; i += grid.size()) {
    //     // printf("THREAD RANK %d", i);
    //     m_buffer[i] = beta1 * m_buffer[i] + (1 - beta1) * grad[i];
    //     v_buffer[i] = beta2 * v_buffer[i] + (1 - beta2) * grad[i] * grad[i];

    //     float moment1 = m_buffer[i] / (1 - powf(beta1, (float)t));
    //     float moment2 = v_buffer[i] / (1 - powf(beta2, (float)t));

    //     param[i] = param[i] * (1 - lr * weight_decay) - lr * moment1 / (sqrtf(moment2) +
    //     epsilon);
    // }
    // grid.sync();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    m_buffer[i] = beta1 * m_buffer[i] + (1 - beta1) * grad[i];
    v_buffer[i] = beta2 * v_buffer[i] + (1 - beta2) * grad[i] * grad[i];
    float moment1 = m_buffer[i] / (1 - powf(beta1, (float)t));
    float moment2 = v_buffer[i] / (1 - powf(beta2, (float)t));
    param[i] = param[i] * (1 - lr * weight_decay) - lr * moment1 / (sqrtf(moment2) + epsilon);
}

void Adam::step() {
    this->t++;

    for (int i = 0; i < this->params.size(); i++) {
        float *param = this->params[i]->getdata(), *m_buffer = this->m_buffers[i].getdata(),
              *v_buffer = this->v_buffers[i].getdata();

        const float *grad = this->grads[i]->getdata();
        int n = this->params[i]->get_shape()[0] * this->params[i]->get_shape()[1];
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        ker_adam_update<<<blocks, threads>>>(grad, param, m_buffer, v_buffer, this->beta1,
                                             this->beta2, this->epsilon, this->weight_decay,
                                             this->t, this->lr, n);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Adam kernel error " << cudaGetErrorString(err) << '\n';
        }
    }
}
} // namespace minitorch
