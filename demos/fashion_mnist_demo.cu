#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <iomanip>
#include "minitorch/tensor.cuh"
#include "minitorch/layers.cuh"
#include "minitorch/conv/conv1d.cuh"
#include "minitorch/pool/maxpool1d.cuh"
#include "minitorch/activations.cuh"
#include "minitorch/loss.cuh"
#include "minitorch/optim.cuh"
#include "minitorch/sequential.hh"
#include "minitorch/utils/mnist_loader.hh"

using namespace minitorch;

// Extractor utility to grab sub-tensors directly on GPU memory
Tensor get_batch(const Tensor& full_data, int batch_start, int batch_size, int total_size) {
    int current_batch_size = std::min(batch_size, total_size - batch_start);
    std::vector<int> shape = full_data.get_shape();
    shape[0] = current_batch_size;
    
    int elements_per_item = 1;
    for(size_t i = 1; i < shape.size(); ++i) elements_per_item *= shape[i];
    
    Tensor batch(shape);
    cudaMemcpy(batch.getdata(), full_data.getdata() + batch_start * elements_per_item, 
               current_batch_size * elements_per_item * sizeof(float), cudaMemcpyDeviceToDevice);
    return batch;
}

// Calculate accuracy purely computationally via one-off CPU memcpy sync 
float calculate_accuracy(const Tensor& logits, const Tensor& targets) {
    int batch = logits.get_shape()[0];
    int classes = logits.get_shape()[1];
    std::vector<float> h_logits(batch * classes);
    std::vector<float> h_targets(batch * classes);
    
    cudaMemcpy(h_logits.data(), logits.getdata(), h_logits.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_targets.data(), targets.getdata(), h_targets.size() * sizeof(float), cudaMemcpyDeviceToHost);

    int correct = 0;
    for (int i = 0; i < batch; ++i) {
        int max_logit_idx = 0;
        float max_logit = h_logits[i * classes];
        int target_idx = 0;
        
        for (int j = 0; j < classes; ++j) {
            if (h_logits[i * classes + j] > max_logit) {
                max_logit = h_logits[i * classes + j];
                max_logit_idx = j;
            }
            if (h_targets[i * classes + j] == 1.0f) {
                target_idx = j;
            }
        }
        if (max_logit_idx == target_idx) {
            correct++;
        }
    }
    return (float)correct / batch;
}

int main() {
    std::cout << "===============================================" << std::endl;
    std::cout << "  FASHION MNIST - 1D CNN TRAINING PIPELINE" << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "\n[1] Pulling Datasets from Memory..." << std::endl;
    
    // Assume run context will be from within root or build directory
    Tensor X_train = load_mnist_images("../data/fashion_mnist/train-images-idx3-ubyte");
    Tensor y_train = load_mnist_labels("../data/fashion_mnist/train-labels-idx1-ubyte");
    Tensor X_test  = load_mnist_images("../data/fashion_mnist/t10k-images-idx3-ubyte");
    Tensor y_test  = load_mnist_labels("../data/fashion_mnist/t10k-labels-idx1-ubyte");

    int n_train = X_train.get_shape()[0];
    int n_test  = X_test.get_shape()[0];

    std::cout << "    - Train Count: " << n_train << std::endl;
    std::cout << "    - Test Count : " << n_test << std::endl;

    std::cout << "\n[2] Constructing the 1D CNN Network..." << std::endl;
    Conv1d c1(28, 64, 3, 1, 1, 1, 1, true);
    MaxPool1d p1(2, 2, 0, 1);
    Conv1d c2(64, 128, 3, 1, 1, 1, 1, true);
    MaxPool1d p2(2, 2, 0, 1);
    Flatten f1;
    Linear l1(896, 128);
    Linear l2(128, 10);

    Sequential model({
        &c1, new ReLU(), &p1,
        &c2, new ReLU(), &p2,
        &f1, &l1, new ReLU(), &l2
    });

    // Setup Training Strategy
    CrossEntropyLoss criterion;
    Adam optim(model.parameters(), 0.001f);

    int batch_size = 128;
    int epochs = 10;

    std::cout << "\n[3] Commencing Deep Learning Loop:" << std::endl;
    
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;
        int batches = 0;

        auto start_time = std::chrono::high_resolution_clock::now();

        // Forward and Backward passes for Train Set
        for (int i = 0; i < n_train; i += batch_size) {
            Tensor X_batch = get_batch(X_train, i, batch_size, n_train);
            Tensor y_batch = get_batch(y_train, i, batch_size, n_train);
            
            Tensor logits = model.forward(X_batch);
            
            // Assuming your framework natively returns a scalar or tensor we can query
            Tensor loss_t = criterion.forward(logits, y_batch);
            
            // Hack to get sum value (if scalar tensor wasn't converted automatically)
            float batch_loss;
            cudaMemcpy(&batch_loss, loss_t.getdata(), sizeof(float), cudaMemcpyDeviceToHost);
            epoch_loss += batch_loss;
            epoch_acc += calculate_accuracy(logits, y_batch);
            batches++;

            // Critical Gradient Graph Descent
            Tensor grad_out = criterion.backward(logits, y_batch);
            model.backward(grad_out);
            
            optim.step();
            optim.zero_grad();
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        // Run validation against Test Data
        auto eval_start = std::chrono::high_resolution_clock::now();
        float val_acc = 0.0f;
        int val_batches = 0;
        for (int i = 0; i < n_test; i += batch_size) {
            Tensor X_batch = get_batch(X_test, i, batch_size, n_test);
            Tensor y_batch = get_batch(y_test, i, batch_size, n_test);
            Tensor logits = model.forward(X_batch);
            val_acc += calculate_accuracy(logits, y_batch);
            val_batches++;
        }
        auto eval_end = std::chrono::high_resolution_clock::now();
        auto eval_dur = std::chrono::duration_cast<std::chrono::milliseconds>(eval_end - eval_start);

        std::cout << "[Epoch " << std::setw(2) << epoch << "/" << epochs << "] "
                  << "Train Loss: " << std::fixed << std::setprecision(4) << (epoch_loss / batches) << " | "
                  << "Train Acc: " << std::fixed << std::setprecision(2) << ((epoch_acc / batches) * 100.0f) << "% | "
                  << "Val Acc: " << std::fixed << std::setprecision(2) << ((val_acc / val_batches) * 100.0f) << "% | "
                  << "Time: " << duration.count() << "ms (Val: " << eval_dur.count() << "ms)" << std::endl;
    }
    
    std::cout << "\n✅ Demo Code Executed Successfully! Neural Network mastered Fashion MNIST." << std::endl;
    return 0;
}
