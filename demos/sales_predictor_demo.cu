// ============================================================================
// MiniTorch C++ Framework - Sales Predictor Demo
// ============================================================================
// A minimal, PyTorch-like deep learning framework built from scratch in C++/CUDA.
// Demonstrates custom MemoryPool allocator, Sequential API, and Adam Optimizer.
// 
// Training on ~350k samples of real-world sales data using a 3-layer DNN.
// ============================================================================

#include "minitorch/activations.cuh"
#include "minitorch/layers.cuh"
#include "minitorch/loss.cuh"
#include "minitorch/matrix.cuh"
#include "minitorch/optim.cuh"
#include "minitorch/sequential.hh"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace minitorch;

// ----------------------------------------------------------------------------
// Utility: Fast Binary Loader directly to GPU memory
// ----------------------------------------------------------------------------
Matrix load_to_gpu(const std::string &filepath, int rows, int cols) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open: " + filepath);

    std::vector<float> cpu_data(rows * cols);
    file.read(reinterpret_cast<char *>(cpu_data.data()), cpu_data.size() * sizeof(float));

    Matrix M(rows, cols);
    M.to_device(cpu_data.data()); // Transfers directly to custom Caching Allocator
    return M;
}

// ----------------------------------------------------------------------------
// Main Training Loop
// ----------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    std::cout << "\n🚀 Initialize MiniTorch C++ Backend...\n" << std::endl;

    // Hyperparameters
    const int EPOCHS = (argc > 1) ? std::atoi(argv[1]) : 10;
    const float LR = (argc > 2) ? std::atof(argv[2]) : 0.001f;
    const int BATCH_SIZE = 64;
    
    // Dataset Dimensions
    const int NUM_SAMPLES = 358843;
    const int NUM_FEATURES = 18;

    try {
        // 1. Load Data
        std::cout << "[1/4] Loading Dataset onto Device Memory..." << std::endl;
        Matrix X = load_to_gpu("../data/X_sales.bin", NUM_SAMPLES, NUM_FEATURES);
        Matrix y = load_to_gpu("../data/y_sales.bin", NUM_SAMPLES, 1);
        
        std::cout << "      X shape: (" << X.getrows() << ", " << X.getcols() << ")" << std::endl;
        std::cout << "      y shape: (" << y.getrows() << ", " << y.getcols() << ")\n" << std::endl;

        // 2. Define Neural Network Architecture
        std::cout << "[2/4] Constructing Sequential Computational Graph..." << std::endl;
        Linear l1(NUM_FEATURES, 64000);
        Linear l2(64000, 3200);
        Linear l3(3200, 1);

        Sequential model({
            &l1,
            new Relu(),
            &l2,
            new Sigmoid(),
            &l3
        });
        std::cout << "      Model architecture built successfully.\n" << std::endl;

        // 3. Setup Optimizer
        std::cout << "[3/4] Initializing Adam Optimizer (LR=" << LR << ")..." << std::endl;
        Adam optimizer(LR);
        
        // Register parameters
        for (Matrix* param : model.parameters()) {
            // Note: In a fully complete v2, parameters() would return paired Gradients too.
            // For now, mapping explicitly handles the C++ memory layout.
        }
        optimizer.add_parameter(l1.get_weights(), l1.get_grad_weights());
        optimizer.add_parameter(l1.get_bias(), l1.get_grad_bias());
        optimizer.add_parameter(l2.get_weights(), l2.get_grad_weights());
        optimizer.add_parameter(l2.get_bias(), l2.get_grad_bias());
        optimizer.add_parameter(l3.get_weights(), l3.get_grad_weights());
        optimizer.add_parameter(l3.get_bias(), l3.get_grad_bias());
        
        std::cout << "      Optimizer ready.\n" << std::endl;

        // 4. Training Loop Prep (Shuffling Indices)
        std::vector<int> cpu_indices(NUM_SAMPLES);
        std::iota(cpu_indices.begin(), cpu_indices.end(), 0);
        
        int *gpu_index_ptr;
        cudaMalloc(&gpu_index_ptr, sizeof(int) * NUM_SAMPLES);
        
        std::random_device rd;
        std::mt19937 gen(rd());

        std::cout << "[4/4] Commencing Training Loop...\n" << std::endl;
        std::cout << std::fixed << std::setprecision(6);

        // Epochs
        for (int e = 0; e < EPOCHS; e++) {
            std::cout << "==========================================================" << std::endl;
            std::cout << " EPOCH " << e + 1 << "/" << EPOCHS << std::endl;
            std::cout << "==========================================================" << std::endl;

            // Shuffle data efficiently by moving pointer indices, not heavy matrices
            std::shuffle(cpu_indices.begin(), cpu_indices.end(), gen);
            cudaMemcpy(gpu_index_ptr, cpu_indices.data(), cpu_indices.size() * sizeof(int), cudaMemcpyHostToDevice);

            float epoch_loss = 0.0f;
            int loss_reports = 0;

            // Mini-Batches
            for (int batch_start = 0; batch_start < NUM_SAMPLES; batch_start += BATCH_SIZE) {
                int current_batch = std::min(BATCH_SIZE, NUM_SAMPLES - batch_start);

                // Zero-copy extraction via CUDA kernel
                Matrix batch_X = X.extract_batch(gpu_index_ptr, batch_start, current_batch);
                Matrix batch_y = y.extract_batch(gpu_index_ptr, batch_start, current_batch);

                // --- FORWARD PASS ---
                Matrix out = model.forward(batch_X);

                // Logging
                if (batch_start % 25600 == 0) { // Every ~400 batches
                    float loss = mse_forward(out, batch_y);
                    std::cout << "   [Step " << std::setw(6) << batch_start << "/" << NUM_SAMPLES 
                              << "] MSE Loss: " << loss << std::endl;
                    epoch_loss += loss;
                    loss_reports++;
                }

                // --- BACKWARD PASS ---
                Matrix batch_grad = mse_backward(out, batch_y);
                model.backward(batch_grad);

                // --- OPTIMIZE ---
                optimizer.step();
            }
            
            std::cout << "   >>> Avg Sampled Loss: " << (epoch_loss / loss_reports) << "\n\n";
        }

        cudaFree(gpu_index_ptr);
        std::cout << "✅ Training Completed Successfully." << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "💥 Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
