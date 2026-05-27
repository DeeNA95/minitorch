#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include "minitorch/tensor.cuh"

namespace minitorch {

inline int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

inline Tensor load_mnist_images(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << path << std::endl;
        exit(1);
    }
    
    int magic_number = 0, n_images = 0, n_rows = 0, n_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    if (magic_number != 2051) {
        std::cerr << "Invalid MNIST image file! (Magic number " << magic_number << ")" << std::endl;
        exit(1);
    }
    
    file.read((char*)&n_images, sizeof(n_images));
    n_images = reverseInt(n_images);
    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);
    
    // We treat 28 rows as 28 channels, and 28 columns as sequence length for our 1D CNN implementation
    Tensor data_tensor(n_images, n_rows, n_cols);
    std::vector<float> host_data(n_images * n_rows * n_cols);
    
    for (int i = 0; i < n_images * n_rows * n_cols; ++i) {
        unsigned char px=0;
        file.read((char*)&px, sizeof(px));
        host_data[i] = (float)px / 255.0f; // Normalize pixel intensities to [0, 1] range
    }
    
    cudaMemcpy(data_tensor.getdata(), host_data.data(), host_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    return data_tensor;
}

inline Tensor load_mnist_labels(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << path << std::endl;
        exit(1);
    }
    
    int magic_number = 0, n_items = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    if (magic_number != 2049) {
        std::cerr << "Invalid MNIST label file! (Magic number " << magic_number << ")" << std::endl;
        exit(1);
    }
    
    file.read((char*)&n_items, sizeof(n_items));
    n_items = reverseInt(n_items);
    
    // One-hot encode the labels to match CrossEntropyLoss expectations [batch, 10]
    Tensor label_tensor(n_items, 10);
    label_tensor.fill(0.0f);
    
    std::vector<float> host_data(n_items * 10, 0.0f);
    for (int i = 0; i < n_items; ++i) {
        unsigned char label=0;
        file.read((char*)&label, sizeof(label));
        host_data[i * 10 + (int)label] = 1.0f;
    }
    
    cudaMemcpy(label_tensor.getdata(), host_data.data(), host_data.size() * sizeof(float), cudaMemcpyHostToDevice);
    return label_tensor;
}

} // namespace minitorch
