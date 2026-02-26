# 🔥 MiniTorch

A neural network framework built from scratch in C++ & CUDA.

> See [ROADMAP.md](ROADMAP.md) for full details, code examples, and benchmarks for each phase.

---

## Progress

### Phase 1 — The Matrix: GPU Memory Management

- [-] `Matrix` class with `cudaMalloc` / `cudaFree` (RAII)
- [-] `to_device()` and `to_host()` with `cudaMemcpy`
- [-] `fill()` CUDA kernel
- [-] `print()` helper
- [-] CMake build system
- [-] **Deliverable**: Create, fill, round-trip, and print a 4×4 GPU matrix

### Phase 2 — Matrix Math: First Real Kernels

- [-] Element-wise add kernel
- [-] Element-wise multiply kernel
- [-] Scalar multiply kernel
- [-] Matrix transpose kernel
- [-] Naive matrix multiply kernel
- [-] **Deliverable**: MatMul correctness test (max error < 1e-4)
- [ ] **Benchmark**: Compare matmul speed vs PyTorch

### Phase 3a — Forward Pass: Making Predictions

- [-] `Linear` layer forward (matmul + bias)
- [-] `sigmoid_forward` kernel
- [-] `relu_forward` kernel
- [-] Manual weight setting from CPU
- [-] **Deliverable**: Forward pass with known weights matches hand-calculated values

### Phase 3b — Backward Pass: Learning from Mistakes

- [-] `Linear::backward()` — gradient w.r.t. input, weights, bias
- [-] `sigmoid_backward` kernel
- [-] `relu_backward` kernel
- [-] MSE loss (forward + backward)
- [-] SGD optimizer
- [-] Gradient checking with finite differences
- [-] **Deliverable**: Train XOR — loss → ~0, predictions match truth table
- [-] **Benchmark**: XOR training speed vs PyTorch

### Phase 4 — Real Data: sales.tsv Regression

- [ ] TSV parser (C++ `fstream`)
- [ ] Feature extraction (quantity, unit_price, category one-hot, day_of_year)
- [ ] Feature normalization kernels (mean/std)
- [ ] Mini-batch iterator with shuffling
- [ ] Train/test split (80/20)
- [ ] **Deliverable**: Train on ~366K rows, test MAE beats predict-the-mean baseline
- [ ] **Benchmark**: Training speed & memory vs PyTorch

### Phase 5 — Optimization & Abstractions

- [ ] Shared memory tiled matmul kernel
- [ ] Simple memory pool
- [ ] `Sequential` container API
- [ ] Adam optimizer
- [ ] Xavier/He initialization with `cuRAND`
- [ ] **Deliverable**: Tiled matmul passes correctness, Sequential trains sales model
- [ ] **Benchmark**: Tiled vs naive matmul speedup

### Phase 6 — CNNs: Conv2d + MNIST

- [ ] `Conv2d` forward kernel (naive)
- [ ] `Conv2d` backward kernel
- [ ] `MaxPool2d` forward + backward
- [ ] Softmax + Cross-Entropy loss
- [ ] Flatten operation
- [ ] MNIST data loader (IDX binary format)
- [ ] **Deliverable**: CNN achieves >95% test accuracy on MNIST
- [ ] **Benchmark**: Training speed & memory vs PyTorch

### Phase 7 *(Tentative)* — Transformer

- [ ] Scaled dot-product attention
- [ ] Multi-head attention
- [ ] LayerNorm kernel
- [ ] Causal masking
- [ ] Positional encoding + embedding lookup
- [ ] TransformerBlock (attention + FFN + residuals)
- [ ] **Deliverable**: Character-level Shakespeare text generation

### Phase 8 — Bonus Projects

- [ ] Flash Attention
- [ ] Mixed Precision (FP16)
- [ ] Custom Autograd (computation graph)
- [ ] Model Serialization (save/load weights)
- [ ] Python Bindings (`pybind11`)
- [ ] Profiling with `nsys` / `ncu`
