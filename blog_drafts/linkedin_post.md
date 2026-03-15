I built a Deep Learning Framework from scratch in C++/CUDA (POST 2)

After weeks of wrestling with memory leaks, shared memory tiling, and backpropagation math, I finally have my own engine (MiniTorch) training on 350k+ real-world samples using a custom GPU Caching Allocator and the Adam optimizer.

This is the next step in my journey to learn the nitty gritty of Machine Learning using CUDA

I just posted a deep dive into:
✅ Why I chose a **Module-based architecture** over PyTorch's Tensor Autograd (for now).
✅ How **CNN intuition** helped me optimize my MatMul kernels.
✅ Implementing an **Object-Oriented Sequential API** in C++ for clean, readable deep learning code.

Check out the full blog post here: blog.derekamuna.com/posts/minitorch_demo

Video below: MiniTorch in action training a 3-layer Sales Prediction model on a Vast.ai 3090 instance.

#C++ #CUDA #DeepLearning #GPU #MachineLearning #SoftwareEngineering #FromScratch
