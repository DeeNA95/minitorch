#include <mutex>
#include <unordered_map>
#include <vector>

namespace minitorch {

class MemoryPool {

private:
    std::unordered_map<size_t, std::vector<float *>> cache;
    // unordered map is a hashmap here with byte size as the key and a vector of float points(Matrix
    // data) as the key
    std::mutex cache_mutex;
    MemoryPool() = default; // private constructor
public:
    static MemoryPool &instance() {
        static MemoryPool pool;
        return pool;
    }
    float *allocate(size_t bytes);
    void deallocate(float *ptr, size_t bytes);
    ~MemoryPool(); // calls cudaFree on ALL cached pointers
};
} // namespace minitorch
