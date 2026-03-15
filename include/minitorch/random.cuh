#include <ctime>
#include <curand.h>

namespace minitorch {

class Random_Manager {

private:
    curandGenerator_t gen;
    unsigned long long seed;
    Random_Manager() {
        this->seed = time(NULL);
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);
        this->gen = gen;
    };

public:
    static Random_Manager &instance() {
        // A 'static' local variable lives through the whole program
        static Random_Manager self;
        return self;
    }
    ~Random_Manager() {
        curandDestroyGenerator(gen);
    };

    void uniform(float *data, int size) {
        curandGenerateUniform(gen, data, size);
    };

    void normal(float *data, int size) {
        curandGenerateNormal(gen, data, size, 0.0f, 1.0f);
    };
};

} // namespace minitorch
