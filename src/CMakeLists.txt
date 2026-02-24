# Define the executable target and list all source files
add_library(${PROJECT_NAME} matrix.cu ops.cu layers.cu activations.cu loss.cu
                            optim.cu)

# Link against the necessary CUDA runtime library target_link_libraries
# (${PROJECT_NAME} PRIVATE CUDA::cudart)
