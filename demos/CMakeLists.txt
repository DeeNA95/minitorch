add_executable(sales_demo sales_predictor_demo.cu)
target_link_libraries(sales_demo PRIVATE minitorchlib)
set_target_properties(sales_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
