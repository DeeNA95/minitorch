add_executable(test_matrix test_matrix.cu)
add_executable( test_ops test_ops.cu)
target_link_libraries(test_matrix PRIVATE minitorchlib)
target_link_libraries(test_ops PRIVATE minitorchlib)
