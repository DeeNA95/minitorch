# Find all CUDA source files in the current (tests) directory
file(GLOB TEST_SOURCES "*.cu")

foreach(TEST_SOURCE ${TEST_SOURCES})
    # Get the file name without extension to use as target name
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    
    # Add an executable for each test file
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    
    # Link each test to the main library
    target_link_libraries(${TEST_NAME} PUBLIC minitorchlib)
    
    # Include headers from parent include directory
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)
endforeach()
