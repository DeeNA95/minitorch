# Define the executable target and list all source files
file(GLOB_RECURSE SOURCE_FILES CONFIGURE_DEPENDS "*.cu")

add_library(${PROJECT_NAME} ${SOURCE_FILES})

install(
  TARGETS minitorchlib
  EXPORT minitorchlibtargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/ DESTINATION include)

target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:include>)
