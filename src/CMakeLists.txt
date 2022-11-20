set(lib_name cuda_tools)

set(${lib_name}_sources
    cuda_tools/types.h
    cuda_tools/error.h
    cuda_tools/exception.h
    cuda_tools/warp.cuh
    cuda_tools/random.h
    cuda_tools/random.cuh
    cuda_tools/random.cu
    cuda_tools/timer.h
    cuda_tools/timer.cu
)

generate_cpp_sources(FILES ${${lib_name}_sources})

# find_package(Eigen3 3.3 REQUIRED NO_MODULE)
# find_package(OpenCV REQUIRED)
# find_package(PCL 1.12.1 REQUIRED COMPONENTS visualization)

# add_definitions(${PCL_DEFINITIONS})
# add_library(${lib_name} INTERFACE)

# target_include_directories(${lib_name} INTERFACE
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
# ) 

add_library(${lib_name} STATIC ${${lib_name}_sources})
target_include_directories(${lib_name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
) 
target_compile_features(${lib_name} PUBLIC cxx_std_14)

list(FILTER ${lib_name}_sources INCLUDE REGEX ".\\.h$")
message("headers: ${${lib_name}_sources}")
set_target_properties(${lib_name} PROPERTIES PUBLIC_HEADER "${${lib_name}_sources}")
set_target_properties(${lib_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# target_link_libraries(${lib_name} 
#     Eigen3::Eigen
#     ${OpenCV_LIBS}
#     ${PCL_LIBRARIES}
# )

INSTALL(TARGETS ${lib_name}
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/cuda_tools)