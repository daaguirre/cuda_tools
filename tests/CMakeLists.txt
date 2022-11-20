add_definitions(-DRESOURCES_DIR_PATH="${CMAKE_CURRENT_LIST_DIR}/resources")

SET(app_name cuda_tools_tests)

SET(${app_name}_sources
    test_base.h
    test_base.cpp
    cuda_tools/test_random.h
    cuda_tools/test_random.cpp
)

generate_cpp_sources(FILES ${${app_name}_sources})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR})

# Test dependencies
conan_cmake_configure(
    REQUIRES 
    gtest/1.12.1
    GENERATORS cmake_find_package)
conan_cmake_autodetect(settings)
conan_cmake_install(PATH_OR_REFERENCE .
                    BUILD missing
                    REMOTE conancenter
                    SETTINGS ${settings})
find_package(GTest)

add_executable(${app_name} ${${app_name}_sources})

target_include_directories(${app_name} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
) 

target_link_libraries(${app_name} PUBLIC GTest::gtest_main PRIVATE cuda_tools)

target_compile_features(${app_name} PUBLIC cxx_std_14)

set_target_properties(${app_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
