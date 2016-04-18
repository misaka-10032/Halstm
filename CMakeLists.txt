cmake_minimum_required(VERSION 3.2)
project(Halstm)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# includes
include_directories(/usr/local/include)
include_directories(include)
include_directories(3rdparty)

# libraries
link_directories(/usr/local/lib)

set(SRC main.cpp)
add_executable(Halstm ${SRC})

set(SRC src/maths.cpp
        3rdparty/hblas/halide_blas.cpp)
add_library(Halstm-maths ${SRC})

# CxxTest
find_package(CxxTest)
if(CXXTEST_FOUND)
    set(CXXTEST_TESTGEN_INTERPRETER bash)  # FindCxxTest module bug
    include_directories(${CXXTEST_INCLUDE_DIR})
    enable_testing()
    CXXTEST_ADD_TEST(test_layer test_layer_lstm.cpp
            ${PROJECT_SOURCE_DIR}/test/test_layer_lstm.h)
    target_link_libraries(test_layer Halstm-maths)
endif()

add_executable(play src/play.cpp)