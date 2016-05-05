cmake_minimum_required(VERSION 3.2)
project(Halstm)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# default locations
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

# extra dependencies
include(cmake/utils.cmake)
include(cmake/dep.cmake)

# 3rdparty
add_subdirectory(3rdparty/caffe)

# includes
include_directories(include)
include_directories(3rdparty/caffe/include)

# collect src
set(SRC "")
file(GLOB LAYER_SRC "${PROJECT_SOURCE_DIR}/src/layer/*.cpp")
list(APPEND SRC ${LAYER_SRC})
#file(GLOB ROOT_SRC "${PROJECT_SOURCE_DIR}/src/*.cpp")
#list(APPEND SRC ${ROOT_SRC})
list(APPEND SRC "${PROJECT_SOURCE_DIR}/src/maths.cpp")
list(APPEND SRC "${PROJECT_SOURCE_DIR}/src/utils.cpp")

# CxxTest
find_package(CxxTest)
if(CXXTEST_FOUND)
    set(CXXTEST_TESTGEN_INTERPRETER bash)  # FindCxxTest module bug
    include_directories(${CXXTEST_INCLUDE_DIR})
    enable_testing()

    # test_layer
    CXXTEST_ADD_TEST(test_layer test_layer_lstm.cpp
            ${PROJECT_SOURCE_DIR}/test/test_layer_lstm.h
            ${SRC} ${CAFFE_SRC})
    target_link_libraries(test_layer ${LIBS})

    # test_matrix
    CXXTEST_ADD_TEST(test_matrix test_matrix.cpp
            ${PROJECT_SOURCE_DIR}/test/test_matrix.h
            ${SRC} ${CAFFE_SRC})
    target_link_libraries(test_matrix ${LIBS})
endif()

# binaries
#set(SRC main.cpp)
#add_executable(Halstm ${SRC})
