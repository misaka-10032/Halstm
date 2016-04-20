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

# CxxTest
find_package(CxxTest)
if(CXXTEST_FOUND)
    set(CXXTEST_TESTGEN_INTERPRETER bash)  # FindCxxTest module bug
    include_directories(${CXXTEST_INCLUDE_DIR})
    enable_testing()
    CXXTEST_ADD_TEST(test_layer test_layer_lstm.cpp
            ${PROJECT_SOURCE_DIR}/test/test_layer_lstm.h
            ${CAFFE_SRC})
    target_link_libraries(test_layer ${LIBS})
endif()

add_executable(play src/play.cpp)
target_link_libraries(play ${LIBS})

# binaries
set(SRC main.cpp)
add_executable(Halstm ${SRC})
