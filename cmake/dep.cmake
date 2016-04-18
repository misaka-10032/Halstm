include(ExternalProject)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)
set(LIBS "")

# halide
# TODO: extract to FindHalide
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
list(APPEND LIBS "-lHalide")

# blas
if(NOT APPLE)
    set(BLAS "Atlas" CACHE STRING "Selected BLAS library")
    set_property(CACHE BLAS PROPERTY STRINGS "Atlas;Open;MKL")

    if(BLAS STREQUAL "Atlas" OR BLAS STREQUAL "atlas")
        find_package(Atlas REQUIRED)
        include_directories(SYSTEM ${Atlas_INCLUDE_DIR})
        list(APPEND LIBS ${Atlas_LIBRARIES})
    elseif(BLAS STREQUAL "Open" OR BLAS STREQUAL "open")
        find_package(OpenBLAS REQUIRED)
        include_directories(SYSTEM ${OpenBLAS_INCLUDE_DIR})
        list(APPEND LIBS ${OpenBLAS_LIB})
    elseif(BLAS STREQUAL "MKL" OR BLAS STREQUAL "mkl")
        find_package(MKL REQUIRED)
        include_directories(SYSTEM ${MKL_INCLUDE_DIR})
        list(APPEND LIBS ${MKL_LIBRARIES})
        add_definitions(-DUSE_MKL)
    endif()
elseif(APPLE)
    find_package(vecLib REQUIRED)
    include_directories(SYSTEM ${vecLib_INCLUDE_DIR})
    list(APPEND LIBS ${vecLib_LINKER_LIBS})
endif()

# boots
find_package(Boost 1.46 REQUIRED COMPONENTS system thread)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
list(APPEND LIBS ${Boost_LIBRARIES})

# threads
find_package(Threads REQUIRED)
list(APPEND LIBS ${CMAKE_THREAD_LIBS_INIT})

# glog
include("cmake/External/glog.cmake")
include_directories(SYSTEM ${GLOG_INCLUDE_DIRS})
list(APPEND LIBS ${GLOG_LIBRARIES})

# gflags
include("cmake/External/gflags.cmake")
include_directories(SYSTEM ${GFLAGS_INCLUDE_DIRS})
list(APPEND LIBS ${GFLAGS_LIBRARIES})

# protobuf
include(cmake/ProtoBuf.cmake)