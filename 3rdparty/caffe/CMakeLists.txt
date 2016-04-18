file(GLOB CAFFE_SRC
        src/*.cpp
        src/layers/*.cpp
        src/utils/*.cpp
        include/caffe/proto/*.cc)
set(CAFFE_SRC ${CAFFE_SRC} PARENT_SCOPE)
