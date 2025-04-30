cmake_minimum_required(VERSION 3.28)

project(imgui_trt)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
enable_language(CUDA)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/imgui)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends)
include_directories("~/include")
# include_directories("~/include/SDL3")
include_directories("/usr/local/cuda/include")

include_directories("src")
find_package(OpenGL REQUIRED)
find_package(SDL3 REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})

link_directories("/home/gopizza/lib")
link_directories("/usr/local/cuda/lib64")

add_executable(imgui_trt
    main.cpp
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_internal.h
    imgui/imconfig.h
    imgui/imgui.h
    imgui/imstb_rectpack.h
    imgui/imstb_textedit.h
    imgui/imstb_truetype.h
        imgui/backends/imgui_impl_glfw.cpp
        imgui/backends/imgui_impl_glfw.h

    imgui/backends/imgui_impl_opengl3.cpp
    imgui/backends/imgui_impl_opengl3.h
    imgui/backends/imgui_impl_opengl3_loader.h

        src/module/stream/stream.cpp
        src/module/stream/stream.h
    src/data/pipeline.cpp
    src/data/pipeline.h
    src/module/module.cpp
    src/module/module.h

        src/utils/trt/dm_engine.cpp
        src/utils/trt/dm_engine.h
        src/utils/trt/dm_buffer.cpp
        src/utils/trt/dm_buffer.h
    src/utils/trt/logger.h

    src/utils/DateTime/DateTime.cpp
    src/utils/DateTime/DateTime.h
    src/utils/Logger/Logger.cpp
    src/utils/Logger/Logger.h
    src/utils/Json/Json.hpp
    src/utils/Json/JsonObject.cpp
    src/utils/Json/JsonObject.h
    src/utils/Timer/Timer.cpp
    src/utils/Timer/Timer.h
    src/utils/FileSystem/FileSystem.cpp
    src/utils/FileSystem/FileSystem.h
        src/utils/ColorMap/ColorMap.cpp
        src/utils/ColorMap/ColorMap.h
    src/data/shape.h
    src/data/box/bbox.h
    src/data/box/trackbox.h
    src/data/container/container.h
    src/module/display/display.cpp
    src/module/display/display.h
        src/imgui_trt.cpp
        src/imgui_trt.h
        src/application/application.cpp
        src/application/application.h
        src/module/model/model.cpp
        src/module/model/model.h
        src/module/model/segmentation/segmentation.cpp
        src/module/model/segmentation/segmentation.h
        src/module/model/preprocess/preprocess.cpp
        src/module/model/postprocess/postprocess.cpp
        src/module/model/postprocess/postprocess.h
        src/module/model/preprocess/preprocess.h
        src/module/model/preprocess/preprocess.cu
        src/module/model/detection/detection.cpp
        src/module/model/detection/detection.h
        src/module/draw/draw.cpp
        src/module/draw/draw.h
        src/module/tracker/tracker.cpp
        src/module/tracker/tracker.h
        src/module/tracker/hungarian/hungarian_algorithm.cpp
        src/module/tracker/hungarian/hungarian_algorithm.hpp
        src/module/filter/filter.cpp
        src/module/filter/filter.h
        src/module/filter/remover/remover.cpp
        src/module/filter/remover/remover.h

)

target_link_libraries(imgui_trt

)

set_target_properties(imgui_trt PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(imgui_trt
        # PRIVATE SDL3
        PRIVATE glfw3 glfw
        ${OPENGL_LIBRARIES}
        PRIVATE cuda cudart cudnn cublas nppig nppc nppial nppim nppidei
        PRIVATE nvinfer nvonnxparser
        PRIVATE opencv_world
        PRIVATE tbb
        stdc++fs
)