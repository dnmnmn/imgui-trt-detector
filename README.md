# imgui-trt-detector
## Overview
C++(TensorRT) 기반의 추론 프로그램. <br>
비디오 영상에 대해 Detection을 수행하고, <br>결과로 나온 Object에 대해 Segmentation 수행
* OS : Unbuntu 22.04
* GUI : Dear ImGui
* inference : TensorRT
* video player : opencv
* pre/post postprocess : cuda programing

## Install
* TBB 2022.1.0
  * https://github.com/uxlfoundation/oneTBB/releases
* opencv 4.10.0
  * https://github.com/opencv/opencv/tree/4.10.0
* cuda toolkit 12.4.1
* cudnn 8.9.7
* TensorRT 10.7.0
* ImGui
  * https://github.com/ocornut/imgui
  * glfw 3.4
  * opengl3

## Model
* yolo base model
* weights : .onnx / .engine

## Usage
* config/config.json 파일 편집
  * Detect, Segment weights 경로 설정
    * .onnx 파일 경로를 설정하면 .engine파일로 자동 변환
  * video 경로 설정 (Engine/Stream/Stream)

![Segmentation](https://github.com/dnmnmn/imgui-trt-detector/tree/main/data/Segmentation.png)

![Segmentation](https://github.com/dnmnmn/imgui-trt-detector/tree/main/data/Remover.png)