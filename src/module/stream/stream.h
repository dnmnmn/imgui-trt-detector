//
// Created by Dongmin on 25. 4. 24.
//

#ifndef STREAM_H
#define STREAM_H

#include <driver_types.h>
#include <memory>
#include <tbb/concurrent_queue.h>
#include <thread>
#include <opencv2/opencv.hpp>
#include "module/module.h"
class Stream : public Module {
public:
    Stream() {};
    ~Stream() {};

    bool Initialize() override;
    void Release() override;
    void Run() override;
private:
    cv::VideoCapture capture_;
    cudaStream_t input_cuda_stream_;
    int height_;
    int width_;
};



#endif //STREAM_H
