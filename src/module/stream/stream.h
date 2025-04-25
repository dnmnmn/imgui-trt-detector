//
// Created by Dongmin on 25. 4. 24.
//

#ifndef STREAM_H
#define STREAM_H

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
};



#endif //STREAM_H
