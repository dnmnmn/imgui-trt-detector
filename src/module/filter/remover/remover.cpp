//
// Created by Dongmin on 25. 4. 30.
//

#include "remover.h"

bool Remover::Initialize() {
    name_ = "Remover";
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
    JsonObject config_json;
    config_json.load(config_path_);
    debug_time_ = config_json.get_int("Engine/Log/DebugTime") * 1000;
    width_ = config_json.get_int("Engine/Stream/Width");
    height_ = config_json.get_int("Engine/Stream/Height");

    cudaStreamCreate(&stream_);
    filter_image_ = cv::Mat(height_, width_, CV_32FC3, cv::Scalar(0, 0, 0));
    // Set the GPU image memory
    cudaMalloc(&gpu_filter_, width_ * height_ * 3 * sizeof(uchar));
    cudaMemset(gpu_filter_, 0, width_ * height_ * 3 * sizeof(uchar));
    return true;
}

void Remover::Release()
{
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
    cudaFree(gpu_filter_);
    filter_image_.release();
}

void Remover::AddFrame(std::shared_ptr<cv::Mat> _frame) {
    if (first_time_) {
        cudaMemcpyAsync(gpu_filter_, container_->gpu_data_, width_ * height_ * 3 * sizeof(uchar), cudaMemcpyDeviceToDevice, stream_);
        first_time_ = false;
        return;
    }
    gpu_set_mask();
    gpu_remover((unsigned char*)container_->gpu_data_);
    return;
}

void Remover::GetImage(cv::Mat &_filtered_image) {
    cudaMemcpyAsync((void*)_filtered_image.data, gpu_filter_,  width_ * height_ * 3 * sizeof(uchar), cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
}