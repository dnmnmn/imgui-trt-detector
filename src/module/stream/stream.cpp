//
// Created by Dongmin on 25. 4. 24.
//

#include "stream.h"
#include <cuda_runtime_api.h>

bool Stream::Initialize() {
    name_ = "Video";
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
    JsonObject config_json;
    if(FileSystem::exist(config_path_)) {
        config_json.load(config_path_);
    } else assert(false);
    debug_time_ = config_json.get_int("Engine/Log/DebugTime") * 1000;
    sleep_time_ = config_json.get_int("Engine/Log/SleepTime");
    string video_path = config_json.get_string("Engine/Stream/Stream");
    height_ = config_json.get_int("Engine/Stream/Height");
    width_ = config_json.get_int("Engine/Stream/Width");
    capture_.open(video_path);
    if (!capture_.isOpened()) {
        DM::Logger::GetInstance().Log("Video::initialize() - Could not open video file: " + video_path, LOGLEVEL::ERROR);
        return false;
    }
    if (width_ != capture_.get(cv::CAP_PROP_FRAME_WIDTH) || height_ != capture_.get(cv::CAP_PROP_FRAME_HEIGHT))
    {
        config_json.set_int("Engine/Stream/Height", (int)capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
        config_json.set_int("Engine/Stream/Width", (int)capture_.get(cv::CAP_PROP_FRAME_WIDTH));
        config_json.save(config_path_);
        DM::Logger::GetInstance().Log("Video::initialize() - Video Width or Height Size Error, Please Restart", LOGLEVEL::ERROR);
        return false;
    }
    cudaStreamCreate(&input_cuda_stream_);

    return true;
}

void Stream::Release() {
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
    capture_.release();
    cudaStreamDestroy(input_cuda_stream_);
}


void Stream::Run() {
    uint index = 0;
    if(data_store_->module_index_queues_[input_module_index_].try_pop(index)) {
        cv::Mat frame;
        if (capture_.read(frame)) {
            if (frame.empty()) {
                DM::Logger::GetInstance().Log("Error: VideoStream::update_frame() - Could not read frame from video file", LOGLEVEL::WARN);
                data_store_->module_index_queues_[input_module_index_].push(index);
            }
            auto container = data_store_->contaiers_[index];
            container->org_image_ = std::make_shared<cv::Mat>(frame);
            cudaMemcpyAsync(container->gpu_data_,
                       container->org_image_->data,
                       data_store_->org_image_shape_->size_,
                       cudaMemcpyHostToDevice, input_cuda_stream_);
            cudaStreamSynchronize(input_cuda_stream_);
        }
        else {
            DM::Logger::GetInstance().Log("Video Done", LOGLEVEL::WARN);
            capture_.set(cv::CAP_PROP_POS_FRAMES, 0);
        }

        ++pass_count_;
        data_store_->module_index_queues_[output_module_index_].push(index);
    }
    else {
        ++fail_count_;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_));
    }
}