//
// Created by Dongmin on 25. 4. 24.
//

#include "stream.h"

bool Stream::Initialize() {
    name_ = "Video";
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
    JsonObject config_json;
    if(FileSystem::exist(config_path_)) {
        config_json.load(config_path_);
    } else assert(false);
    debug_time_ = config_json.get_int("GoEngine/Log/DebugTime") * 1000;
    sleep_time_ = config_json.get_int("GoEngine/Log/SleepTime");
    string video_path = config_json.get_string("GoEngine/Stream/Stream");
    capture_.open(video_path);
    if (!capture_.isOpened()) {
        std::cerr << "Error: Video::initialize() - Could not open video file: " << video_path << std::endl;
        return false;
    }
    return true;
}

void Stream::Release() {
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
    capture_.release();
}


void Stream::Run() {
    uint index = 0;
    if(data_store_->module_index_queues_[input_module_index_].try_pop(index)) {
        cv::Mat frame;
        if (capture_.read(frame)) {
            auto container = data_store_->contaiers_[index];
            container->org_image_ = std::make_shared<cv::Mat>(frame);
        }
        else {
            DM::Logger::GetInstance().Log("Video Done", LOGLEVEL::WARN);
            stop_flag_ = true;
        }

        ++pass_count_;
        data_store_->module_index_queues_[output_module_index_].push(index);
    }
    else {
        ++fail_count_;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_));
    }
}