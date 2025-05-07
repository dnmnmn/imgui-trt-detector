//
// Created by Dongmin on 25. 4. 30.
//

#include "filter.h"

bool Filter::Initialize() {
    name_ = "Filter";
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
    JsonObject config_json;
    config_json.load(config_path_);
    debug_time_ = config_json.get_int("Engine/Log/DebugTime") * 1000;
    width_ = config_json.get_int("Engine/Segment/MaskWidth");
    height_ = config_json.get_int("Engine/Segment/MaskHeight");

    filter_image_ = cv::Mat(height_, width_, CV_8UC3, cv::Scalar(0, 0, 0));
    return true;
}

void Filter::Release() {
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
    filter_image_.release();
}

void Filter::Run() {
    uint index = 0;
    if(data_store_->module_index_queues_[input_module_index_].try_pop(index)) {
        container_ = data_store_->contaiers_[index];
        if (container_->org_image_->empty()) {
            data_store_->module_index_queues_[output_module_index_].push(index);
            return;
        }

        AddFrame(container_->org_image_);
        GetImage(container_->filtered_image_);

        ++pass_count_;
        data_store_->module_index_queues_[output_module_index_].push(index);
    }
    else {
        ++fail_count_;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_));
    }
}

void Filter::AddFrame(std::shared_ptr<cv::Mat> _frame) {
    // Implement the filtering logic here
}

void Filter::GetImage(cv::Mat &_filtered_image) {
    _filtered_image = filter_image_.clone();
}