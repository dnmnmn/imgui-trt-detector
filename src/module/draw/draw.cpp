//
// Created by Dongmin on 25. 4. 28.
//

#include "draw.h"

bool Draw::Initialize() {
    name_ = "Draw";
    DM::Logger::GetInstance().Log(name_ + "::Initialize()", LOGLEVEL::INFO);
    // Config
    JsonObject config_json;
    config_json.load(config_path_);
    debug_time_ = config_json.get_int("GoEngine/Log/DebugTime") * 1000;
    sleep_time_ = config_json.get_int("GoEngine/Log/SleepTime");
    width_ = config_json.get_int("GoEngine/Stream/Width");
    height_ = config_json.get_int("GoEngine/Stream/Height");
    seg_width_ = config_json.get_int("GoEngine/Segment/Width");
    seg_height_ = config_json.get_int("GoEngine/Segment/Height");
    return true;
}

void Draw::Release() {
    DM::Logger::GetInstance().Log(name_ + "::Release()", LOGLEVEL::INFO);
}

void Draw::Run() {
    uint index = 0;
    if (data_store_->module_index_queues_[input_module_index_].try_pop(index)) {
        auto container = data_store_->contaiers_[index];
        auto org_image = container->org_image_;
        auto bboxes = container->track_boxes_;
        // Draw the bounding boxes and masks on the original image
        for (const auto& bbox : bboxes) {
            if (bbox.active_ == false) continue;
            int bx1 = bbox.x1_ * width_;
            int by1 = bbox.y1_ * height_;
            int bx2 = bbox.x2_ * width_;
            int by2 = bbox.y2_ * height_;
            string label = "id:"+ std::to_string(bbox.track_id_);
            label += ", score:" + std::to_string(bbox.score_).substr(0,4);
            cv::rectangle(*container->org_image_,
                cv::Point(bx1, by1), cv::Point(bx2, by2),
                cv::Scalar(0, 255, 0), 2);
            cv::rectangle(*container->org_image_,
                cv::Point(bx1 - 1, by1 - 30), cv::Point(bx2 + 1, by1),
                cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(*container->org_image_,
                label, cv::Point(bx1 + 1, by1 - 3), cv::FONT_HERSHEY_SIMPLEX,
                0.8f, cv::Scalar(255, 255, 255), 2);
        }
        cv::Mat resized_image;
        cv::resize(container->mask_, resized_image, cv::Size(seg_width_, seg_height_));
        container->color_mask_ = colormap_.GetColorMap(resized_image);
        ++pass_count_;
        data_store_->module_index_queues_[output_module_index_].push(index);
    } else {
        ++fail_count_;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_));
    }
}