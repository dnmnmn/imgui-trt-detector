//
// Created by gopizza on 25. 4. 30.
//

#include "remover.h"

bool Remover::Initialize() {
    name_ = "Remover";
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
    JsonObject config_json;
    config_json.load(config_path_);
    debug_time_ = config_json.get_int("GoEngine/Log/DebugTime") * 1000;
    width_ = config_json.get_int("GoEngine/Stream/Width");
    height_ = config_json.get_int("GoEngine/Stream/Height");
    filter_image_ = cv::Mat(height_, width_, CV_32FC3, cv::Scalar(0, 0, 0));
    return true;
}

void Remover::AddFrame(std::shared_ptr<cv::Mat> _frame) {
    if (first_time_) {
        cv::Mat frame;
        frame = container_->org_image_->clone();
        frame.convertTo(filter_image_, CV_32FC3, 1.0 / 255.0);
        first_time_ = false;
        return;
    }
    cv::Mat image = container_->org_image_->clone();
    cv::Mat mask( height_, width_, CV_8UC1, cv::Scalar(255));
    for (auto box : container_->bboxes_) {
        int bx1 = box.x1_ * width_;
        int by1 = box.y1_ * height_;
        int bx2 = box.x2_ * width_;
        int by2 = box.y2_ * height_;
        cv::rectangle(mask,
            cv::Point(bx1, by1), cv::Point(bx2, by2),
            cv::Scalar(0), cv::FILLED);
    }
    filter_image_ *= 0.99f;
    cv::Mat float_frame;
    image.convertTo(float_frame, CV_32FC3, 1.0 / 25500.0);
    cv::add(filter_image_, float_frame, filter_image_, mask);
}

void Remover::GetImage(cv::Mat &_filtered_image) {
    filter_image_.convertTo(_filtered_image, CV_8UC3, 255.0);
    cv::imshow("Filtered", _filtered_image);
    cv::waitKey(1);
}