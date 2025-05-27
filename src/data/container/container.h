//
// Created by Dongmin on 25. 4. 24.
//

#ifndef CONTAINER_H
#define CONTAINER_H

#include <cstdint>
#include <opencv2/opencv.hpp>
#include "data/shape.h"
#include "data/box/trackbox.h"

template <class T>
class Container {
public:
    Container(int _max_detect_box = 30) {
        bboxes_.resize(_max_detect_box);
        bboxes_.clear();
    };
    ~Container() {
        bboxes_.resize(bboxes_.capacity());
        for(int i =0 ; i < bboxes_.size(); i++) bboxes_[i].Clear();
        bboxes_.clear();
    };
    void Initialize(void* _gpu_data, std::shared_ptr<Shape> _shape, int _mask_width, int _mask_height) {
        gpu_data_ = _gpu_data;
        org_image_ = std::make_shared<cv::Mat>(_shape->height_, _shape->width_, CV_8UC3, cv::Scalar(0,0,0));
        mask_ = cv::Mat(_mask_height, _mask_width, CV_8UC1, cv::Scalar(255));
        menu_ = cv::Mat(_mask_height, _mask_width, CV_8UC3, cv::Scalar(0,0,0));
        filtered_image_ = cv::Mat(_shape->height_, _shape->width_, CV_8UC3, cv::Scalar(0,0,0));
        color_mask_ = cv::Mat(_mask_height, _mask_width, CV_8UC3, cv::Scalar(0,0,0));
        stacked_image_ = cv::Mat(_mask_height, _mask_width, CV_8UC1, cv::Scalar(255));
    };
    void Release() {
        org_image_->release();
        mask_.release();
        filtered_image_.release();
        stacked_image_.release();
    };

public:
    std::shared_ptr<cv::Mat> org_image_;                         // 원본 이미지
    cv::Mat mask_;                              // 마스크 이미지 (0:도우, ..., 255:손)
    cv::Mat menu_;
    cv::Mat color_mask_;
    cv::Mat filtered_image_;                    // 필터링된 이미지 (avg_pixel or add_Weight)
    cv::Mat stacked_image_;                     // 누적 이미지   (spatio_temporal)
    void* gpu_data_;
    std::vector<T> bboxes_;
    std::vector<TrackBox> track_boxes_;
    TrackBox tracker_box_;
    int new_pixel_ = 0;                         // n(현재 마스크 - (현재 마스크 & 도우 마스크))
};
#endif //CONTAINER_H
