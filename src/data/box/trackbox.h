//
// Created by Dongmin on 25. 4. 24.
//

#ifndef TRACKBOX_H
#define TRACKBOX_H

#include "bbox.h"

class TrackBox : public Bbox {
public:
    TrackBox() {
        for (int i = 0; i < 2; i++)
        {
            kf_[i].init(4, 2, 0);
            kf_[i].transitionMatrix = (cv::Mat_<float>(4, 4) << 1, 0, 1, 0,
                                            0, 1, 0, 1,
                                            0, 0, 1, 0,
                                            0, 0, 0, 1);
            kf_[i].measurementMatrix = (cv::Mat_<float>(2,4) << 1, 0, 0, 0,
                                            0, 1, 0, 0);
            // cv::setIdentity(kf_[i].measurementMatrix);
            cv::setIdentity(kf_[i].processNoiseCov, cv::Scalar::all(1e-5));
            cv::setIdentity(kf_[i].measurementNoiseCov, cv::Scalar::all(1e-5));
            cv::setIdentity(kf_[i].errorCovPost, cv::Scalar::all(0.1));
            kf_[i].statePre =(cv::Mat_<float>(4, 1) << 0, 0, 0, 0);
            kf_[i].statePost =(cv::Mat_<float>(4, 1) << 0, 0, 0, 0);
        }
    };
    TrackBox(Bbox &_bbox, int _track_id) : Bbox(_bbox) {track_id_ = _track_id;};
    ~TrackBox() {};
    void Clear() {
        kf_[0].statePost.at<float>(0) = 0;
        kf_[0].statePost.at<float>(1) = 0;
        kf_[0].statePost.at<float>(2) = 0;
        kf_[0].statePost.at<float>(3) = 0;
        kf_[1].statePost.at<float>(0) = 0;
        kf_[1].statePost.at<float>(1) = 0;
        kf_[1].statePost.at<float>(2) = 0;
        kf_[1].statePost.at<float>(3) = 0;
        track_id_ = 0;
        alive_ = 0;
        is_matched_ = false;
        active_ = false;
        reliability_ = 0.0f;
    }
    void Copy(Bbox &_bbox) {
        Bbox::Copy(_bbox);
        reliability_ = _bbox.score_;
        is_matched_ = false;
        kf_[0].statePre = (cv::Mat_<float>(4, 1) << cx_, cy_, 0, 0);
        kf_[1].statePre = (cv::Mat_<float>(4, 1) << w_, h_, 0, 0);
        kf_[0].statePost = (cv::Mat_<float>(4, 1) << cx_, cy_, 0, 0);
        kf_[1].statePost = (cv::Mat_<float>(4, 1) << w_, h_, 0, 0);
    };
    void Correct(Bbox &_bbox) {
        kf_[0].correct((cv::Mat_<float>(2,1) << _bbox.cx_, _bbox.cy_));
        kf_[1].correct((cv::Mat_<float>(2,1) << _bbox.w_, _bbox.h_));
        x1_ = _bbox.x1_;
        y1_ = _bbox.y1_;
        x2_ = _bbox.x2_;
        y2_ = _bbox.y2_;
        w_ = _bbox.w_;
        h_ = _bbox.h_;
        cx_ = _bbox.cx_;
        cy_ = _bbox.cy_;
        score_ = _bbox.score_;
        this->class_id_ = _bbox.class_id_;
        alive_ = 0;
        is_matched_ = true;
        reliability_ = reliability_ * (1 - ema_) + score_ * ema_;
        if(reliability_ < reliability_level) active_ = false;
        else active_ = true;
    };
    bool Update() {
        ++alive_;
        // is_matched_ = false;
        if (alive_ > alive_level_) {
            Clear();
            reliability_ = 0.f;
            return false;
        }
        return true;
    }
    void Predict() {
        auto center = kf_[0].predict();
        auto size = kf_[1].predict();
        float w = size.at<float>(0) * 0.5f;
        float h = size.at<float>(1) * 0.5f;
        // float w = w_ * 0.5f, h = h_ * 0.5f;
        if (center.at<float>(0) - w < 0 || center.at<float>(1) - h < 0 || center.at<float>(0) + w > 1 || center.at<float>(1) + h > 1) {
            return;
        }
        x1_ = center.at<float>(0) - w;
        y1_ = center.at<float>(1) - h;
        x2_ = center.at<float>(0) + w;
        y2_ = center.at<float>(1) + h;
        cx_ = center.at<float>(0);
        cy_ = center.at<float>(1);
    };
    float IoU(Bbox &_bbox) {
        float xx1 = std::max(x1_, _bbox.x1_);
        float yy1 = std::max(y1_, _bbox.y1_);
        float xx2 = std::min(x2_, _bbox.x2_);
        float yy2 = std::min(y2_, _bbox.y2_);
        float w = std::max(0.0f, xx2 - xx1);
        float h = std::max(0.0f, yy2 - yy1);
        float inter = w * h;
        float o = inter / (w_ * h_ + _bbox.GetArea() - inter);
        o = std::clamp(o, 0.0f, 1.0f);
        return o;
    };
    float Distance(Bbox &_bbox) {
        float dx = std::abs(cx_ - _bbox.cx_);
        float dy = std::abs(cy_ - _bbox.cy_);
        float weight = (dx + dy) / 2.0f;
        weight = std::pow(weight, 0.4);
        return weight;
    };
public:
    cv::KalmanFilter kf_[2];
    int track_id_ = -1;
    bool active_ = false;
    bool is_matched_ = false;
    float reliability_ = 0.0f;
    const float reliability_level = 0.5f;
    const float ema_ = 0.3f;
    const float alive_level_ = 60.0f;
};

#endif //TRACKBOX_H
