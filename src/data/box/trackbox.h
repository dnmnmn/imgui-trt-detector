//
// Created by Dongmin on 25. 4. 24.
//

#ifndef TRACKBOX_H
#define TRACKBOX_H

#include "bbox.h"

class TrackBox : public Bbox {
public:
    TrackBox() {};
    TrackBox(Bbox &_bbox, int _track_id) : Bbox(_bbox) {track_id_ = _track_id;};
    ~TrackBox() {};
    void Correct(Bbox &_bbox, int _count) {
        cx_ = float(cx_ * _count + _bbox.cx_) / float(_count + 1);
        cy_ = float(cy_ * _count + _bbox.cy_) / float(_count + 1);
        w_ = float(w_ * _count + _bbox.w_) / float(_count + 1);
        h_ = float(h_ * _count + _bbox.h_) / float(_count + 1);
        x1_ = cx_ - w_ * 0.5f;
        y1_ = cy_ - h_ * 0.5f;
        x2_ = cx_ + w_ * 0.5f;
        y2_ = cy_ + h_ * 0.5f;
        score_ = float(score_ * _count + _bbox.score_) / float(_count + 1);
        this->class_id_ = _bbox.class_id_;
        is_matched_ = true;
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
    int track_id_ = -1;
    bool is_matched_=false;
};

#endif //TRACKBOX_H
