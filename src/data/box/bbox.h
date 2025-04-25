//
// Created by Dongmin on 25. 4. 24.
//

#ifndef BBOX_H
#define BBOX_H

class Bbox {
public:
    Bbox() {
        alive_ = false;
        x1_ = 0.0f;
        y1_ = 0.0f;
        x2_ = 0.0f;
        y2_ = 0.0f;
        cx_ = 0.0f;
        cy_ = 0.0f;
        w_ = 0.0f;
        h_ = 0.0f;
        class_id_ = 0;
        score_ = 0.0f;
    };
    ~Bbox() {};
    void New_xywh(float _cx, float _cy, float _w, float _h, int32_t _class_id, float score_) {
        alive_ = true;
        cx_ = std::clamp(_cx, 0.0f, 1.0f);
        cy_ = std::clamp(_cy, 0.0f, 1.0f);
        w_ = std::clamp(_w, 0.0f, 1.0f);
        h_ = std::clamp(_h, 0.0f, 1.0f);
        x1_ = cx_ - w_ / 2;
        x1_ = std::clamp(x1_, 0.0f, 1.0f);
        y1_ = cy_ - h_ / 2;
        y1_ = std::clamp(y1_, 0.0f, 1.0f);
        x2_ = cx_ + w_ / 2;
        x2_ = std::clamp(x2_, 0.0f, 1.0f);
        y2_ = cy_ + h_ / 2;
        y2_ = std::clamp(y2_, 0.0f, 1.0f);
        class_id_ = _class_id;
        score_ = score_;
    };
    void Copy(Bbox _bbox) {
        alive_ = _bbox.alive_;
        x1_ = _bbox.x1_;
        y1_ = _bbox.y1_;
        x2_ = _bbox.x2_;
        y2_ = _bbox.y2_;
        cx_ = _bbox.cx_;
        cy_ = _bbox.cy_;
        w_ = _bbox.w_;
        h_ = _bbox.h_;
        class_id_ = _bbox.class_id_;
        score_ = _bbox.score_;
    };
    void Clear() { alive_ = false; };
    inline float GetArea() { return w_ * h_; };
public:
    bool alive_ = false;
    float x1_ = 0.0f;
    float y1_ = 0.0f;
    float x2_ = 0.0f;
    float y2_ = 0.0f;
    float cx_ = 0.0f;
    float cy_ = 0.0f;
    float w_ = 0.0f;
    float h_ = 0.0f;
    int32_t class_id_ = 0;
    float score_ = 0.0f;
};
#endif //BBOX_H
