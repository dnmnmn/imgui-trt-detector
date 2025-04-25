//
// Created by Dongmin on 25. 4. 24.
//

#ifndef SHAPE_H
#define SHAPE_H

enum eDTYPE {
    FLOAT32 = 0,
    FLOAT16 = 1,
    INT32 = 2,
    INT8 = 3,
    UINT8 = 4,
    BOOL = 5
};

class Shape {
public:
    Shape() {};
    Shape(int32_t _batch, int32_t _height, int32_t _width, int32_t _channel, eDTYPE _dtype)
            : batch_(_batch), height_(_height), width_(_width), channel_(_channel), dtype(_dtype) {size_ = batch_ * width_ * height_ * channel_;};
    ~Shape() {};
    void Init(int32_t _batch, int32_t _height, int32_t _width, int32_t _channel, eDTYPE _dtype) {
        batch_ = _batch;
        height_ = _height;
        width_ = _width;
        channel_ = _channel;
        size_ = batch_ * width_ * height_ * channel_;
        dtype = _dtype;
    };
    int32_t batch_;
    int32_t width_;
    int32_t height_;
    int32_t channel_;
    int32_t size_;
    eDTYPE dtype;
};

#endif //SHAPE_H
