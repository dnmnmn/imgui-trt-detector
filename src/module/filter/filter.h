//
// Created by Dongmin on 25. 4. 30.
//

#ifndef FILTER_H
#define FILTER_H

#include "module/module.h"

class Filter : public Module {
public:
    Filter() {};
    ~Filter() {};

    bool Initialize() override;
    void Release() override;
    void Run() override;

public:
    virtual void AddFrame(std::shared_ptr<cv::Mat> _frame);
    virtual void GetImage(cv::Mat &_filtered_image);

public:
    int width_, height_;
    cv::Mat filter_image_;
    std::shared_ptr<Container<Bbox>> container_;
};



#endif //FILTER_H
