//
// Created by Dongmin on 25. 4. 30.
//

#ifndef REMOVER_H
#define REMOVER_H

#include "module/filter/filter.h"

class Remover : public Filter {
public:
    Remover() {};
    ~Remover() {};

    bool Initialize() override;
    // void Run() override;

protected:
    void AddFrame(std::shared_ptr<cv::Mat> _frame) override;
    void GetImage(cv::Mat &_filtered_image) override;
    bool first_time_ = true;
};



#endif //REMOVER_H
