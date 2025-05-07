//
// Created by Dongmin on 25. 4. 30.
//

#ifndef REMOVER_H
#define REMOVER_H

#include "module/filter/filter.h"
#include <cuda_runtime_api.h>
#include <nppdefs.h>

class Remover : public Filter {
public:
    Remover() {};
    ~Remover() {};

    bool Initialize() override;
    void Release() override;
    // void Run() override;

protected:
    void AddFrame(std::shared_ptr<cv::Mat> _frame) override;
    void GetImage(cv::Mat &_filtered_image) override;
private:
    void gpu_set_mask();
    void gpu_remover(unsigned char* _in);
    bool first_time_ = true;
    cudaStream_t stream_;
    NppStreamContext ctx_;
    void *gpu_filter_;
};



#endif //REMOVER_H
