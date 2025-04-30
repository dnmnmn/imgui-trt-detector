//
// Created by Dongmin on 25. 3. 12.
//

#ifndef MAS_MODEL_H
#define MAS_MODEL_H

#include "module/module.h"
#include "utils/trt/dm_buffer.h"
#include "utils/trt/dm_engine.h"
#include "preprocess/preprocess.h"
#include "postprocess/postprocess.h"

class Model : public Module {
public:
    Model() {};
    ~Model() {};
    void Release() override;
    void Run() override;
    void RunThread() override;

    // Process
    virtual void Preprocess();
    void Inference();
    virtual void Postprocess();

protected:
    int batch_size_;
    int class_num_;
    int max_objects_;
    std::shared_ptr<Shape> input_shape_;
    std::shared_ptr<std::vector<Shape>> output_shape_;
    std::shared_ptr<Shape> org_image_shape_;

    std::shared_ptr<PreProcess> preprocess_;
    std::shared_ptr<PostProcess> postprocess_;
    std::shared_ptr<dm_trt::Engine> engine_;
    std::shared_ptr<dm_trt::Buffer> buffer_;
    std::shared_ptr<Container<Bbox>> container_;
};



#endif //MAS_MODEL_H
