//
// Created by Dongmin on 25. 3. 12.
//

#ifndef MAS_SEGMENTATION_H
#define MAS_SEGMENTATION_H

#include "module/model/model.h"

class Segmentation : public Model {
public:
    Segmentation() {};
    ~Segmentation() {};
    bool Initialize() override;

    // Process
    void Run() override;
    void Preprocess();
    void Postprocess();

private:
    bool LoadEngine(std::string _model_path, JsonObject &_config);

private:
    Bbox tracker_box_;
    float scale_;
    int b = 0;
};



#endif //MAS_SEGMENTATION_H
