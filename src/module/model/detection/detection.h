//
// Created by Dongmin on 25. 4. 28.
//

#ifndef DETECTION_H
#define DETECTION_H

#include "module/model/model.h"

class Detection : public Model {
public:
    Detection() {};
    ~Detection() {};
    bool Initialize() override;

    // Process
    void Preprocess() override;
    void Postprocess() override;

private:
    bool LoadEngine(std::string _model_path, JsonObject &_config);

    bool width_margin_;
    int context_width_;
    int context_height_;
    float width_scale_;
    float height_scale_;
};



#endif //DETECTION_H
