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
};



#endif //DETECTION_H
