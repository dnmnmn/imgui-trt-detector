//
// Created by Dongmin on 25. 4. 24.
//

#ifndef IMGUI_TRT_H
#define IMGUI_TRT_H

#include "data/pipeline.h"
#include "module/module.h"
#include "module/stream/stream.h"
#include "module/display/display.h"
#include "application/application.h"
#include "module/model/model.h"
#include "module/model/segmentation/segmentation.h"
#include "module/model/detection/detection.h"
#include "module/draw/draw.h"
#include "module/tracker/tracker.h"
#include "module/filter/filter.h"
#include "module/filter/remover/remover.h"

class ImGuiTRT {
public:
    ImGuiTRT() {};
    ~ImGuiTRT() {};
    bool Initialize();
    void Release();
    void Run();
public:
    std::vector<std::shared_ptr<Module>> modules_;
    std::shared_ptr<Pipeline> pipeline_;
};

#endif //IMGUI_TRT_H
