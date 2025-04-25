//
// Created by Dongmin on 25. 4. 24.
//

#ifndef PIPELINE_H
#define PIPELINE_H

#include <opencv2/opencv.hpp>
#include <tbb/concurrent_queue.h>
#include <memory>

#include "box/bbox.h"
#include "box/trackbox.h"
#include "container/container.h"
#include "shape.h"
#include "utils/Logger/Logger.h"
#include "utils/Json/JsonObject.h"
#include "utils/FileSystem/FileSystem.h"

class Pipeline {
public:
    Pipeline() {};
    ~Pipeline() {};

    bool Initialize();
    void Release();

public:
     // module pipeline
     std::vector<tbb::concurrent_queue<uint>> module_index_queues_;

     // containers memory
    std::vector<std::shared_ptr<Container<Bbox>>> contaiers_;

     // Shape
     std::shared_ptr<Shape> org_image_shape_;

     // gpu memory
     void* gpu_org_image_buffer_;

     // tracker box
     std::vector<TrackBox> tracker_boxes_;

     // mask
     int frame_count_;
     std::vector<float> mask_count_;
};


#endif //PIPELINE_H
