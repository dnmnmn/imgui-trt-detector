//
// Created by Dongmin on 25. 4. 24.
//

#include "pipeline.h"
#include <cuda_runtime_api.h>

bool Pipeline::Initialize() {
    DM::Logger::GetInstance().Log("Pipeline::Initialize()", LOGLEVEL::INFO);
    JsonObject config_json;
    config_json.load(CONFIG_PATH);

    int MAX_OBJECT = config_json.get_int("GoEngine/Utils/MaxBuffer");
    int mask_width = config_json.get_int("GoEngine/Segment/MaskWidth");
    int mask_height = config_json.get_int("GoEngine/Segment/MaskHeight");
    int _batch_size = config_json.get_int("GoEngine/Segment/Batch");
    int _org_height = config_json.get_int("GoEngine/Stream/Height");
    int _org_width = config_json.get_int("GoEngine/Stream/Width");
    int _org_channel = 3;
    int _type = CV_8UC3;

    // Set the original image shape
    int value_size = 0;
    if(_type == CV_8UC3) value_size = sizeof(uchar);
    else if(_type == CV_32FC3) value_size = sizeof(float);
    else std::cout << "DataStore::Ready() : Invalid image type" << std::endl;
    org_image_shape_ = std::make_shared<Shape>(_batch_size, _org_height, _org_width, _org_channel, eDTYPE::UINT8);

    // Set the GPU image memory
    cudaMalloc(&gpu_org_image_buffer_, org_image_shape_->size_ * value_size * MAX_OBJECT);
    cudaMemset(gpu_org_image_buffer_, 0, (size_t)org_image_shape_->size_ * value_size * MAX_OBJECT);

    contaiers_.resize(MAX_OBJECT);
    module_index_queues_.push_back(tbb::concurrent_queue<uint>());
    for(int i =0 ; i < MAX_OBJECT; i++) {
        contaiers_[i] = std::make_shared<Container<Bbox>>(100);
        contaiers_[i]->Initialize((uchar*)gpu_org_image_buffer_ + org_image_shape_->size_ * i,
                                    org_image_shape_, mask_width, mask_height);
        module_index_queues_[0].push((uint)i);
    }
    frame_count_ = 1;
    mask_count_.resize(mask_width * mask_height, 0.0f);
    return true;
}

void Pipeline::Release() {
    DM::Logger::GetInstance().Log("DataStore::Release()", LOGLEVEL::INFO);
    for(auto obj : contaiers_) {
        obj->Release();
        obj.reset();
    }
    cudaFree(gpu_org_image_buffer_);
}