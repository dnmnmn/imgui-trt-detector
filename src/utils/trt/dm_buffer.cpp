//
// Created by Dongmin on 25. 3. 12.
//

#include "dm_buffer.h"

#include <utils/Logger/Logger.h>
using namespace gotrt;
void GoBuffer::Initialize(int _batch_size,
                          const std::shared_ptr<Shape> _input_shape,
                          const std::shared_ptr<std::vector<Shape>> _output_shape,
                          int _org_image_height,
                          int _org_image_width,
                          int _class_num) {
    /*
     * mDeviceBindings : best_det2.onnx
     * [0] images - { 1, 3, 640, 640 }
     * [1] output0 - { 1, 1, 6, 8400 }
     * [2] ReShape421 - { 1, 66, 80, 80 }
     * [3] ReShape436 - { 1, 66, 40, 40 }
     * [4] ReShape451 - { 1, 66, 20, 20 }
     *
    * * mDeviceBindings : best_seg.onnx
     * [0] images - { 1, 3, 640, 640 }
     * [1] output0 - { 1, 1, 140, 8400 }
     * [2] ReShape718 - { 1, 168, 80, 80 }
     * [3] ReShape733 - { 1, 168, 40, 40 }
     * [4] ReShape748 - { 1, 168, 20, 20 }
     * [5] Concat_703 - { 1, 1, 32, 8400 }
     * [6] 654 - { 1, 32, 160, 160 }
     */
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
    cudaStreamCreate(&stream_);
    batch_size_ = _batch_size;
    input_tensor.batch = batch_size_;
    input_tensor.channel = _input_shape->channel_;
    input_tensor.height = _input_shape->height_;
    input_tensor.width = _input_shape->width_;
    input_tensor.tensor_size = batch_size_ * input_tensor.channel * input_tensor.height * input_tensor.width * sizeof(float);
    if(_input_shape->dtype==eDTYPE::INT32){
        input_tensor.cpu_tensor = new int32_t[input_tensor.tensor_size];
        input_tensor.type = eDTYPE::INT32;
    }
    else if (_input_shape->dtype==eDTYPE::FLOAT32){
        input_tensor.cpu_tensor = new float[input_tensor.tensor_size];
        input_tensor.type = eDTYPE::FLOAT32;
    }
    else
        assert(false);
    cudaMallocAsync(&input_tensor.gpu_tensor, input_tensor.tensor_size, stream_);
    mDeviceBindings_.push_back(input_tensor.gpu_tensor);

    output_tensor_.resize(_output_shape->size());
    for(int i = 0; i < output_tensor_.size(); i++)
    {
        output_tensor_[i].batch = batch_size_;
        output_tensor_[i].channel = (*_output_shape)[i].channel_;
        output_tensor_[i].height = (*_output_shape)[i].height_;
        output_tensor_[i].width = (*_output_shape)[i].width_;

        output_tensor_[i].tensor_size = batch_size_ * output_tensor_[i].channel * output_tensor_[i].height * output_tensor_[i].width * sizeof(float);
        if(_output_shape->at(i).dtype==eDTYPE::INT32){
            output_tensor_[i].cpu_tensor = new int32_t[output_tensor_[i].tensor_size];
            output_tensor_[i].type = eDTYPE::INT32;
        }
        else if (_output_shape->at(i).dtype==eDTYPE::FLOAT32){
            output_tensor_[i].cpu_tensor = new float[output_tensor_[i].tensor_size];
            output_tensor_[i].type = eDTYPE::FLOAT32;
        }
        else
            assert(false);

        cudaMallocAsync(&output_tensor_[i].gpu_tensor, output_tensor_[i].tensor_size, stream_);
        cudaMemset(output_tensor_[i].gpu_tensor, 0, (size_t)output_tensor_[i].tensor_size);
        mDeviceBindings_.push_back(output_tensor_[i].gpu_tensor);
    }
    // xywhsck tensor
    class_num_ = _class_num;
}

void GoBuffer::Release() {
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
    delete[] input_tensor.cpu_tensor;
    cudaFree(input_tensor.gpu_tensor);
    for(int i = 0; i < output_tensor_.size(); i++)
    {
        delete[] output_tensor_[i].cpu_tensor;
        cudaFree(output_tensor_[i].gpu_tensor);
    }

    cudaStreamDestroy(stream_);

    output_tensor_.clear();
    mDeviceBindings_.clear();
}

float* GoBuffer::GetInputGpuData()
{
    return static_cast<float*>(input_tensor.gpu_tensor);
}

Tensor* GoBuffer::GetOutputTensor(int index)
{
    return &output_tensor_[index];
}

std::vector<void*> GoBuffer::GetDeviceBindings()
{
    return mDeviceBindings_;
}