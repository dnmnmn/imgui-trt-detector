//
// Created by Dongmin on 25. 3. 12.
//

#include "segmentation.h"

bool Segmentation::Initialize() {
    name_ = "Segmentation";
    DM::Logger::GetInstance().Log(name_ + "::Initialize()", LOGLEVEL::INFO);
    // Config
    JsonObject config_json;
    config_json.load(config_path_);

    max_objects_ = config_json.get_int("Engine/Segment/MaxObjects");
    batch_size_ = config_json.get_int("Engine/Segment/Batch");
    debug_time_ = config_json.get_int("Engine/Log/DebugTime") * 1000;
    int org_image_width = config_json.get_int("Engine/Stream/Width");
    int org_image_height = config_json.get_int("Engine/Stream/Height");
    scale_ = (float)org_image_width / (float)org_image_height;
    // Input Ouput Shape Initialization
    input_shape_ = std::make_shared<Shape>();
    output_shape_ = std::make_shared<std::vector<Shape>>();
    org_image_shape_ = std::make_shared<Shape>(1, org_image_height, org_image_width, 3, eDTYPE::UINT8);


    // Engine Initialization
    engine_ = std::make_shared<dm_trt::Engine>();
    engine_->Initialize();

    // PreProcess PostProcess Initialization
    preprocess_ = std::make_shared<PreProcess>();
    postprocess_ = std::make_shared<PostProcess>();

    // Load model
    std::string segment_weight = config_json.get_string("Engine/Path/SegmentEngine");
    if(FileSystem::exist(segment_weight) == true)
    {
        engine_->LoadEngine(segment_weight, input_shape_, output_shape_);
    }
    else {
        segment_weight = config_json.get_string("Engine/Path/SegmentWeights");
        segment_weight += ".onnx";
        if(FileSystem::exist(segment_weight) == false)
        {
            DM::Logger::GetInstance().Log("Error: Segmentation::Initialize() - Invalid file path : " + segment_weight, LOGLEVEL::ERROR);
            assert(false);
        }
        LoadEngine(segment_weight, config_json);
    }

    buffer_ = std::make_shared<dm_trt::Buffer>();
    buffer_->Initialize(batch_size_, input_shape_, output_shape_, org_image_height, org_image_width, class_num_);
    preprocess_->Initialize();
    postprocess_->Initialize();
    preprocess_->SetCudaStreamCtx(buffer_->stream_);
    engine_->SetBuffer(buffer_);

    return true;
}

void Segmentation::Run() {
    uint index = 0;
    if(data_store_->module_index_queues_[input_module_index_].try_pop(index))
    {
        if (data_store_->tracker_box_.active_==false) {
            data_store_->module_index_queues_[output_module_index_].push(index);
            return;
        }
        tracker_box_ = static_cast<Bbox>(data_store_->tracker_box_);
        container_ = data_store_->contaiers_[index];
        Preprocess();
        Inference();
        Postprocess();
        ++pass_count_;
        data_store_->module_index_queues_[output_module_index_].push(index);
    }
    else {
        ++fail_count_;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}


void Segmentation::Preprocess() {
    preprocess_->gpu_crop_resize((uchar*)container_->gpu_data_,
                                 (uchar*)engine_->GetInputResizeGpuData(),
                                 tracker_box_, org_image_shape_->height_, org_image_shape_->width_, org_image_shape_->channel_, 1,
                                 input_shape_->height_, input_shape_->width_, input_shape_->channel_);
    preprocess_->gpu_nhwc_bgr_to_nchw_rgb((uchar*)engine_->GetInputResizeGpuData(),
                                          (float*)engine_->GetInputGpuData(),
                                          input_shape_->batch_, input_shape_->height_,
                                          input_shape_->width_, input_shape_->channel_,
                                          engine_->GetBufferStream());
}

void Segmentation::Postprocess() {
    // gpu to cpu
    // output_tensor2 = dtype float32
    // output_tensor3 = dtype int32 (박스 인덱스 + 클래스)
    // output_tensor4 = dtype int32 (박스 개수)
    // output_tensor5 = dtype float32 (마스크 tensor)
    auto output_tensor2 = engine_->GetOutputTensor(2);
    auto output_tensor3 = engine_->GetOutputTensor(3);
    auto output_tensor4 = engine_->GetOutputTensor(4);
    auto output_tensor5 = engine_->GetOutputTensor(5);

    cudaMemcpyAsync(output_tensor2->cpu_tensor, output_tensor2->gpu_tensor, output_tensor2->tensor_size, cudaMemcpyDeviceToHost, buffer_->stream_);
    cudaMemcpyAsync(output_tensor3->cpu_tensor, output_tensor3->gpu_tensor, output_tensor3->tensor_size, cudaMemcpyDeviceToHost, buffer_->stream_);
    cudaMemcpyAsync(output_tensor4->cpu_tensor, output_tensor4->gpu_tensor, output_tensor4->tensor_size, cudaMemcpyDeviceToHost, buffer_->stream_);
    cudaMemcpyAsync(output_tensor5->cpu_tensor, output_tensor5->gpu_tensor, output_tensor5->tensor_size, cudaMemcpyDeviceToHost, buffer_->stream_);
    int mask_size = output_shape_->at(5).width_ * output_shape_->at(5).height_;
    int obj_count = ((int*)output_tensor4->cpu_tensor)[0];

    container_->mask_.setTo(cv::Scalar(255));           // 마스크 초기화
    ++data_store_->frame_count_;

    for(int i = 0; i < obj_count; i++) {
        int index = static_cast<int*>(output_tensor3->cpu_tensor)[(obj_count - (i + 1)) * 3 + 2];
        for(int j = 0; j < mask_size; j++){
            // output_tensor5는 0~255 사이의 float32 값을 가짐
            // 해당 값은 정수형으로 변환되어야 함
            float temp = static_cast<float*>(output_tensor5->cpu_tensor)[index * mask_size + j];
            // 마스크 영역 복사 (0=손, 1=도우 -> 0=도우, 1=소스 ...)
            if(temp > 0.5)
            {
                uint8_t temp2 = std::round(temp) - 1;
                container_->mask_.data[j] = temp2;
            }
        }
    }
}

bool Segmentation::LoadEngine(std::string _model_path, JsonObject &_config) {
    DM::Logger::GetInstance().Log("Segmentation::LoadEngine()", LOGLEVEL::INFO);
    dm_trt::ModelParams model_params;
    model_params.weight_path = _model_path;
    model_params.engine_path = _config.get_string("Engine/Path/SegmentEngine");
    model_params.input_index = _config.get_int("Engine/Segment/InputIdx");;
    model_params.fp16 = (bool)_config.get_int("Engine/Segment/FP16");
    model_params.iou_threshold = _config.get_float("Engine/Segment/IoULevel");
    model_params.confidence_threshold = _config.get_float("Engine/Segment/ConfLevel");
    model_params.max_objects = _config.get_int("Engine/Segment/MaxObjects");
    engine_->LoadEngine(model_params, input_shape_, output_shape_);
    class_num_ = (*output_shape_)[0].height_ - 4 - 32;
    return true;
}
