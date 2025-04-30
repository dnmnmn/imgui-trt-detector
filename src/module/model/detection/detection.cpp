//
// Created by gopizza on 25. 4. 28.
//

#include "detection.h"

bool Detection::Initialize() {
    name_ = "Detection";
    DM::Logger::GetInstance().Log(name_ + "::Initialize()", LOGLEVEL::INFO);
    // Config
    JsonObject config_json;
    config_json.load(config_path_);

    max_objects_ = config_json.get_int("GoEngine/Detect/MaxObjects");
    batch_size_ = config_json.get_int("GoEngine/Detect/Batch");
    debug_time_ = config_json.get_int("GoEngine/Log/DebugTime") * 1000;
    int org_image_width = config_json.get_int("GoEngine/Stream/Width");
    int org_image_height = config_json.get_int("GoEngine/Stream/Height");

    // Input Ouput Shape Initialization
    input_shape_ = std::make_shared<Shape>();
    output_shape_ = std::make_shared<std::vector<Shape>>();
    org_image_shape_ = std::make_shared<Shape>(1, org_image_height, org_image_width, 3, eDTYPE::UINT8);

    // Engine Initialization
    engine_ = std::make_shared<dm_trt::Engine>();
    engine_->Initialize();



    // Load model
    std::string detect_weight = config_json.get_string("GoEngine/Path/DetectEngine");
    if(FileSystem::exist(detect_weight) == true)
    {
        engine_->LoadEngine(detect_weight, input_shape_, output_shape_);
    }
    else {
        detect_weight = config_json.get_string("GoEngine/Path/DetectWeights");
        detect_weight += ".onnx";
        if(FileSystem::exist(detect_weight) == false)
        {
            DM::Logger::GetInstance().Log("Error: Detection::Initialize() - Invalid file path : " + detect_weight, LOGLEVEL::ERROR);
            assert(false);
        }
        LoadEngine(detect_weight, config_json);
    }
    class_num_ = (*output_shape_)[0].height_ - 4;

    // PreProcess PostProcess Initialization
    preprocess_ = std::make_shared<PreProcess>();
    preprocess_->Initialize();
    postprocess_ = std::make_shared<PostProcess>();
    postprocess_->Initialize();

    buffer_ = std::make_shared<dm_trt::Buffer>();
    buffer_->Initialize(batch_size_, input_shape_, output_shape_, org_image_height, org_image_width, class_num_);

    engine_->SetBuffer(buffer_);

    return true;
}

void Detection::Preprocess() {
    preprocess_->gpu_resize((uchar*)container_->gpu_data_,
                            (uchar*)engine_->GetInputResizeGpuData(),
                            org_image_shape_->height_, org_image_shape_->width_, org_image_shape_->channel_,
                            input_shape_->height_, input_shape_->width_, input_shape_->channel_);
    preprocess_->gpu_nhwc_bgr_to_nchw_rgb((uchar*)engine_->GetInputResizeGpuData(),
                                          (float*)engine_->GetInputGpuData(),
                                          input_shape_->batch_, input_shape_->height_,
                                          input_shape_->width_, input_shape_->channel_,
                                          engine_->GetBufferStream());
}

void Detection::Postprocess() {
    auto output_tensor2 = engine_->GetOutputTensor(1);
    auto output_tensor3 = engine_->GetOutputTensor(2);
    auto output_tensor4 = engine_->GetOutputTensor(3);


    cudaMemcpyAsync(output_tensor2->cpu_tensor, output_tensor2->gpu_tensor, output_tensor2->tensor_size, cudaMemcpyDeviceToHost, buffer_->stream_);
    cudaMemcpyAsync(output_tensor3->cpu_tensor, output_tensor3->gpu_tensor, output_tensor3->tensor_size, cudaMemcpyDeviceToHost, buffer_->stream_);
    cudaMemcpyAsync(output_tensor4->cpu_tensor, output_tensor4->gpu_tensor, output_tensor4->tensor_size, cudaMemcpyDeviceToHost, buffer_->stream_);
    int obj_count = ((int*)output_tensor4->cpu_tensor)[0];
    obj_count = obj_count > MAX_DETECT_BOX ? MAX_DETECT_BOX : obj_count;
    container_->bboxes_.resize(obj_count);
    float scale = (float)input_shape_->width_ / (float)input_shape_->height_;
    for(int i = 0; i < obj_count; i++) {
        int index = ((int*)output_tensor3->cpu_tensor)[i * 3 + 2];
        int class_id = ((int*)output_tensor3->cpu_tensor)[i * 3 + 1];
        float score = ((float*)output_tensor2->cpu_tensor)[index + 4 * max_objects_];
        float cx = ((float*)output_tensor2->cpu_tensor)[index];
        cx = std::clamp(cx, 0.0f, 1.0f);
        float cy = ((float*)output_tensor2->cpu_tensor)[index + 1 * max_objects_] * scale;
        cy = std::clamp(cy, 0.0f, 1.0f);
        float w = ((float*)output_tensor2->cpu_tensor)[index + 2 * max_objects_];
        w = std::clamp(w, 0.0f, 1.0f);
        float h = ((float*)output_tensor2->cpu_tensor)[index + 3 * max_objects_] * scale;
        h = std::clamp(h, 0.0f, 1.0f);
        container_->bboxes_[i].NewXywh(
            cx, cy, w, h,
            class_id,
            score
            );
    }
}

bool Detection::LoadEngine(std::string _model_path, JsonObject &_config) {
    // std::cout << "yolo_v8_det::LoadEngine()" << std::endl;
    DM::Logger::GetInstance().Log("Yolov8Det::LoadEngine()", LOGLEVEL::INFO);
    dm_trt::ModelParams model_params;
    model_params.weight_path = _model_path;
    model_params.engine_path = _config.get_string("GoEngine/Path/DetectEngine");
    model_params.input_index = _config.get_int("GoEngine/Detect/InputIdx");
    model_params.fp16 = (bool)_config.get_int("GoEngine/Detect/FP16");
    model_params.iou_threshold = _config.get_float("GoEngine/Detect/IoULevel");
    model_params.confidence_threshold = _config.get_float("GoEngine/Detect/ConfLevel");
    model_params.max_objects = _config.get_int("GoEngine/Detect/MaxObjects");
    engine_->LoadEngine(model_params, input_shape_, output_shape_);
    return true;
}
