//
// Created by Dongmin on 25. 3. 12.
//

#include "dm_engine.h"

using namespace gotrt;

void GoEngine::Initialize()
{
    std::cout << "GoEngine::initialize()" << std::endl;
}

void GoEngine::Release()
{
    std::cout << "GoEngine::release()" << std::endl;
    if (buffer_ != nullptr)
    {
        buffer_->Release();
        buffer_.reset();
    }

    if (context_ != nullptr) {
        context_.release();
        context_.reset();
    }
    if (engine_ != nullptr) {
        engine_.release();
        engine_.reset();
    }
    if (runtime_ != nullptr) {
        runtime_.release();
        runtime_.reset();
    }

}

bool GoEngine::LoadEngine(ModelParams _model_params,
                          std::shared_ptr<Shape> _input_shape,
                          std::shared_ptr<std::vector<Shape>> _output_shape) {
    std::cout << "GoEngine::load_engine()" << std::endl;
    gotrt::Logger trtlogger;
    std::unique_ptr<nvinfer1::IBuilder> builder_ = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trtlogger));
    if (!builder_) return false;

    const unsigned explicitBatch = (int)nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED;
    std::unique_ptr<nvinfer1::INetworkDefinition> network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder_->createNetworkV2(explicitBatch));
    if (!network) return false;

    std::unique_ptr<nvinfer1::IBuilderConfig> config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder_->createBuilderConfig());
    if (!config) return false;

    std::unique_ptr<nvonnxparser::IParser> parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, trtlogger));
    if (!parser) return false;

    auto constructed = ConstructNetwork(builder_, network, config, parser, _model_params);
    if (!constructed) return false;

    // NMS
    {
        auto output_tensor = network->getOutput(0);         // output_tensor [1 x 6 x 3549]
        auto output_dim = output_tensor->getDimensions();
        auto class_num = output_dim.d[1] - 4;              // det : 2, seg : 141
        bool is_segment = false;
        if(class_num > 32) {
            class_num -= 32;
            is_segment = true;
        }

        auto slice = network->addSlice(*output_tensor, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{1, 4, output_dim.d[2]}, nvinfer1::Dims3{1, 1, 1});
        auto org_box_tensor = slice->getOutput(0);         // org_box_tensor [1 x 4 x 3549]
        // width and height normalization
        // width > height
        // x : (0~1), y : (0~160/416), w : (0~1), h : (0~160/416)
        // if height > width
        //      float box_scale_value = (1.0f/float(network->getInput(0)->getDimensions().d[2]));
        float box_scale_value = (1.0f/float(network->getInput(0)->getDimensions().d[3]));
        float box_shift_value = 0.0f;
        float box_power_value = 1.0f;
        auto box_shift = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, &box_shift_value, 1};
        auto box_scale = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, &box_scale_value, 1};
        auto box_power = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, &box_power_value, 1};

        auto box_normalize = network->addScale(*org_box_tensor, nvinfer1::ScaleMode::kUNIFORM, box_shift, box_scale, box_power);
        box_normalize->setScale(box_scale);
        box_normalize->setShift(box_shift);
        box_normalize->setPower(box_power);
        auto norm_box_tensor = box_normalize->getOutput(0);        // norm_box_tensor [1 x 4 x 3549]

        slice = network->addSlice(*output_tensor, nvinfer1::Dims3{0, 4, 0}, nvinfer1::Dims3{1, class_num, output_dim.d[2]}, nvinfer1::Dims3{1, 1, 1});
        auto org_score_tensor = slice->getOutput(0);               // org_score_tensor [1 x 2 x 3549]

        auto score_topk = network->addTopK(*org_score_tensor, nvinfer1::TopKOperation::kMAX, 1, 2);
        auto Bobj_score_tensor = score_topk->getOutput(0);          // Bobj_score_tensor [1 x 1 x 3549]
        auto Bobj_label_tensor = score_topk->getOutput(1);          // Bobj_label_tensor [1 x 1 x 3549]

        nvinfer1::IIdentityLayer* identity = network->addIdentity(*Bobj_label_tensor);
        identity->setOutputType(0, nvinfer1::DataType::kFLOAT);
        Bobj_label_tensor = identity->getOutput(0);

        nvinfer1::ITensor* cat_tensors[] = {norm_box_tensor, Bobj_score_tensor, Bobj_label_tensor};
        auto concat = network->addConcatenation(cat_tensors, 3);
        concat->setAxis(1);
        auto Bobj_tensor = concat->getOutput(0);                    // Bobj_tensor [1 x 6 x 3549]

        auto shuffle = network->addShuffle(*Bobj_score_tensor);
        shuffle->setFirstTranspose(nvinfer1::Permutation{0, 2, 1});
        auto shf_Bobj_score_tensor = shuffle->getOutput(0);              // shf_score_tensor [1 x 3549 x 1]

        auto topk_obj = network->addTopK(*shf_Bobj_score_tensor, nvinfer1::TopKOperation::kMAX, _model_params.max_objects, 2);
        // auto shf_topk_obj_tensor = topk_obj->getOutput(0);              // shf_topk_obj_tensor [1 x 100 x 1]
        auto shf_topk_index_tensor = topk_obj->getOutput(1);            // shf_topk_index_tensor [1 x 100 x 1]

        auto shuffle2 = network->addShuffle(*shf_topk_index_tensor);
        shuffle2->setFirstTranspose(nvinfer1::Permutation{0, 2, 1});
        auto topk_index_tensor = shuffle2->getOutput(0);                 // topk_index_tensor [1 x 1 x 100]

        auto gather = network->addGather(*Bobj_tensor, *topk_index_tensor, 2);
        auto topk_gather_tensor = gather->getOutput(0);                // topk_gather_tensor [1 x 6 x 1 x 1 x 100]

        auto reshape = network->addShuffle(*topk_gather_tensor);
        reshape->setReshapeDimensions(nvinfer1::Dims3{1, -1, _model_params.max_objects});
        auto topk_result_tensor = reshape->getOutput(0);               // topk_index_tensor1 [1 x 6 x 100]
        topk_result_tensor->setName("result_tensor");
        network->markOutput(*topk_result_tensor);

        auto gather2 = network->addGather(*org_score_tensor, *topk_index_tensor, 2);
        auto topk_score_gather_tensor = gather2->getOutput(0);                // topk_score_gather_tensor [1 x 2 x 1 x 1 x 100]

        auto reshape3 = network->addShuffle(*topk_score_gather_tensor);
        reshape3->setReshapeDimensions(nvinfer1::Dims3{1, -1, _model_params.max_objects});
        auto topk_score_reshape_tensor = reshape3->getOutput(0);               // topk_index_tensor1 [1 x 2 x 100]

        auto shuffle4 = network->addShuffle(*topk_score_reshape_tensor);
        shuffle4->setFirstTranspose(nvinfer1::Permutation{0, 2, 1});
        auto score_answer = shuffle4->getOutput(0);
        auto gather3 = network->addGather(*norm_box_tensor, *topk_index_tensor, 2);
        auto topk_box_gather_tensor = gather3->getOutput(0);                // topk_score_gather_tensor [1 x 4 x 1 x 1 x 100]

        auto reshape5 = network->addShuffle(*topk_box_gather_tensor);
        reshape5->setReshapeDimensions(nvinfer1::Dims3{1, -1, _model_params.max_objects});
        auto topk_box_reshape_tensor = reshape5->getOutput(0);

        auto shuffle6 = network->addShuffle(*topk_box_reshape_tensor);
        shuffle6->setFirstTranspose(nvinfer1::Permutation{0, 2, 1});
        auto box_answer = shuffle6->getOutput(0);

        // ======================= nms layer
        int32_t max_box_value = _model_params.max_objects;
        auto max_box_constant = network->addConstant(nvinfer1::Dims{0, 1}, nvinfer1::Weights{nvinfer1::DataType::kINT32, &max_box_value, 1});
        auto max_box_tensor = max_box_constant->getOutput(0);

        float iou_threshold = _model_params.iou_threshold; // 0.45f
        auto iou_constant = network->addConstant(nvinfer1::Dims{0, 1}, nvinfer1::Weights{nvinfer1::DataType::kFLOAT, &iou_threshold, 1});
        auto iou_threshold_tensor = iou_constant->getOutput(0);

        float score_threshold = _model_params.confidence_threshold; // 0.35f;
        auto score_constant = network->addConstant(nvinfer1::Dims{0, 1}, nvinfer1::Weights{nvinfer1::DataType::kFLOAT, &score_threshold, 1});
        auto score_threshold_tensor = score_constant->getOutput(0);

        auto nms_layer = network->addNMS(*box_answer, *score_answer, *max_box_tensor);
        nms_layer->setInput(3, *iou_threshold_tensor);
        nms_layer->setInput(4, *score_threshold_tensor);
        nms_layer->setBoundingBoxFormat(nvinfer1::BoundingBoxFormat::kCENTER_SIZES);
        nms_layer->setTopKBoxLimit(max_box_value);

        auto nms_index_tensor = nms_layer->getOutput(0);
        nms_index_tensor->setName("nms_boxes");
        // nms_index_tensor->setDimensions(nvinfer1::Dims2{max_box_value, 3});
        network->markOutput(*nms_index_tensor);

        auto nms_count_tensor = nms_layer->getOutput(1);
        nms_count_tensor->setName("nms_count");
        network->markOutput(*nms_count_tensor);

        // Segmentation PostProcess
        if(is_segment) {
            slice = network->addSlice(*output_tensor, nvinfer1::Dims3{0, 4 + class_num, 0}, nvinfer1::Dims3{1, 32, output_dim.d[2]}, nvinfer1::Dims3{1, 1, 1});
            auto org_mask_tensor = slice->getOutput(0);             // org_score_tensor [1 x 32 x 8400]

            auto seg_gather = network->addGather(*org_mask_tensor, *topk_index_tensor, 2);
            auto topk_mask_tensor = seg_gather->getOutput(0);                // topk_gather_tensor [1 x 32 x 1 x 1 x 100]

            reshape = network->addShuffle(*topk_mask_tensor);
            reshape->setReshapeDimensions(nvinfer1::Dims2{32, -1});
            auto topk_mask_2d_tensor = reshape->getOutput(0);               // mask_tensor [1 x 32 x 100]

            auto org_proto_tensor = network->getOutput(1);          // org_proto_tensor [1 x 32 x 160 x 160]
            auto proto_dim = org_proto_tensor->getDimensions();
            reshape = network->addShuffle(*org_proto_tensor);
            reshape->setReshapeDimensions(nvinfer1::Dims2{32, -1});
            auto inverse_tensor = reshape->getOutput(0);               // proto_tensor [32 x 25600]

            auto multiply = network->addMatrixMultiply(*topk_mask_2d_tensor, nvinfer1::MatrixOperation::kTRANSPOSE, *inverse_tensor, nvinfer1::MatrixOperation::kNONE);
            auto mask_proto_tensor = multiply->getOutput(0);               // mask_proto_tensor [100 x 25600]

            reshape = network->addShuffle(*mask_proto_tensor);
            reshape->setReshapeDimensions(nvinfer1::Dims3{-1, proto_dim.d[2], proto_dim.d[2]});
            auto mask_tensor = reshape->getOutput(0);               // proto_tensor [100 x 160 x 160]

            shuffle = network->addShuffle(*topk_box_reshape_tensor);
            shuffle->setFirstTranspose(nvinfer1::Permutation{2, 1, 0});
            auto norm_box_tensor3 = shuffle->getOutput(0);                   // norm_box_tensor [100 x 4 x 1]

            slice = network->addSlice(*norm_box_tensor3, nvinfer1::Dims3{0, 0, 0}, nvinfer1::Dims3{_model_params.max_objects, 1, 1}, nvinfer1::Dims3{1, 1, 1});
            auto box_cx = slice->getOutput(0);                 // box_x1 [100 x 1 x 1]
            slice = network->addSlice(*norm_box_tensor3, nvinfer1::Dims3{0, 1, 0}, nvinfer1::Dims3{_model_params.max_objects, 1, 1}, nvinfer1::Dims3{1, 1, 1});
            auto box_cy = slice->getOutput(0);                 // box_y1 [100 x 1 x 1]
            slice = network->addSlice(*norm_box_tensor3, nvinfer1::Dims3{0, 2, 0}, nvinfer1::Dims3{_model_params.max_objects, 1, 1}, nvinfer1::Dims3{1, 1, 1});
            auto box_w = slice->getOutput(0);                 // box_x2 [100 x 1 x 1]
            slice = network->addSlice(*norm_box_tensor3, nvinfer1::Dims3{0, 3, 0}, nvinfer1::Dims3{_model_params.max_objects, 1, 1}, nvinfer1::Dims3{1, 1, 1});
            auto box_h = slice->getOutput(0);                 // box_y2 [100 x 1 x 1]
            auto box_scale_value2 = 0.5f;
            box_scale = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, &box_scale_value2, 1};
            box_normalize = network->addScale(*box_w, nvinfer1::ScaleMode::kUNIFORM, box_shift, box_scale, box_power);
            box_normalize->setScale(box_scale);
            box_w = box_normalize->getOutput(0);                       // norm_box_tensor [100 x 1 x 1]

            box_normalize = network->addScale(*box_h, nvinfer1::ScaleMode::kUNIFORM, box_shift, box_scale, box_power);
            box_normalize->setScale(box_scale);
            box_h = box_normalize->getOutput(0);                       // norm_box_tensor [100 x 1 x 1]

            auto tensor_sum = network->addElementWise(*box_cx, *box_w, nvinfer1::ElementWiseOperation::kSUM);
            auto box_x2 = tensor_sum->getOutput(0);                 // box_x2 [100 x 1 x 1]
            tensor_sum = network->addElementWise(*box_cx, *box_w, nvinfer1::ElementWiseOperation::kSUB);
            auto box_x1 = tensor_sum->getOutput(0);                 // box_x2 [100 x 1 x 1]
            tensor_sum = network->addElementWise(*box_cy, *box_h, nvinfer1::ElementWiseOperation::kSUM);
            auto box_y2 = tensor_sum->getOutput(0);                 // box_x2 [100 x 1 x 1]
            tensor_sum = network->addElementWise(*box_cy, *box_h, nvinfer1::ElementWiseOperation::kSUB);
            auto box_y1 = tensor_sum->getOutput(0);                 // box_x2 [100 x 1 x 1]


            rvalues_.resize(160);
            for(int i = 0; i < 160; i++)
                rvalues_[i] = static_cast<float>(i/160.0f);
            nvinfer1::Weights rangeWeights = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, rvalues_.data(), static_cast<int64_t>(rvalues_.size())};
            rangeWeights.values = rvalues_.data();
            auto range_row = network->addConstant(nvinfer1::Dims3{1, 1, proto_dim.d[2]}, rangeWeights);
            auto range_row_tensor = range_row->getOutput(0);
            auto range_col = network->addConstant(nvinfer1::Dims3{1, proto_dim.d[2], 1}, rangeWeights);
            auto range_col_tensor = range_col->getOutput(0);

            auto ElementWise = network->addElementWise(*range_row_tensor, *box_x1, nvinfer1::ElementWiseOperation::kGREATER);
            auto mask_x1 = ElementWise->getOutput(0);
            ElementWise = network->addElementWise(*range_row_tensor, *box_x2, nvinfer1::ElementWiseOperation::kLESS);
            auto mask_x2 = ElementWise->getOutput(0);
            ElementWise = network->addElementWise(*range_col_tensor, *box_y1, nvinfer1::ElementWiseOperation::kGREATER);
            auto mask_y1 = ElementWise->getOutput(0);
            ElementWise = network->addElementWise(*range_col_tensor, *box_y2, nvinfer1::ElementWiseOperation::kLESS);
            auto mask_y2 = ElementWise->getOutput(0);

            ElementWise = network->addElementWise(*mask_x1, *mask_x2, nvinfer1::ElementWiseOperation::kAND);
            auto mul_x = ElementWise->getOutput(0);
            ElementWise = network->addElementWise(*mask_y1, *mask_y2, nvinfer1::ElementWiseOperation::kAND);
            auto mul_y = ElementWise->getOutput(0);
            ElementWise = network->addElementWise(*mul_x, *mul_y, nvinfer1::ElementWiseOperation::kAND);
            auto filter_tensor_bool = ElementWise->getOutput(0);

            identity = network->addIdentity(*filter_tensor_bool);
            identity->setOutputType(0, nvinfer1::DataType::kFLOAT);
            auto filter_tensor_float = identity->getOutput(0);
            ElementWise = network->addElementWise(*mask_tensor, *filter_tensor_float, nvinfer1::ElementWiseOperation::kPROD);
            auto mask_tensor_float = ElementWise->getOutput(0);

            auto box_shift = nvinfer1::Weights{nvinfer1::DataType::kFLOAT, &box_shift_value, 1};
            auto xor_layer = network->addConstant(nvinfer1::Dims3{1,1, 1}, box_shift);
            auto xor_layer_tensor = xor_layer->getOutput(0);
            ElementWise = network->addElementWise(*mask_tensor_float, *xor_layer_tensor, nvinfer1::ElementWiseOperation::kGREATER);

            identity = network->addIdentity(*ElementWise->getOutput(0));
            identity->setOutputType(0, nvinfer1::DataType::kFLOAT);
            auto mask_tensor_float2 = identity->getOutput(0);

            slice = network->addSlice(*topk_result_tensor, nvinfer1::Dims3{0, 5, 0}, nvinfer1::Dims3{1, 1, _model_params.max_objects}, nvinfer1::Dims3{1, 1, 1});
            auto class_id_tensor = slice->getOutput(0);

            float constantValue = 1.0f;
            nvinfer1::Weights constantWeights{nvinfer1::DataType::kFLOAT, &constantValue, 1};
            // Add the constant to the network
            nvinfer1::IConstantLayer* constantLayer = network->addConstant(nvinfer1::Dims3{1, 1, 1}, constantWeights);
            nvinfer1::ITensor* constantTensor = constantLayer->getOutput(0);

            // Element-wise addition of input tensor and constant tensor
            nvinfer1::IElementWiseLayer* addLayer = network->addElementWise(*class_id_tensor, *constantTensor, nvinfer1::ElementWiseOperation::kSUM);
            nvinfer1::ITensor* class_id_p1_tensor = addLayer->getOutput(0);

            shuffle = network->addShuffle(*class_id_p1_tensor);
            shuffle->setFirstTranspose(nvinfer1::Permutation{2, 1, 0});
            class_id_tensor = shuffle->getOutput(0);
            ElementWise = network->addElementWise(*class_id_tensor, *mask_tensor_float2, nvinfer1::ElementWiseOperation::kPROD);
            auto mask_result_tensor = ElementWise->getOutput(0);
            network->markOutput(*mask_result_tensor);
        }
    }

    auto stream_ = std::make_unique<cudaStream_t>();
    if (cudaStreamCreateWithFlags(stream_.get(), cudaStreamNonBlocking) != cudaSuccess)
    {
        stream_.reset(nullptr);
        return false;
    }
    config->setProfileStream(*stream_);

    auto profile0 = builder_->createOptimizationProfile();
    profile0->setDimensions("nms_boxes", nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims2{0, 3});
    profile0->setDimensions("nms_boxes", nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims2{30, 3});
    profile0->setDimensions("nms_boxes", nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims2{100, 3});
    config->addOptimizationProfile(profile0);

    std::unique_ptr<nvinfer1::IHostMemory> plan{builder_->buildSerializedNetwork(*network, *config)};
    if (!plan)
        return false;

    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(trtlogger));
    if (!runtime_)
        return false;

    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(plan->data(), plan->size()));
    if(!engine_) {
        engine_.reset();
        return false;
    }

    auto input_name = engine_->getIOTensorName(0);

    auto input_dim = engine_->getTensorShape(input_name);
    int io_size = engine_->getNbIOTensors();
    _input_shape->Init(input_dim.d[0], input_dim.d[2], input_dim.d[3] , input_dim.d[1], eDTYPE::FLOAT32);

    //(*_output_dims).push_back(engine_->getBindingDimensions(engine_->getNbBindings() - 1));
    (*_output_shape).resize(io_size - 1);
    for (int i = 1; i < io_size; i++) { // [ 1, 2, 3, 4]
        auto output_name = engine_->getIOTensorName(i);
        auto output_dim = engine_->getTensorShape(output_name);
        auto tensor_type = engine_->getTensorDataType(output_name);
        eDTYPE dtype = eDTYPE::FLOAT32;
        if(tensor_type== nvinfer1::DataType::kINT32) dtype= eDTYPE::INT32;
        else if(tensor_type== nvinfer1::DataType::kHALF) dtype= eDTYPE::FLOAT16;
        else if(tensor_type== nvinfer1::DataType::kINT8) dtype= eDTYPE::INT8;
        else if(tensor_type== nvinfer1::DataType::kBOOL) dtype= eDTYPE::BOOL;
        else if(tensor_type== nvinfer1::DataType::kUINT8) dtype= eDTYPE::UINT8;

        if(output_dim.nbDims == 3)
            (*_output_shape)[i - 1].Init(1, output_dim.d[1], output_dim.d[2], output_dim.d[0], dtype);
        else if(output_dim.nbDims == 4)
            (*_output_shape)[i - 1].Init(output_dim.d[0], output_dim.d[2], output_dim.d[3], output_dim.d[1], dtype);
        else if(output_dim.nbDims == 2)
            (*_output_shape)[i - 1].Init(1, _model_params.max_objects, output_dim.d[1], 1, dtype);
        else if(output_dim.nbDims == 0)
            (*_output_shape)[i - 1].Init(1, 1, 1, 1, dtype);
        //else
            // assert(false);pass;
    }
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if(!context_) return false;

    // .engine file save
    if(_model_params.engine_path.size() > 10) {
        std::ofstream engine_file(_model_params.engine_path, std::ios::binary);
        if(engine_file.is_open()) {
            engine_file.write((char*)plan->data(), plan->size());
            engine_file.close();
        }

    }
    context_->setOptimizationProfileAsync(0, *stream_);
    return true;
}

bool GoEngine::LoadEngine(std::string _engine_path,
                            std::shared_ptr<Shape> _input_shape,
                            std::shared_ptr<std::vector<Shape> > _output_shape) {
    gotrt::Logger trtlogger;
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(trtlogger));
    if (!runtime_)
        return false;

    std::ifstream file(_engine_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open engine file.");
    }
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(fileSize);
    file.read(engineData.data(), fileSize);
    engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(engineData.data(), engineData.size()));
    if(!engine_) {
        engine_.reset();
        return false;
    }

    auto input_name = engine_->getIOTensorName(0);

    auto input_dim = engine_->getTensorShape(input_name);
    int io_size = engine_->getNbIOTensors();
    _input_shape->Init(input_dim.d[0], input_dim.d[2], input_dim.d[3] , input_dim.d[1], eDTYPE::FLOAT32);

    //(*_output_dims).push_back(engine_->getBindingDimensions(engine_->getNbBindings() - 1));
    (*_output_shape).resize(io_size - 1);
    for (int i = 1; i < io_size; i++) { // [ 1, 2, 3, 4]
        auto output_name = engine_->getIOTensorName(i);
        auto output_dim = engine_->getTensorShape(output_name);
        auto tensor_type = engine_->getTensorDataType(output_name);
        eDTYPE dtype = eDTYPE::FLOAT32;
        if(tensor_type== nvinfer1::DataType::kINT32) dtype= eDTYPE::INT32;
        else if(tensor_type== nvinfer1::DataType::kHALF) dtype= eDTYPE::FLOAT16;
        else if(tensor_type== nvinfer1::DataType::kINT8) dtype= eDTYPE::INT8;
        else if(tensor_type== nvinfer1::DataType::kBOOL) dtype= eDTYPE::BOOL;
        else if(tensor_type== nvinfer1::DataType::kUINT8) dtype= eDTYPE::UINT8;

        if(output_dim.nbDims == 3)
            (*_output_shape)[i - 1].Init(1, output_dim.d[1], output_dim.d[2], output_dim.d[0], dtype);
        else if(output_dim.nbDims == 4)
            (*_output_shape)[i - 1].Init(output_dim.d[0], output_dim.d[2], output_dim.d[3], output_dim.d[1], dtype);
        else if(output_dim.nbDims == 2)
            (*_output_shape)[i - 1].Init(1, 100, output_dim.d[1], 1, dtype);
        else if(output_dim.nbDims == 0)
            (*_output_shape)[i - 1].Init(1, 1, 1, 1, dtype);
        //else
        // assert(false);pass;
    }

    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if(!context_) return false;
    // context_->setOptimizationProfileAsync(0, *stream_);
    return true;
}


bool GoEngine::Inference() {
    bool status = context_->executeV2(buffer_->GetDeviceBindings().data());
    return status;
}

float* GoEngine::GetInputGpuData()
{
    return buffer_->GetInputGpuData();
}

Tensor* GoEngine::GetOutputTensor(int _index)
{
    return buffer_->GetOutputTensor(_index);
}

bool GoEngine::ConstructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
                                std::unique_ptr<nvinfer1::INetworkDefinition>& network,
                                std::unique_ptr<nvinfer1::IBuilderConfig>& config,
                                std::unique_ptr<nvonnxparser::IParser>& parser,
                                ModelParams model_params)
{
    auto parsed = parser->parseFromFile(model_params.wight_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kINFO));
    if (!parsed) return false;
    if (model_params.fp16) config->setFlag(nvinfer1::BuilderFlag::kFP16);
    if (model_params.int8) config->setFlag(nvinfer1::BuilderFlag::kINT8);
    return true;
}