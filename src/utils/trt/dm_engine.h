//
// Created by Dongmin on 25. 3. 12.
//

#ifndef TRT_ENGINE_H
#define TRT_ENGINE_H

#include <fstream>
#include "dm_buffer.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"

namespace dm_trt
{
    struct ModelParams
    {
        int32_t batch_size{1};              //!< Number of inputs in a batch
        int32_t dla_core{-1};               //!< Specify the DLA core to run network on.
        int32_t max_objects{100};           //!< Maximum number of objects in a frame
        float iou_threshold{0.45f};          //!< IOU threshold for non-max suppression
        float confidence_threshold{0.35f};  //!< Confidence threshold for detection
        bool int8{false};                  //!< Allow runnning the network in Int8 mode.
        bool fp16{false};                  //!< Allow running the network in FP16 mode.
        std::string weight_path; //!< Directory paths where sample data files are stored
        std::string engine_path; //!< Directory paths where sample data files are stored
        int32_t input_index{0};             //!< Input index of the network
        std::vector<int32_t> output_index_vector; //!< Output index of the network
    };

    class Engine {
    public:
        Engine() {};
        ~Engine() {};
        void Initialize();
        void Release();
        bool Inference();

        bool LoadEngine(ModelParams _model_params,
                            std::shared_ptr<Shape> _input_shape,
                            std::shared_ptr<std::vector<Shape>> _output_shape);
        bool LoadEngine(std::string _engine_path,
                        std::shared_ptr<Shape> _input_shape,
                        std::shared_ptr<std::vector<Shape>> _output_shape);

    public:
        void SetBuffer(std::shared_ptr<dm_trt::Buffer> _buffer) { buffer_ = _buffer; }
        cudaStream_t GetBufferStream() { return buffer_->stream_; }

    public:
        float* GetInputGpuData();
        void* GetInputResizeGpuData();
        Tensor* GetOutputTensor(int _index);
        Tensor* GetInputTensor() { return buffer_->GetInputTensor(); }
        Tensor* GetInputResizeTensor() { return buffer_->GetInputResizeTensor(); }
    private:
        bool ConstructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
                         std::unique_ptr<nvinfer1::INetworkDefinition>& network,
                         std::unique_ptr<nvinfer1::IBuilderConfig>& config,
                         std::unique_ptr<nvonnxparser::IParser>& parser,
                         ModelParams model_params);

    private:
        std::unique_ptr<nvinfer1::IExecutionContext> context_;
        std::unique_ptr<nvinfer1::IRuntime> runtime_;
        std::unique_ptr<nvinfer1::ICudaEngine> engine_;
        std::shared_ptr<dm_trt::Buffer> buffer_;
        std::vector<float> rvalues_;
    };
}



#endif //TRT_ENGINE_H
