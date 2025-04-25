//
// Created by Dongmin on 25. 3. 12.
//

#ifndef GO_BUFFER_H
#define GO_BUFFER_H

#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include "data/shape.h"

namespace gotrt {
    struct Tensor {
        void* cpu_tensor;
        void* gpu_tensor;
        int tensor_size;
        int batch;
        int channel;
        int width;
        int height;
        eDTYPE type;
    };

    class GoBuffer {
    public:
        GoBuffer() {};
        ~GoBuffer() {};
        void Initialize(int _batch_size,
                        const std::shared_ptr<Shape> _input_shape,
                        const std::shared_ptr<std::vector<Shape>> _output_shape,
                        int _org_image_height, int _org_image_width,
                        int _class_num=1);
        void Release();

        // IO
        float *GetInputGpuData();
        Tensor *GetOutputTensor(int _index);
        std::vector<void *> GetDeviceBindings();

    public:
        std::vector<Tensor> output_tensor_;
        cudaStream_t stream_;
    private:
        int batch_size_;
        int class_num_;
        Tensor input_tensor;
        std::vector<void*> mDeviceBindings_;
    };
}


#endif //GO_BUFFER_H
