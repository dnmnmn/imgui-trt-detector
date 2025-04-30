//
// Created by Dongmin on 25. 3. 12.
//

#ifndef MAS_PREPROCESS_H
#define MAS_PREPROCESS_H

#include <vector>
#include "data/shape.h"
#include "data/box/bbox.h"
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <npp.h>

class PreProcess {
public:
    PreProcess() {};
    ~PreProcess() {};
    void Initialize();
    void Release();

    // GPU
    void SetCudaStreamCtx(cudaStream_t _stream);
    void gpu_resize(float* src, float* dst,
                    int _org_height, int _org_width, int _org_channel,
                    int _height, int _width, int _channel);
    void gpu_resize(uchar* src, uchar* dst,
                    int _org_height, int _org_width, int _org_channel,
                    int _height, int _width, int _channel);
    void gpu_nhwc_bgr_to_nchw_rgb(uchar* src, float* dst,
                                  int _batch, int _height,
                                  int _width, int _channel, cudaStream_t _stream);
    void gpu_nhwc_bgr_to_nchw_rgb(float* src, float* dst,
                                  int _batch, int _height,
                                  int _width, int _channel, cudaStream_t _stream);
    void gpu_crop_resize_batch(uchar* src, uchar* dst,
                               std::vector<Bbox> boxes,
                               int _src_height, int _src_width, int _src_channel,
                               int _MAX_BATCH, int _dst_height,
                               int _dst_width, int _dst_channel);
    void gpu_crop_resize(uchar* src, uchar* dst,
                         Bbox box,
                         int _src_height, int _src_width, int _src_channel,
                         int _MAX_BATCH, int _dst_height,
                         int _dst_width, int _dst_channel);
    void gpu_bitwise_and(uchar* src, uchar* dst, uchar* mask,
                        int _height, int _width, int _channel);

protected:
    NppStreamContext ctx_;
};

#endif //MAS_PREPROCESS_H
