#include "cuda.h"
#include "preprocess.h"
#include <cstdio>
#include "nppi.h"
#include "device_launch_parameters.h"

__global__ void cuda_kernel_convert_uchar_to_norm_float_nchw_with_rgb(unsigned char* in, float* out, int in_size, int height, int width, int channel)
{
    unsigned int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < in_size)
    {
        unsigned int stride = height * width * channel;
        unsigned int b = index / stride;
        unsigned int stride_b = b * stride;
        unsigned int stride_y = width * channel;
        unsigned int y = (index - stride_b) / stride_y;
        unsigned int stride_x = y * stride_y;
        unsigned int x = (index - stride_b - stride_x) / channel;
        unsigned int d = index - stride_b - stride_x - (x * channel);
        //in = NHWC(BGR)
        //out = NCHW(RGB)
        unsigned int tmp_d = (channel - 1) - d;
        unsigned int out_index = (b * channel * height * width) + (tmp_d * height * width) + (y * width) + x;
        out[out_index] = ((float)in[index]) / 255.0f;
    }
}

void PreProcess::gpu_nhwc_bgr_to_nchw_rgb(uchar *src, float *dst, int _batch, int _height, int _width, int _channel, cudaStream_t stream_) {
    dim3 blockDim(16, 16);
    dim3 gridDim((_width * _height * _channel + blockDim.x - 1) / blockDim.x, _batch);
    int tensor_size = _height * _width * _channel;
    cuda_kernel_convert_uchar_to_norm_float_nchw_with_rgb << < gridDim, blockDim, 0, stream_ >> > (
            (unsigned char*)src,
            (float*)dst,
            tensor_size,
            _height, _width, _channel
    );
    cudaStreamSynchronize(stream_);
}

__global__ void cuda_kernel_convert_norm_to_nchw_with_rgb(float* in, float* out, int in_size, int height, int width, int channel)
{
    unsigned int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < in_size)
    {
        unsigned int stride = height * width * channel;
        unsigned int b = index / stride;
        unsigned int stride_b = b * stride;
        unsigned int stride_y = width * channel;
        unsigned int y = (index - stride_b) / stride_y;
        unsigned int stride_x = y * stride_y;
        unsigned int x = (index - stride_b - stride_x) / channel;
        unsigned int d = index - stride_b - stride_x - (x * channel);
        //in = NHWC(BGR)
        //out = NCHW(RGB)
        unsigned int tmp_d = (channel - 1) - d;
        unsigned int out_index = (b * channel * height * width) + (tmp_d * height * width) + (y * width) + x;
        out[out_index] = ((float)in[index]) / 255.0f;
    }
}

void PreProcess::gpu_nhwc_bgr_to_nchw_rgb(float *src, float *dst, int _batch ,int _height, int _width, int _channel, cudaStream_t stream_) {
    dim3 blockDim(16, 16);
    dim3 gridDim((_width * _height + blockDim.x - 1) / blockDim.x, _batch);
    int tensor_size = _height * _width * _channel;
    cuda_kernel_convert_norm_to_nchw_with_rgb << < gridDim, blockDim, 0, stream_ >> > (
            (float*)src,
            (float*)dst,
            tensor_size,
            _height, _width, _channel
    );
}

void PreProcess::gpu_resize(float* src, float* dst,
                            int _org_height, int _org_width, int _org_channel,
                            int _height, int _width, int _channel)
{
    int src_step = _org_width * _org_channel * sizeof(float);
    NppiSize src_size = { _org_width, _org_height };
    NppiRect src_roi = { 0, 0, _org_width, _org_height };
    int dst_step = _width * _channel * sizeof(float);
    NppiSize dst_size = { _width, _height };
    NppiRect dst_roi = { 0, 0, _width, _height };

    nppiResize_32f_C3R(
            //nppiResize_32f_C3R(
            src, src_step, src_size, src_roi,
            dst, dst_step, dst_size, dst_roi,
            NPPI_INTER_CUBIC   // interpolation method[0:NN, 1:Linear, 2:Cubic]
    );
}

void PreProcess::gpu_resize(uchar* src, uchar* dst,
                            int _org_height, int _org_width, int _org_channel,
                            int _height, int _width, int _channel)
{
    int src_step = _org_width * _org_channel * sizeof(uchar);
    NppiSize src_size = { _org_width, _org_height };
    NppiRect src_roi = { 0, 0, _org_width, _org_height };
    int dst_step = _width * _channel * sizeof(uchar);
    NppiSize dst_size = { _width, _height };
    NppiRect dst_roi = { 0, 0, _width, _height };

    nppiResize_8u_C3R_Ctx(
            src, src_step, src_size, src_roi,
            dst, dst_step, dst_size, dst_roi,
            NPPI_INTER_CUBIC, ctx_  // interpolation method[0:NN, 1:Linear, 2:Cubic],

    );
    // nppiResize_8u_C3R(
    //         src, src_step, src_size, src_roi,
    //         dst, dst_step, dst_size, dst_roi,
    //         NPPI_INTER_CUBIC // interpolation method[0:NN, 1:Linear, 2:Cubic],
    //
    // );
}

void PreProcess::gpu_crop_resize_batch(uchar* src, uchar* dst,
                                       std::vector<Bbox> boxes,
                                       int _src_height, int _src_width, int _src_channel,
                                       int _MAX_BATCH, int _dst_height,
                                       int _dst_width, int _dst_channel) {
    int src_step = _src_width * _src_channel * sizeof(uchar);
    NppiSize src_size = { _src_width, _src_height };
    int dst_step = _dst_width * _dst_channel * sizeof(uchar);
    NppiSize dst_size = { _dst_width, _dst_height };
    int min = boxes.size() < _MAX_BATCH ? boxes.size() : _MAX_BATCH;
    for(int i = 0; i < min; i++)
    {
        NppiRect src_roi = { int(boxes[i].x1_ * _src_width),
                             int(boxes[i].y1_ * _src_height),
                             int(boxes[i].w_ * _src_width),
                             int(boxes[i].h_ * _src_height)};
        NppiRect dst_roi = { 0, _dst_height * i, _dst_width, _dst_height * (i + 1) };
        nppiResize_8u_C3R_Ctx(
                src, src_step, src_size, src_roi,
                dst, dst_step, dst_size, dst_roi,
                NPPI_INTER_CUBIC, ctx_   // interpolation method[0:NN, 1:Linear, 2:Cubic]
        );
    }
}

void PreProcess::gpu_crop_resize(uchar* src, uchar* dst,
                                 Bbox box,
                                 int _src_height, int _src_width, int _src_channel,
                                 int _MAX_BATCH, int _dst_height,
                                 int _dst_width, int _dst_channel) {
    int src_step = _src_width * _src_channel * sizeof(uchar);
    NppiSize src_size = { _src_width, _src_height };
    int dst_step = _dst_width * _dst_channel * sizeof(uchar);
    NppiSize dst_size = { _dst_width, _dst_height };
    NppiRect src_roi = { int(box.x1_ * _src_width),
                         int(box.y1_ * _src_height),
                         int(box.w_ * _src_width),
                         int(box.h_ * _src_height)};
    NppiRect dst_roi = { 0, 0, _dst_width, _dst_height };
    nppiResize_8u_C3R_Ctx(
            src, src_step, src_size, src_roi,
            dst, dst_step, dst_size, dst_roi,
            NPPI_INTER_CUBIC, ctx_   // interpolation method[0:NN, 1:Linear, 2:Cubic]
    );
}

void PreProcess::gpu_bitwise_and(uchar *src, uchar *dst, uchar *mask,
                                int _height, int _width, int _channel) {
    nppiAnd_8u_C3R_Ctx(
        src, _width * _channel * sizeof(uchar),
        mask, _width * _channel * sizeof(uchar),
        dst, _width * _channel * sizeof(uchar),
        { _width, _height }, ctx_
        );
}