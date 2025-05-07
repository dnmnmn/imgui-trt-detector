#include "remover.h"

#include <nppi_data_exchange_and_initialization.h>

__global__ void cuda_kernel_pixel_avg(unsigned char* _in, unsigned char* _out, int _in_size)
{
    unsigned int index = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < _in_size)
    {
        float avg = (float)_out[index] * 0.99 + (float)_in[index] * 0.01;
        _out[index] = _in[index] == 0 ? _out[index] : (uchar)avg;
        // _out[index] = _in[index];
    }
}

void Remover::gpu_remover(unsigned char* _in)
{
    int channels = 3;
    int batch = 1;
    dim3 blockDim(16, 16);
    dim3 gridDim((width_ * height_ * channels + blockDim.x - 1) / blockDim.x, batch);
    int tensor_size = height_ * width_ * channels;
    cuda_kernel_pixel_avg << < gridDim, blockDim, 0, stream_ >> > (
        _in, (unsigned char*)gpu_filter_,
        tensor_size
        );

}

void Remover::gpu_set_mask()
{
    int count = container_->bboxes_.size();
    int nSrcStep = width_ * 3;
    Npp8u nVal[3] = {0,0,0};
    for (auto box : container_->bboxes_)
    {
        int w = box.w_ * width_;
        int h = box.h_ * height_;
        int x = box.x1_ * width_;
        int y = box.y1_ * height_;
        NppiSize roi = {w, h};
        // NppiRect roi = {(int)box.x1_ * 3, (int)box.y1_, (int)box.w_ * 3,(int) box.h_};

        Npp8u* pRoiStart = (unsigned char*)container_->gpu_data_ + y * nSrcStep + x * 3;
        nppiSet_8u_C3R_Ctx(nVal, pRoiStart, nSrcStep, roi, ctx_);
    }
}
