//
// Created by Dongmin on 25. 4. 28.
//
#include "preprocess.h"
#include <utils/Logger/Logger.h>

void PreProcess::Initialize() {
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
}

void PreProcess::Release() {
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
}

void PreProcess::SetCudaStreamCtx(cudaStream_t _stream) {
    nppSetStream(_stream);
    nppGetStreamContext(&ctx_);
    ctx_.hStream = _stream;
}