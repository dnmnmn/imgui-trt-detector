//
// Created by Dongmin on 25. 3. 12.
//

#include "model.h"

void Model::Release() {
    name_ = "Model";
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
    preprocess_->Release();
    postprocess_->Release();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    engine_->Release();
    container_->Release();

    // if (buffer_ != nullptr)
    //     buffer_->Release();
}
void Model::RunThread() {
    preprocess_->SetCudaStreamCtx(buffer_->stream_);
    fps_timer_.start();
    while(!stop_flag_) {
        Run();
        if(fps_timer_.end() >= debug_time_)
        {
            // std::cout << name_<<"::RunThread() - pass count: " << pass_count_ << std::endl;
            char log[100];
            int pass_fps = pass_count_ / (debug_time_ / 1000);
            std::snprintf(log, 100,"%s::RunThread() - pass count: %d", name_.c_str(), pass_fps);
            DM::Logger::GetInstance().Log(log, LOGLEVEL::DEBUG);
            // std::cout << name_<<"::RunThread() - fail count: " << fail_count_ << std::endl;
            int fail_fps = fail_count_ / (debug_time_ / 1000);
            std::snprintf(log, 100,"%s::RunThread() - fail count: %d", name_.c_str(), fail_fps);
            DM::Logger::GetInstance().Log(log, LOGLEVEL::DEBUG);
            pass_count_ = 0;
            fail_count_ = 0;
            fps_timer_.start();
        }
        this_thread::sleep_for(chrono::milliseconds(10));
    }
}
void Model::Run() {
    uint index = 0;
    if(data_store_->module_index_queues_[input_module_index_].try_pop(index))
    {
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

void Model::Inference() {
    engine_->Inference();
}

void Model::Preprocess() {
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
}

void Model::Postprocess() {
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
}