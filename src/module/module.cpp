//
// Created by Dongmin on 25. 4. 24.
//

#include "module.h"

bool Module::Initialize() {
    DM::Logger::GetInstance().Log(name_ + "::Initialize()", LOGLEVEL::INFO);
    return false;
};

void Module::Release() {
    DM::Logger::GetInstance().Log(name_ + "::Release()", LOGLEVEL::INFO);
};

void Module::Run() {
    DM::Logger::GetInstance().Log(name_ + "::Run()", LOGLEVEL::INFO);
};

// Thread
void Module::StartThread() {
    DM::Logger::GetInstance().Log(name_ + "::StartThread()", LOGLEVEL::INFO);
    stop_flag_.store(false);
    thread_ = std::thread(&Module::RunThread, this);
};

void Module::RunThread() {
    fps_timer_.start();
    while(!stop_flag_) {
        Run();
        if(fps_timer_.end() >= debug_time_)
        {
            char log[100];
            std::snprintf(log, 100,"::RunThread() - pass count: %d", pass_count_/60);
            DM::Logger::GetInstance().Log(name_+log, LOGLEVEL::DEBUG);
            std::snprintf(log, 100, "::RunThread() - fail count: %d", fail_count_/60);
            DM::Logger::GetInstance().Log(name_+log, LOGLEVEL::DEBUG);
            pass_count_ = 0;
            fail_count_ = 0;
            fps_timer_.start();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_));
    }
};

void Module::StopThread() {
    DM::Logger::GetInstance().Log(name_ + "::StopThread()", LOGLEVEL::INFO);
    stop_flag_.store(true);
    if(thread_.joinable())
        thread_.join();
};