//
// Created by gopizza on 25. 4. 24.
//

#include "imgui_trt.h"

bool ImGuiTRT::Initialize() {
    DM::Logger::GetInstance().Log("ImGuiTRT::Init()", LOGLEVEL::INFO);

    // Load the config file
    JsonObject config_json;
    if (FileSystem::exist(CONFIG_PATH)) {
        config_json.load(CONFIG_PATH);
    }
    else {
        DM::Logger::GetInstance().Log("ImGuiTRT::initialize() - Could not open config file ", LOGLEVEL::ERROR);
        return false;
    }

    int use_stream = config_json.get_int("GoEngine/Stream/Use");
    int use_display = 1;
    // Initialize the data store
    pipeline_ = std::make_shared<Pipeline>();
    pipeline_->Initialize();

    int queue_index = 0;

    auto app = std::make_shared<Application>();
    if (!app->Initialize()) {
        DM::Logger::GetInstance().Log("ImGuiTRT::initialize() - App Error", LOGLEVEL::ERROR);
        return false;
    }
    app->SetDataStore(pipeline_);
    app->SetOutputModuleIndex(queue_index);
    modules_.push_back(app);

    if (use_stream > 0) {
        // Initialize the stream
        auto stream = std::make_shared<Stream>();
        if (!stream->Initialize()) {
            DM::Logger::GetInstance().Log("ImGuiTRT::initialize() - Stream Error", LOGLEVEL::ERROR);
            return false;
        }
        stream->SetDataStore(pipeline_);
        stream->SetInputModuleIndex(queue_index);
        stream->SetOutputModuleIndex(++queue_index);
        pipeline_->module_index_queues_.push_back(tbb::concurrent_queue<uint>());
        modules_.push_back(stream);
    } else {
        DM::Logger::GetInstance().Log("ImGuiTRT::initialize() - Stream Error", LOGLEVEL::ERROR);
        return false;
    }

    if (use_display > 0) {
        // Initialize the display
        auto display = std::make_shared<Display>();
        if (!display->Initialize()) {
            DM::Logger::GetInstance().Log("ImGuiTRT::initialize() - Display Error", LOGLEVEL::ERROR);
            return false;
        }
        display->SetDataStore(pipeline_);
        display->SetInputModuleIndex(queue_index);
        display->SetOutputModuleIndex(++queue_index);
        pipeline_->module_index_queues_.push_back(tbb::concurrent_queue<uint>());
        modules_.push_back(display);
    }

    modules_.at(0)->SetInputModuleIndex(queue_index);
    return true;
}

void ImGuiTRT::Release() {
    DM::Logger::GetInstance().Log("ImGuiTRT::Release()", LOGLEVEL::INFO);
    for (auto module : modules_) {
        module->Release();
        module.reset();
    }
    pipeline_->Release();
}

void ImGuiTRT::Run() {
    DM::Logger::GetInstance().Log("ImGuiTRT::Run()", LOGLEVEL::INFO);
    for (auto module : modules_) {
        module->StartThread();
    }
    while(true) {
        if (modules_.front()->stop_flag_ ) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    for (auto module : modules_) {
        module->StopThread();
    }
}