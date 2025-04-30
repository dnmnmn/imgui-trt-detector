//
// Created by Dongmin on 25. 4. 24.
//

#include "imgui_trt.h"

bool ImGuiTRT::Initialize() {
    DM::Logger::GetInstance().Initialize();

    // Load the config file
    JsonObject config_json;
    if (FileSystem::exist(CONFIG_PATH)) {
        config_json.load(CONFIG_PATH);
    }
    else {
        DM::Logger::GetInstance().Log("ImGuiTRT::initialize() - Could not open config file ", LOGLEVEL::ERROR);
        return false;
    }

    int use_stream = config_json.get_int("Engine/Stream/Use");
    int use_detect = config_json.get_int("Engine/Detect/Use");
    int use_segment = config_json.get_int("Engine/Segment/Use");
    int use_tracker = config_json.get_int("Engine/Tracker/Use");
    int use_draw = config_json.get_int("Engine/Draw/Use");
    int use_filter = config_json.get_int("Engine/Filter/Use");
    int use_display = 0;
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
    if (use_detect) {
        // Initialize the detection
        auto detect = std::make_shared<Detection>();
        if (!detect->Initialize()) {
            DM::Logger::GetInstance().Log("ImGuiTRT::initialize() - Detection Error", LOGLEVEL::ERROR);
            return false;
        }
        detect->SetDataStore(pipeline_);
        detect->SetInputModuleIndex(queue_index);
        detect->SetOutputModuleIndex(++queue_index);
        pipeline_->module_index_queues_.push_back(tbb::concurrent_queue<uint>());
        modules_.push_back(detect);
    }
    if (use_tracker > 0) {
        // Initialize the tracker
        auto tracker = std::make_shared<Tracker>();
        if (!tracker->Initialize()) {
            DM::Logger::GetInstance().Log("ImGuiTRT::initialize() - Tracker Error", LOGLEVEL::ERROR);
            return false;
        }
        tracker->SetDataStore(pipeline_);
        tracker->SetInputModuleIndex(queue_index);
        tracker->SetOutputModuleIndex(++queue_index);
        pipeline_->module_index_queues_.push_back(tbb::concurrent_queue<uint>());
        modules_.push_back(tracker);
    }
    if (use_segment) {
        // Initialize the segmentation
        auto seg = make_shared<Segmentation>();
        if (!seg->Initialize()) {
            DM::Logger::GetInstance().Log("ImGuiTRT::initialize() - Segment Error", LOGLEVEL::ERROR);
            return false;
        }
        seg->SetDataStore(pipeline_);
        seg->SetInputModuleIndex(queue_index);
        seg->SetOutputModuleIndex(++queue_index);
        pipeline_->module_index_queues_.push_back(tbb::concurrent_queue<uint>());
        modules_.push_back(seg);
    }
    if (use_filter > 0) {
        // Initialize the filter
        auto filter = std::make_shared<Remover>();
        if (!filter->Initialize()) {
            DM::Logger::GetInstance().Log("ImGuiTRT::initialize() - Filter Error", LOGLEVEL::ERROR);
            return false;
        }
        filter->SetDataStore(pipeline_);
        filter->SetInputModuleIndex(queue_index);
        filter->SetOutputModuleIndex(++queue_index);
        pipeline_->module_index_queues_.push_back(tbb::concurrent_queue<uint>());
        modules_.push_back(filter);
    }
    if (use_draw > 0) {
        // Initialize the draw
        auto draw = std::make_shared<Draw>();
        if (!draw->Initialize()) {
            DM::Logger::GetInstance().Log("ImGuiTRT::initialize() - Draw Error", LOGLEVEL::ERROR);
            return false;
        }
        draw->SetDataStore(pipeline_);
        draw->SetInputModuleIndex(queue_index);
        draw->SetOutputModuleIndex(++queue_index);
        pipeline_->module_index_queues_.push_back(tbb::concurrent_queue<uint>());
        modules_.push_back(draw);
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
    pipeline_.reset();
    modules_.clear();
    DM::Logger::GetInstance().Release();
}

void ImGuiTRT::Run() {
    DM::Logger::GetInstance().Log("ImGuiTRT::Run()", LOGLEVEL::INFO);
    for (auto module : modules_) {
        module->StartThread();
    }
    while(true) {
        if (modules_.front()->stop_flag_ ) break;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    for (auto module : modules_) {
        module->StopThread();
    }
}