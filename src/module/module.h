//
// Created by Dongmin on 25. 4. 24.
//

#ifndef MODULE_H
#define MODULE_H

#include <thread>
#include "data/pipeline.h"
#include "utils/Timer/Timer.h"
#include "utils/Logger/Logger.h"

class Module {
public:
    Module() {};
    ~Module() {};
    virtual bool Initialize();
    virtual void Release();
    virtual void Run();

    // Thread
    virtual void StartThread();
    virtual void RunThread();
    virtual void StopThread();

    //Pipeline
    virtual void SetDataStore(std::shared_ptr<Pipeline> _data_store){
        data_store_ = _data_store;
    };
    void SetInputModuleIndex(uint _index) { input_module_index_ = _index; };
    void SetOutputModuleIndex(uint _index) { output_module_index_ = _index; };

public:
    std::shared_ptr<Pipeline> data_store_;
    std::string config_path_ = CONFIG_PATH;
    std::atomic<bool> stop_flag_=false;
protected:
    std::thread thread_;
    uint input_module_index_ = 0;
    uint output_module_index_ = 0;
    Timer fps_timer_;
    int sleep_time_=10;
    int pass_count_=0;
    int fail_count_=0;
    int debug_time_=1000;
    std::string name_ = "Module";
};


#endif //MODULE_H
