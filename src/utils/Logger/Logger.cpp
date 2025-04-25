//
// Created by Dongmin on 25. 3. 13.
//
#include "Logger.h"

using namespace DM;

void Logger::Initialize()
{
    JsonObject config_json;
    config_json.load(CONFIG_PATH);

    bool use_log = (bool)config_json.get_int("GoEngine/Log/Use");
    int log_level = config_json.get_int("GoEngine/Log/Level");
    log_path_ = config_json.get_string("GoEngine/Log/Path");
    if (!use_log) return;

    verbosity = log_level;
    if(!FileSystem::exist(log_path_)) FileSystem::make_folder(log_path_);
    log_queue = std::make_shared<tbb::concurrent_queue<std::string>>();
    timer.now();
    log_path_ = log_path_ + "/" + timer.get_date_string();
    Log("Logger::Initialize()", LOGLEVEL::INFO);
    Log(log_path_, LOGLEVEL::INFO);
    StartThread();
}

void Logger::Release()
{
    // std::cout << "Logger::Release()" << std::endl;
    Log("Logger::Release()", LOGLEVEL::INFO);
    bool expect = true;
    if (running_.compare_exchange_strong(expect, false))
    {
        thread_.join();
        StopThread();
        log_queue.reset();
    }
}

void Logger::StartThread() {
    {
        std::string message = "";
        timer.now();
        message.append(timer.get_log_string());
        message.append("==================================\n");
        message.append(timer.get_log_string());
        message.append("GO LOGGER START\n");
        FileSystem::write_all_text(log_path_, message);
    }
    bool expect = false;
    if (running_.compare_exchange_strong(expect, true))
    {
        Log("Logger::StartThread()", LOGLEVEL::INFO);
        thread_ = std::thread(&Logger::RunThread, this);
    }
}

void Logger::RunThread() {
    while(running_) {
        std::string message;
        if(log_queue->try_pop(message))
        {
            FileSystem::write_all_text(log_path_, message);
        }
        else std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void Logger::StopThread() {
    // thread end
    while(!log_queue->empty())
    {
        std::string message;
        log_queue->try_pop(message);
        FileSystem::write_all_text(log_path_, message);
    }
    {
        std::string message = "";
        timer.now();
        message.append(timer.get_log_string());
        message.append("GO LOGGER STOP\n");
        message.append(timer.get_log_string());
        message.append("==================================\n");
        FileSystem::write_all_text(log_path_, message);
    }
}

void Logger::Log(std::string message, LOGLEVEL level)
{
    if(running_.load())
    {
        if(level < verbosity) return;
        std::string log_message = "";
        timer.now();
        log_message.append(timer.get_log_string());
        log_message.append(log_levels[level]);
        log_message.append(message);
        log_message.append("\n");
        std::cout << log_message;
        log_queue->push(log_message);
    }
}

void Logger::Run() {
    // thread loop
    while(running_) {
        std::string message;
        if(log_queue->try_pop(message))
        {
            FileSystem::write_all_text(log_path_, message);
        }
        else std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    // thread end
    while(!log_queue->empty())
    {
        std::string message;
        log_queue->try_pop(message);
        FileSystem::write_all_text(log_path_, message);
    }
    {
        std::string message = "";
        timer.now();
        message.append(timer.get_log_string());
        message.append("GoLogger stopped\n");
        message.append("==================================\n");
        FileSystem::write_all_text(log_path_, message);
    }
}