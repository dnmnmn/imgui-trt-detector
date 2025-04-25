//
// Created by Dongmin on 24. 9. 19.
//

#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <tbb/concurrent_queue.h>
#include <thread>
#include <utils/FileSystem/FileSystem.h>
#include <utils/Json/JsonObject.h>
#include <utils/DateTime/DateTime.h>
#define CONFIG_PATH "/home/gopizza/data/config/mas.json"

enum LOGLEVEL
{
    DEBUG = 0,
    INFO,
    WARN,
    ERROR,
    FATAL
};
namespace DM{
    class Logger {
    private:
        Logger () { log_queue = nullptr; }
        Logger (const Logger& other);
        ~Logger() {};
    public:
        static Logger& GetInstance()
        {
            static Logger instance;
            return instance;
        }
        void Initialize();
        void Release();

        // Thread
        void StartThread();
        void RunThread();
        void StopThread();
        void Log(std::string message, LOGLEVEL level=DEBUG);

    private:
        void Run();
        std::shared_ptr<tbb::concurrent_queue<std::string>> log_queue = nullptr;
        std::thread thread_;
        std::atomic<bool> running_ = false;
        DateTime timer;
        std::vector<string> log_levels = {"[DEBUG] ", "[INFO]  ", "[WARN]  ", "[ERROR] ", "[FATAL] "};
        int verbosity = 2;
        std::string log_path_ = "";
    };
}



#endif //LOGGER_H
