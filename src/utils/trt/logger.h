//
// Created by Dongmin on 25. 3. 12.
//

#ifndef TRT_LOGGER_H
#define TRT_LOGGER_H

#include <iostream>

namespace dm_trt{
    class Logger : public nvinfer1::ILogger
    {
        void log(Severity severity, const char* msg) noexcept override
        {
            // suppress info-level messages
            if (severity <= Severity::kWARNING)
                std::cout << msg << std::endl;
        }
    };
}

#endif // TRT_LOGGER_H
