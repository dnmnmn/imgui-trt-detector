//
// Created by gopizza on 25. 3. 12.
//

#include "postprocess.h"

#include <utils/Logger/Logger.h>

void PostProcess::Initialize() {
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
}

void PostProcess::Release() {
    DM::Logger::GetInstance().Log(__PRETTY_FUNCTION__, LOGLEVEL::INFO);
}
