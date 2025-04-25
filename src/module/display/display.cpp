//
// Created by Dongmin on 25. 4. 24.
//

#include "display.h"

bool Display::Initialize() {
    name_ = "Display";
    DM::Logger::GetInstance().Log(name_ + "::Initialize()", LOGLEVEL::INFO);
    JsonObject config_json;
    if(FileSystem::exist(config_path_)) {
        config_json.load(config_path_);
    } else assert(false);
    debug_time_ = config_json.get_int("GoEngine/Log/DebugTime") * 1000;
    sleep_time_ = config_json.get_int("GoEngine/Log/SleepTime");
    int frameWidth = config_json.get_int("GoEngine/Stream/Width");
    int frameHeight = config_json.get_int("GoEngine/Stream/Height");
    int windowWidth = config_json.get_int("GoEngine/Window/Width");
    int windowHeight = config_json.get_int("GoEngine/Window/Height");
    frame_texture_id_ = 0;
    glGenTextures(1, &frame_texture_id_);
    glBindTexture(GL_TEXTURE_2D, frame_texture_id_);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    return true;
}

void Display::Release() {
    DM::Logger::GetInstance().Log(name_ + "::Release()", LOGLEVEL::INFO);
}

void Display::Run() {
    uint index;
    if (data_store_->module_index_queues_[input_module_index_].try_pop(index)) {
        auto container = data_store_->contaiers_[index];
        auto org_image = container->org_image_;

        glBindTexture(GL_TEXTURE_2D, frame_texture_id_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, org_image->cols, org_image->rows, 0,
                  GL_BGR, GL_UNSIGNED_BYTE, (const GLvoid*)org_image->data);

        const ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);

        static ImGuiWindowFlags main_window_flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus;
        ImGui::Begin("Display", nullptr, main_window_flags);
        ImGui::Image((ImTextureID)static_cast<uintptr_t>(frame_texture_id_), viewport->Size);
        ImGui::End();

        ++pass_count_;
        data_store_->module_index_queues_[output_module_index_].push(index);
    }
    else {
        ++fail_count_;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_));
    }
}