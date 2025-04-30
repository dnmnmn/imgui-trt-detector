//
// Created by Dongmin on 25. 4. 24.
//

#define GL_CALL(_CALL)      _CALL   // Call without error check

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

    if (!glfwInit())
        return false;
    window_ = glfwCreateWindow(1280, 720, "GLFW Test", nullptr, nullptr);
    if (window_ == nullptr){
        glfwTerminate();
        return false;
    }
    return true;
}

void Display::Release() {
    DM::Logger::GetInstance().Log(name_ + "::Release()", LOGLEVEL::INFO);
    glfwDestroyWindow(window_);
    glfwTerminate();
}

void Display::RunThread() {
    fps_timer_.start();


    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1); // Enable vsync
    frame_texture_id_ = 0;
    glGenTextures(1, &frame_texture_id_);
    glBindTexture(GL_TEXTURE_2D, frame_texture_id_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // 텍스처 래핑 모드 설정 (선택 사항)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
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

void Display::Run() {
    uint index;
    if (data_store_->module_index_queues_[input_module_index_].try_pop(index)) {
        auto container = data_store_->contaiers_[index];
        auto org_image = container->org_image_;
        if (!glfwWindowShouldClose(window_)) {
            glBindTexture(GL_TEXTURE_2D, frame_texture_id_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, org_image->cols, org_image->rows, 0, GL_BGR, GL_UNSIGNED_BYTE, org_image->data);

            glClear(GL_COLOR_BUFFER_BIT);

            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, frame_texture_id_);

            glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
            glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f, -1.0f);
            glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f,  1.0f);
            glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f,  1.0f);
            glEnd();

            glDisable(GL_TEXTURE_2D);

            glfwSwapBuffers(window_);
        }
        else
            stop_flag_ = true;
        glfwPollEvents();
        ++pass_count_;
        data_store_->module_index_queues_[output_module_index_].push(index);
    }
    else {
        ++fail_count_;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_));
    }
}