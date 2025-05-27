//
// Created by Dongmin on 25. 4. 24.
//

#ifndef APPLICATION_H
#define APPLICATION_H

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <imgui/imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_glfw.h>
#include "module/module.h"
class Application : public Module {
public:
    Application() {};
    ~Application() {};
    bool Initialize() override;
    void Release() override;
    void Run() override;

    void RunThread() override;

public:
    GLFWwindow* window_;
    GLuint video_texture_;
    GLuint menu_texture_;
    int imgui_radio_button_ = 0;
    int imgui_fps_slider_ = 30;
    cv::Mat sub_image_;
    bool paused_ = false;
    uint index_;
};



#endif //APPLICATION_H
