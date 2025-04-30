//
// Created by gopizza on 25. 4. 24.
//

#include "application.h"



bool Application::Initialize() {
    name_ = "Application";
    DM::Logger::GetInstance().Log(name_ + "::Initialize()", LOGLEVEL::INFO);
    // Config
    JsonObject config_json;
    config_json.load(config_path_);
    debug_time_ = config_json.get_int("GoEngine/Log/DebugTime") * 1000;
    // Create window with graphics context

    if (!glfwInit())
        return false;
    window_ = glfwCreateWindow(1280, 720, "Dear ImGui GLFW", nullptr, nullptr);
    if (window_ == nullptr){
        glfwTerminate();
        return false;
    }
    return true;
}

void Application::Release() {
    DM::Logger::GetInstance().Log(name_ + "::Release()", LOGLEVEL::INFO);
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window_);
    glfwTerminate();
}

void Application::RunThread() {
    fps_timer_.start();

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1); // Enable vsync

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    //
    // // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init();

    video_texture_ = 0;
    glGenTextures(1, &video_texture_);
    glBindTexture(GL_TEXTURE_2D, video_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // 텍스처 래핑 모드 설정 (선택 사항)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    menu_texture_ = 0;
    glGenTextures(1, &menu_texture_);
    glBindTexture(GL_TEXTURE_2D, menu_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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
}


void Application::Run() {
    // Main loop
    uint index;
    if (data_store_->module_index_queues_[input_module_index_].try_pop(index)) {
        auto container = data_store_->contaiers_[index];
        auto org_image = container->org_image_;
        if (!glfwWindowShouldClose(window_)) {
            glfwPollEvents();

            // Start the Dear ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            glBindTexture(GL_TEXTURE_2D, video_texture_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, org_image->cols, org_image->rows, 0, GL_BGR, GL_UNSIGNED_BYTE, org_image->data);

            ImGui::Begin("Video Stream");
            ImGui::RadioButton("None", &imgui_radio_button_, 0);
            ImGui::RadioButton("Menu", &imgui_radio_button_, 1);
            ImGui::RadioButton("Segmentation", &imgui_radio_button_, 2);
            ImGui::RadioButton("Remove", &imgui_radio_button_, 3);
            ImGui::Image(ImTextureID(video_texture_), ImVec2(org_image->cols, org_image->rows));
            ImGui::End();
            if (imgui_radio_button_ == 1) {
                // crop image
                data_store_->tracker_box_mutex_.lock();
                Bbox bbox = data_store_->tracker_box_;
                data_store_->tracker_box_mutex_.unlock();
                cv::Mat cropped_image(*org_image,cv::Rect(bbox.x1_ * org_image->cols, bbox.y1_ * org_image->rows, bbox.w_ * org_image->cols, bbox.h_ * org_image->rows));
                cv::resize(cropped_image, sub_image_, cv::Size(container->color_mask_.cols, container->color_mask_.rows));
                glBindTexture(GL_TEXTURE_2D, menu_texture_);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, container->color_mask_.cols, container->color_mask_.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, sub_image_.data);

                ImGui::Begin("Menu Stream");
                ImGui::Image(ImTextureID(menu_texture_), ImVec2(container->color_mask_.cols, container->color_mask_.rows));
                ImGui::End();
            }
            else if (imgui_radio_button_ == 2) {
                glBindTexture(GL_TEXTURE_2D, menu_texture_);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, container->color_mask_.cols, container->color_mask_.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, container->color_mask_.data);

                ImGui::Begin("Segment Stream");
                ImGui::Image(ImTextureID(menu_texture_), ImVec2(container->color_mask_.cols, container->color_mask_.rows));
                ImGui::End();
            }
            else if (imgui_radio_button_ == 3) {
                cv::resize(container->filtered_image_, sub_image_, cv::Size(container->color_mask_.cols, container->color_mask_.rows));
                glBindTexture(GL_TEXTURE_2D, menu_texture_);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, container->color_mask_.cols, container->color_mask_.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, sub_image_.data);

                ImGui::Begin("Remove Stream");
                ImGui::Image(ImTextureID(menu_texture_), ImVec2(container->color_mask_.cols, container->color_mask_.rows));
                ImGui::End();
            }
        }
        else
            stop_flag_ = true;
        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window_, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window_);

        ++pass_count_;
        data_store_->module_index_queues_[output_module_index_].push(index);
    }
    else {
        ++fail_count_;
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time_));
    }
}